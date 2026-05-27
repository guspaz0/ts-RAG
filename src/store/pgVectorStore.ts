import { Pool } from "pg";
import { EmbeddingStore } from "./embedding-store";
import { LlamaEmbedding } from "node-llama-cpp";
import { PostgresConfig } from "./pgDaemon";

// PgVector store using PostgreSQL with pgvector extension
export class PgVectorStore implements EmbeddingStore {
  private pool: Pool | null = null;
  private tableName = "embeddings";
  isReady = false;
  isInMemory = false;

  async initialize(config: PostgresConfig): Promise<boolean> {
    try {
      // Get PostgreSQL connection details from environment or use defaults
      console.log(
        `📦 Connecting to PostgreSQL at ${config.host}:${config.port}...`,
      );

      // Create connection pool
      this.pool = new Pool({
        host: config.host || "localhost",
        port: config.port || 5432,
        user: config.user || "postgres",
        password: config.password || "postgres",
        database: config.database || "embeddings",
      });

      // Test connection
      const client = await this.pool.connect();

      // Check if pgvector extension is available
      const extensionResult = await client.query(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')",
      );

      if (!extensionResult.rows[0].exists) {
        console.log("🔧 Creating pgvector extension...");
        await client.query("CREATE EXTENSION IF NOT EXISTS vector");
      }

      // Create embeddings table if it doesn't exist
      await client.query(`
                CREATE TABLE IF NOT EXISTS ${this.tableName} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL UNIQUE,
                    embedding vector(384),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            `);

      // Create index for vector similarity search (for faster queries)
      await client.query(`
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
                ON ${this.tableName} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            `);

      client.release();

      this.isReady = true;
      console.log(`✓ PostgreSQL connection established (pgvector ready)`);
      console.log(`✓ Table '${this.tableName}' ready for embeddings`);
      return true;
    } catch (error) {
      console.warn(
        `⚠ PostgreSQL initialization failed: ${(error as Error).message}`,
      );
      console.warn(
        `  Connection string: ${process.env["PG_HOST"] || "localhost"}:${process.env["PG_PORT"] || 5432}`,
      );
      console.warn("  Falling back to in-memory storage");
      if (this.pool) {
        await this.pool.end();
        this.pool = null;
      }
      return false;
    }
  }

  async addEmbeddings(
    _chunks: string[],
    embeddings: Map<string, LlamaEmbedding>,
  ): Promise<void> {
    if (!this.isReady || !this.pool) {
      throw new Error("PostgreSQL connection not initialized");
    }

    const client = await this.pool.connect();

    try {
      let successCount = 0;
      let skipCount = 0;

      for (const [text, embedding] of embeddings) {
        try {
          // Convert embedding to vector array
          const vector = embedding.vector
            ? `[${Array.from(embedding.vector).join(",")}]`
            : null;

          const metadata = {
            source: "pdf",
            timestamp: new Date().toISOString(),
            length: text.length,
          };

          // Use INSERT ... ON CONFLICT to handle duplicates
          await client.query(
            `INSERT INTO ${this.tableName} (text, embedding, metadata) 
                            VALUES ($1, $2::vector, $3::jsonb)
                            ON CONFLICT (text) DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata`,
            [text, vector, JSON.stringify(metadata)],
          );

          successCount++;
        } catch (error) {
          // Skip duplicate entries
          if ((error as any).code === "23505") {
            skipCount++;
          } else {
            console.warn(
              `⚠ Failed to store embedding: ${(error as Error).message}`,
            );
          }
        }
      }

      console.log(
        `✓ Stored ${successCount} new embeddings in PostgreSQL (${skipCount} duplicates skipped)`,
      );
    } finally {
      client.release();
    }
  }

  async getEmbeddings(_queryText: string, limit: number): Promise<string[]> {
    if (!this.isReady || !this.pool) {
      throw new Error("PostgreSQL connection not initialized");
    }

    try {
      const client = await this.pool.connect();

      try {
        // Query for similar embeddings using cosine distance
        // Note: We use text similarity as a placeholder since we don't have query embedding here
        // The actual semantic ranking happens in pdf-embeddings.ts using findSimilarDocuments
        const result = await client.query(
          `SELECT text FROM ${this.tableName}
                     ORDER BY created_at DESC
                     LIMIT $1`,
          [limit],
        );

        return result.rows.map((row) => row.text);
      } finally {
        client.release();
      }
    } catch (error) {
      console.warn(`⚠ Query failed: ${(error as Error).message}`);
      throw error;
    }
  }

  async getAllEmbeddings(): Promise<string[]> {
    if (!this.isReady || !this.pool) {
      throw new Error("PostgreSQL connection not initialized");
    }

    try {
      const client = await this.pool.connect();

      try {
        // Get all embeddings ordered by creation date
        const result = await client.query(
          `SELECT text FROM ${this.tableName}
                     ORDER BY created_at ASC`,
        );

        return result.rows.map((row) => row.text);
      } finally {
        client.release();
      }
    } catch (error) {
      console.warn(`⚠ Query failed: ${(error as Error).message}`);
      throw error;
    }
  }

  async queryByEmbedding(
    embedding: number[],
    limit: number,
  ): Promise<Array<{ text: string; similarity: number }>> {
    if (!this.isReady || !this.pool) {
      throw new Error("PostgreSQL connection not initialized");
    }

    try {
      const client = await this.pool.connect();

      try {
        const vectorString = `[${embedding.join(",")}]`;

        // Query using cosine similarity (<-> operator)
        const result = await client.query(
          `SELECT text, 1 - (embedding <-> $1::vector) as similarity
                     FROM ${this.tableName}
                     ORDER BY similarity DESC
                     LIMIT $2`,
          [vectorString, limit],
        );

        return result.rows.map((row) => ({
          text: row.text,
          similarity: row.similarity,
        }));
      } finally {
        client.release();
      }
    } catch (error) {
      console.warn(`⚠ Vector query failed: ${(error as Error).message}`);
      // In case of error, return empty array so fallback can happen
      return [];
    }
  }

  async clear(): Promise<void> {
    if (this.isReady && this.pool) {
      try {
        const client = await this.pool.connect();
        try {
          await client.query(`DELETE FROM ${this.tableName}`);
          console.log("✓ PostgreSQL embeddings table cleared");
        } finally {
          client.release();
        }
      } catch (error) {
        console.warn(
          `⚠ Failed to clear embeddings: ${(error as Error).message}`,
        );
      }
    }
  }

  async close(): Promise<void> {
    if (this.pool) {
      await this.pool.end();
      this.pool = null;
    }
  }
}
