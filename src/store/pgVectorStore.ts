import { Pool, PoolClient } from "pg";
import { EmbeddingStore } from "./embedding-store";
import { LlamaEmbedding } from "node-llama-cpp";
import { PostgresConfig } from "./pgDaemon";
import { PgCatalogStore, type CatalogStore } from "./catalogs";

// PgVector store using PostgreSQL with pgvector extension
export class PgVectorStore implements EmbeddingStore {
  private pool: Pool | null = null;
  private tableName = "embeddings";
  private dimension = 384;
  isReady = false;
  isInMemory = false;
  catalogStore: CatalogStore | null = null;

  async initialize(config: PostgresConfig, dimension?: number): Promise<boolean> {
    if (dimension) this.dimension = dimension;
    try {
      // Get PostgreSQL connection details from environment or use defaults
      console.log(
        `📦 Connecting to PostgreSQL at ${config.host}:${config.port}...`,
      );

      // Test connection
      const getClient = async (): Promise<PoolClient> => {
        try {
          // Create connection pool
          this.pool = new Pool({
            host: config.host || "localhost",
            port: config.port || 5432,
            user: config.user || "postgres",
            password: config.password || "postgres",
            database: config.database || "embeddings",
          });
          return await this.pool.connect();
        } catch (e) {
          if ((e as Error).message.includes("database " + '"' + config.database + '"' + ' does not exist')) {
            const pool = new Pool({
              ...config,
              database: "postgres"
            })
            const client = await pool.connect()
            await client.query("CREATE DATABASE " + config.database + ";")
            return await getClient()
          } else {
            throw e
          }
        }
      }
      const client = await getClient()

      // Check if pgvector extension is available
      const extensionResult = await client.query(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')",
      );

      if (!extensionResult.rows[0].exists) {
        console.log("🔧 Creating pgvector extension...");
        await client.query("CREATE EXTENSION IF NOT EXISTS vector");
      }

      // Check if table exists with a mismatched dimension
      const tableInfo = await client.query(
        `SELECT pg_catalog.format_type(atttypid, atttypmod) AS column_type
         FROM pg_catalog.pg_attribute
         WHERE attrelid = (SELECT oid FROM pg_catalog.pg_class WHERE relname = $1)
           AND attname = 'embedding'
           AND attnum > 0
           AND NOT attisdropped`,
        [this.tableName],
      );

      if (tableInfo.rows.length > 0) {
        const colType = tableInfo.rows[0].column_type;
        const match = colType.match(/vector\((\d+)\)/);
        if (match && parseInt(match[1]) !== this.dimension) {
          console.warn(
            `⚠ Existing table has vector(${match[1]}) but model produces ${this.dimension}d embeddings`,
          );
          console.warn(
            `  Drop the table or run: ALTER TABLE ${this.tableName} ALTER COLUMN embedding TYPE vector(${this.dimension});`,
          );
        }
      }

      // Create embeddings table if it doesn't exist
      await client.query(`
                CREATE TABLE IF NOT EXISTS ${this.tableName} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector(${this.dimension}),
                    catalog_id UUID,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            `);

      // Backward compatibility: older schemas had a UNIQUE constraint on text
      // which breaks catalog usage (the same chunk can live in several catalogs).
      await client.query(
        `ALTER TABLE ${this.tableName} DROP CONSTRAINT IF EXISTS embeddings_text_key`,
      );

      // Create index for vector similarity search (for faster queries)
      await client.query(`
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
                ON ${this.tableName} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            `);

      // Catalog schema (catalogs table + catalog_id column + index)
      this.catalogStore = new PgCatalogStore(this.pool);
      await this.catalogStore.ensureSchema();

      client.release();

      this.isReady = true;
      console.log(`✓ PostgreSQL connection established (pgvector ready)`);
      console.log(`✓ Table '${this.tableName}' ready for embeddings`);
      return true;
    } catch (error) {
      const msg = (error as Error).message;
      console.warn(`⚠ PostgreSQL initialization failed: ${msg}`);
      if (msg.includes('extension "vector" is not available')) {
        console.warn(
          "  The pgvector extension is not installed for this PostgreSQL build.\n"
          + "  Options:\n"
          + "   1. Install it:  sudo apt install postgresql-16-pgvector\n"
          + "   2. Use the bundled Docker setup (pgvector image):\n"
          + "        docker compose up -d   # then set POSTGRES_HOST/PORT in .env",
        );
      }
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
    metadata?: Record<string, any>,
    catalogId?: string | null,
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

          const mergedMetadata = {
            source: "pdf",
            timestamp: new Date().toISOString(),
            length: text.length,
            ...metadata,
          };

          // Skip if this exact chunk already exists in the target catalog
          const existing = await client.query(
            `SELECT 1 FROM ${this.tableName}
              WHERE text = $1 AND catalog_id IS ${catalogId ? '= $2' : 'NULL'}`,
            catalogId ? [text, catalogId] : [text],
          );
          if (existing.rows.length > 0) {
            skipCount++;
            continue;
          }

          await client.query(
            `INSERT INTO ${this.tableName} (text, embedding, catalog_id, metadata)
             VALUES ($1, $2::vector, $3, $4::jsonb)`,
            [text, vector, catalogId ?? null, JSON.stringify(mergedMetadata)],
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

  async getEmbeddings(
    _queryText: string,
    limit: number,
    catalogId?: string | null,
  ): Promise<string[]> {
    if (!this.isReady || !this.pool) {
      throw new Error("PostgreSQL connection not initialized");
    }

    try {
      const client = await this.pool.connect();

      try {
        // Query for similar embeddings using cosine distance
        // Note: We use text similarity as a placeholder since we don't have query embedding here
        // The actual semantic ranking happens in pdf-embeddings.ts using findSimilarDocuments
        const catalogClause = catalogId ? "WHERE catalog_id = $2" : "";
        const params = catalogId ? [limit, catalogId] : [limit];
        const result = await client.query(
          `SELECT text FROM ${this.tableName}
             ${catalogClause}
             ORDER BY created_at DESC
             LIMIT $1`,
          params,
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

  async getAllEmbeddings(catalogId?: string | null): Promise<string[]> {
    if (!this.isReady || !this.pool) {
      throw new Error("PostgreSQL connection not initialized");
    }

    try {
      const client = await this.pool.connect();

      try {
        // Get all embeddings ordered by creation date
        const catalogClause = catalogId ? "WHERE catalog_id = $1" : "";
        const params = catalogId ? [catalogId] : [];
        const result = await client.query(
          `SELECT text FROM ${this.tableName}
             ${catalogClause}
             ORDER BY created_at ASC`,
          params,
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
    catalogId?: string | null,
  ): Promise<Array<{ text: string; similarity: number }>> {
    if (!this.isReady || !this.pool) {
      throw new Error("PostgreSQL connection not initialized");
    }

    try {
      const client = await this.pool.connect();

      try {
        const vectorString = `[${embedding.join(",")}]`;
        const catalogClause = catalogId ? "WHERE catalog_id = $3" : "";
        const params = catalogId ? [vectorString, limit, catalogId] : [vectorString, limit];

        // Query using cosine similarity (<-> operator)
        const result = await client.query(
          `SELECT text, 1 - (embedding <-> $1::vector) as similarity
             FROM ${this.tableName}
             ${catalogClause}
             ORDER BY similarity DESC
             LIMIT $2`,
          params,
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

  async clear(catalogId?: string | null): Promise<void> {
    if (this.isReady && this.pool) {
      try {
        const client = await this.pool.connect();
        try {
          if (catalogId) {
            await client.query(
              `DELETE FROM ${this.tableName} WHERE catalog_id = $1`,
              [catalogId],
            );
          } else {
            await client.query(`DELETE FROM ${this.tableName}`);
          }
          console.log(
            catalogId
              ? `✓ Embeddings cleared for catalog ${catalogId}`
              : "✓ PostgreSQL embeddings table cleared",
          );
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
