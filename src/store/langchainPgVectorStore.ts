import { PGVectorStore as LangchainPGVStore, DistanceStrategy } from "@langchain/community/vectorstores/pgvector";
import { EmbeddingsInterface } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { EmbeddingStore } from "./embedding-store";
import { LlamaEmbedding } from "node-llama-cpp";
import { PostgresConfig } from "./pgDaemon";
import { Pool } from "pg";

class DummyEmbeddings implements EmbeddingsInterface {
  async embedDocuments(_documents: string[]): Promise<number[][]> {
    throw new Error(
      "Direct embedding not supported - use addEmbeddings with pre-computed vectors",
    );
  }
  async embedQuery(_document: string): Promise<number[]> {
    throw new Error(
      "Direct embedding not supported - use queryByEmbedding with pre-computed vectors",
    );
  }
}

export class LangchainPgVectorStore implements EmbeddingStore {
  private store: LangchainPGVStore | null = null;
  private tableName = "langchain_embeddings";
  private dimension = 384;
  isReady = false;
  isInMemory = false;

  async initialize(config: PostgresConfig, dimension?: number): Promise<boolean> {
    if (dimension) this.dimension = dimension;
    try {
      console.log(
        `📦 Initializing LangChain PGVectorStore at ${config.host}:${config.port}...`,
      );

      const getClient = async (): Promise<Pool> => {
        try {
          const pool = new Pool({
            host: config.host || "localhost",
            port: config.port || 5432,
            user: config.user || "postgres",
            password: config.password || "postgres",
            database: config.database || "embeddings",
          });
          await pool.connect().then((c) => c.release());
          return pool;
        } catch (e) {
          if (
            (e as Error).message.includes(
              'database "' + config.database + '" does not exist',
            )
          ) {
            const pool = new Pool({ ...config, database: "postgres" });
            const client = await pool.connect();
            await client.query("CREATE DATABASE " + config.database + ";");
            client.release();
            await pool.end();
            return getClient();
          }
          throw e;
        }
      };

      const pool = await getClient();
      await pool.end();

      const embeddings = new DummyEmbeddings();
      this.store = new LangchainPGVStore(embeddings, {
        postgresConnectionOptions: {
          host: config.host || "localhost",
          port: config.port || 5432,
          user: config.user || "postgres",
          password: config.password || "postgres",
          database: config.database || "embeddings",
        },
        tableName: this.tableName,
        distanceStrategy: "cosine" as DistanceStrategy,
        columns: {
          idColumnName: "id",
          vectorColumnName: "embedding",
          contentColumnName: "text",
          metadataColumnName: "metadata",
        },
      });

      await this.store.ensureTableInDatabase(this.dimension);

      this.isReady = true;
      console.log(`✓ LangChain PGVectorStore initialized (${this.dimension}d)`);
      return true;
    } catch (error) {
      console.warn(
        `⚠ LangChain PGVectorStore initialization failed: ${(error as Error).message}`,
      );
      return false;
    }
  }

  async addEmbeddings(
    _chunks: string[],
    embeddings: Map<string, LlamaEmbedding>,
    metadata?: Record<string, any>,
  ): Promise<void> {
    if (!this.isReady || !this.store) {
      throw new Error("LangChain PGVectorStore not initialized");
    }

    const documents: Document[] = [];
    const vectors: number[][] = [];

    for (const [text, embedding] of embeddings) {
      documents.push(
        new Document({
          pageContent: text,
          metadata: {
            ...(metadata || {}),
            source: "pdf",
            timestamp: new Date().toISOString(),
            length: text.length,
          },
        }),
      );
      vectors.push(embedding.vector ? Array.from(embedding.vector) : []);
    }

    try {
      await this.store.addVectors(vectors, documents);
      console.log(`✓ Stored ${documents.length} embeddings via LangChain PGVectorStore`);
    } catch (error) {
      console.warn(
        `⚠ Failed to store embeddings: ${(error as Error).message}`,
      );
      throw error;
    }
  }

  async getEmbeddings(_query: string, limit: number): Promise<string[]> {
    if (!this.isReady || !this.store) {
      throw new Error("LangChain PGVectorStore not initialized");
    }

    const pool = this.store.pool;
    try {
      const result = await pool.query(
        `SELECT "text" FROM ${this.store.computedTableName}
         ORDER BY "id" DESC
         LIMIT $1`,
        [limit],
      );
      return result.rows.map((row) => row.text);
    } catch (error) {
      console.warn(`⚠ Query failed: ${(error as Error).message}`);
      throw error;
    }
  }

  async getAllEmbeddings(): Promise<string[]> {
    if (!this.isReady || !this.store) {
      throw new Error("LangChain PGVectorStore not initialized");
    }

    const pool = this.store.pool;
    try {
      const result = await pool.query(
        `SELECT "text" FROM ${this.store.computedTableName}
         ORDER BY "id" ASC`,
      );
      return result.rows.map((row) => row.text);
    } catch (error) {
      console.warn(`⚠ Failed to get all embeddings: ${(error as Error).message}`);
      throw error;
    }
  }

  async queryByEmbedding(
    embedding: number[],
    limit: number,
  ): Promise<Array<{ text: string; similarity: number }>> {
    if (!this.isReady || !this.store) {
      throw new Error("LangChain PGVectorStore not initialized");
    }

    try {
      const results = await this.store.similaritySearchVectorWithScore(
        embedding,
        limit,
      );
      return results.map(([doc, score]) => ({
        text: doc.pageContent,
        similarity: score,
      }));
    } catch (error) {
      console.warn(`⚠ Vector query failed: ${(error as Error).message}`);
      return [];
    }
  }

  async clear(): Promise<void> {
    if (!this.isReady || !this.store) return;

    const pool = this.store.pool;
    try {
      await pool.query(`DELETE FROM ${this.store.computedTableName}`);
      console.log("✓ LangChain PGVectorStore embeddings cleared");
    } catch (error) {
      console.warn(
        `⚠ Failed to clear embeddings: ${(error as Error).message}`,
      );
    }
  }

  async close(): Promise<void> {
    if (this.store) {
      await this.store.end();
    }
  }
}
