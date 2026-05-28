import { LlamaEmbedding } from "node-llama-cpp";
import { InMemoryStore } from "./inMemoryStore";
import { PgVectorStore } from "./pgVectorStore";
import { PostgresConfig, PostgresDaemon, PostgresServer } from "./pgDaemon";
import dotenv from "dotenv";

dotenv.config();

export interface EmbeddingStore {
  isReady: boolean;
  isInMemory: boolean;
  addEmbeddings(
    chunks: string[],
    embeddings: Map<string, LlamaEmbedding>,
    metadata?: Record<string, any>,
  ): Promise<void>;
  getEmbeddings(query: string, limit: number): Promise<string[]>;
  getAllEmbeddings(): Promise<string[]>;
  clear(): Promise<void>;
  queryByEmbedding(
    embedding: number[],
    limit: number,
  ): Promise<Array<{ text: string; similarity: number }>>;
}

let postgresDaemon: Awaited<PostgresServer> | null = null;

// Export a function to get the postgres daemon reference
export function getPostgresDaemon(): Awaited<PostgresServer> | null {
  return postgresDaemon;
}

// Export a function to explicitly set the postgres daemon reference
export function setPostgresDaemon(
  daemon: Awaited<PostgresServer> | null,
): void {
  postgresDaemon = daemon;
}

export async function createEmbeddingStore(): Promise<EmbeddingStore> {
  // Try to initialize pgvector store
  try {
    console.log("📦 Initializing PostgreSQL store...");
    const config: PostgresConfig = {
      user: process.env["POSTGRES_USER"] || "postgres",
      password: process.env["POSTGRES_PASSWORD"] || "postgres",
      database: process.env["POSTGRES_DB"] || "embeddings",
      port: parseInt(process.env["POSTGRES_PORT"] || "5432"),
      host: process.env["POSTGRES_HOST"] || "localhost",
      dataDir: process.env["POSTGRES_DATA_DIR"] as string,
    };
    if (!getPostgresDaemon() && config.host == "127.0.0.1") {
      const daemon = await new PostgresDaemon(config).startServer();
      setPostgresDaemon(daemon);
    }
    const pgvectorStore = new PgVectorStore();
    const initialized = await pgvectorStore.initialize(config);

    if (initialized) {
      return pgvectorStore;
    }
    throw new Error("Failed to initialize PostgreSQL store");
  } catch (error) {
    // If we have a daemon that was created but failed to initialize, stop it
    if (postgresDaemon) {
      try {
        await postgresDaemon.stop();
      } catch (stopError) {
        console.error("Error stopping daemon during cleanup:", stopError);
      }
      setPostgresDaemon(null);
    }
    console.warn("⚠ Using in-memory store");
    return new InMemoryStore();
  }
}

export async function storeEmbeddingsWithFallback(
  store: EmbeddingStore,
  chunks: string[],
  embeddings: Map<string, LlamaEmbedding>,
): Promise<void> {
  try {
    await store.addEmbeddings(chunks, embeddings);
  } catch (error) {
    if (!store.isInMemory) {
      console.warn("⚠ Failed to store in PostgreSQL, using in-memory fallback");
      const memoryStore = new InMemoryStore();
      await memoryStore.addEmbeddings(chunks, embeddings);
    } else {
      throw error;
    }
  }
}

export async function queryEmbeddingsWithFallback(
  store: EmbeddingStore,
  query: string,
  limit: number,
): Promise<string[]> {
  try {
    return await store.getEmbeddings(query, limit);
  } catch (error) {
    console.warn(
      `⚠ Query failed on ${store.isInMemory ? "in-memory" : "PostgreSQL"} store`,
    );
    throw error;
  }
}

// Helper function to get all embeddings
export async function getAllEmbeddingsWithFallback(
  store: EmbeddingStore,
): Promise<string[]> {
  try {
    return await store.getAllEmbeddings();
  } catch (error) {
    console.warn(
      `⚠ Failed to get all embeddings from ${store.isInMemory ? "in-memory" : "PostgreSQL"} store`,
    );
    throw error;
  }
}
