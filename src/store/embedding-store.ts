import { LlamaEmbedding } from "node-llama-cpp";
import { InMemoryStore } from "./inMemoryStore";
import { PgVectorStore } from "./pgVectorStore";
import { PostgresConfig, PostgresDaemon, PostgresServer } from "./pgDaemon";
import { Pool } from "pg";
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
let daemonStartPromise: Promise<Awaited<PostgresServer>> | null = null;

// Export a function to get the postgres daemon reference
export function getPostgresDaemon(): Awaited<PostgresServer> | null {
  return postgresDaemon;
}

/**
 * Start (or reuse) the embedded PostgreSQL daemon for the given config.
 * The promise is cached so concurrent callers share a single startup attempt.
 */
export function startPostgresDaemon(
  config: PostgresConfig,
): Promise<Awaited<PostgresServer>> {
  if (postgresDaemon) return Promise.resolve(postgresDaemon);
  if (!daemonStartPromise) {
    daemonStartPromise = new PostgresDaemon(config).startServer().then((daemon) => {
      postgresDaemon = daemon;
      return daemon;
    }).catch((error) => {
      daemonStartPromise = null; // allow retry on the next call
      throw error;
    });
  }
  return daemonStartPromise;
}

/**
 * Test a TCP connection to an existing PostgreSQL server (e.g. Docker).
 * Used to decide whether the embedded daemon should be started.
 */
async function isPostgresReachable(config: PostgresConfig): Promise<boolean> {
  const pool = new Pool({
    host: config.host,
    port: config.port,
    user: config.user,
    password: config.password,
    database: "postgres",
    connectionTimeoutMillis: 2000,
  });
  try {
    const client = await pool.connect();
    client.release();
    return true;
  } catch {
    return false;
  } finally {
    await pool.end().catch(() => {});
  }
}

// Export a function to explicitly set the postgres daemon reference
export function setPostgresDaemon(
  daemon: Awaited<PostgresServer> | null,
): void {
  postgresDaemon = daemon;
}

export async function createEmbeddingStore(dimension?: number): Promise<EmbeddingStore> {
  // Try to initialize pgvector store
  try {
    console.log("📦 Initializing PostgreSQL store...");
    const config: PostgresConfig = {
      user: process.env["POSTGRES_USER"] || "postgres",
      password: process.env["POSTGRES_PASSWORD"] || "postgres",
      database: process.env["POSTGRES_DB"] || process.env["POSTGRES_DATABASE"] || "embeddings",
      port: parseInt(process.env["POSTGRES_PORT"] || "5432"),
      host: process.env["POSTGRES_HOST"] || "localhost",
      dataDir: process.env["POSTGRES_DATA_DIR"] as string,
    };

    // Start the embedded PostgreSQL daemon unless a server is already
    // reachable at the configured host:port (e.g. a Docker container).
    if (!getPostgresDaemon()) {
      const reachable = await isPostgresReachable(config);
      if (reachable) {
        console.log(`✓ PostgreSQL already reachable at ${config.host}:${config.port}, skipping embedded daemon`);
      } else {
        console.log(`⏳ No PostgreSQL at ${config.host}:${config.port}, starting embedded daemon...`);
        await startPostgresDaemon(config);
      }
    }

    const pgvectorStore = new PgVectorStore();
    const initialized = await pgvectorStore.initialize(config, dimension);

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
      daemonStartPromise = null;
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
