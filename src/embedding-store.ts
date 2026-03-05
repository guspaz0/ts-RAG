import { LlamaEmbedding } from "node-llama-cpp";
import { Pool } from "pg";

export interface EmbeddingStore {
    isReady: boolean;
    isInMemory: boolean;
    addEmbeddings(chunks: string[], embeddings: Map<string, LlamaEmbedding>): Promise<void>;
    getEmbeddings(query: string, limit: number): Promise<string[]>;
    getAllEmbeddings(): Promise<string[]>;
    clear(): Promise<void>;
}

// In-memory store fallback
class InMemoryStore implements EmbeddingStore {
    isReady = true;
    isInMemory = true;
    private embeddingTexts: Map<string, number[]> = new Map();

    async addEmbeddings(_chunks: string[], embeddings: Map<string, LlamaEmbedding>): Promise<void> {
        console.log("📦 Storing embeddings in memory...");
        
        for (const [text, embedding] of embeddings) {
            // Store embedding vectors
            const vector = embedding.vector ? Array.from(embedding.vector) : [];
            this.embeddingTexts.set(text, vector);
        }
        
        console.log(`✓ Stored ${this.embeddingTexts.size} embeddings in memory`);
    }

    async getEmbeddings(_query: string, limit: number): Promise<string[]> {
        return Array.from(this.embeddingTexts.keys()).slice(0, limit);
    }

    async clear(): Promise<void> {
        this.embeddingTexts.clear();
    }
}

// PgVector store using PostgreSQL with pgvector extension
class PgVectorStore implements EmbeddingStore {
    private pool: Pool | null = null;
    private tableName = "embeddings";
    isReady = false;
    isInMemory = false;

    async initialize(): Promise<boolean> {
        try {
            // Get PostgreSQL connection details from environment or use defaults
            const pgHost = process.env["PG_HOST"] || "localhost";
            const pgPort = parseInt(process.env["PG_PORT"] || "5432");
            const pgUser = process.env["PG_USER"] || "postgres";
            const pgPassword = process.env["POSTGRES_PASSWORD"] || "postgres";
            const pgDatabase = process.env["PG_DATABASE"] || "embeddings";

            console.log(`📦 Connecting to PostgreSQL at ${pgHost}:${pgPort}...`);

            // Create connection pool
            this.pool = new Pool({
                host: pgHost,
                port: pgPort,
                user: pgUser,
                password: pgPassword,
                database: pgDatabase,
            });

            // Test connection
            const client = await this.pool.connect();
            
            // Check if pgvector extension is available
            const extensionResult = await client.query(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
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
            console.warn(`⚠ PostgreSQL initialization failed: ${(error as Error).message}`);
            console.warn(`  Connection string: ${process.env["PG_HOST"] || "localhost"}:${process.env["PG_PORT"] || 5432}`);
            console.warn("  Falling back to in-memory storage");
            if (this.pool) {
                await this.pool.end();
                this.pool = null;
            }
            return false;
        }
    }

    async addEmbeddings(_chunks: string[], embeddings: Map<string, LlamaEmbedding>): Promise<void> {
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
                        [text, vector, JSON.stringify(metadata)]
                    );

                    successCount++;
                } catch (error) {
                    // Skip duplicate entries
                    if ((error as any).code === "23505") {
                        skipCount++;
                    } else {
                        console.warn(`⚠ Failed to store embedding: ${(error as Error).message}`);
                    }
                }
            }

            console.log(`✓ Stored ${successCount} new embeddings in PostgreSQL (${skipCount} duplicates skipped)`);
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
                    [limit]
                );

                return result.rows.map(row => row.text);
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
                     ORDER BY created_at ASC`
                );

                return result.rows.map(row => row.text);
            } finally {
                client.release();
            }
        } catch (error) {
            console.warn(`⚠ Query failed: ${(error as Error).message}`);
            throw error;
        }
    }

    async queryByEmbedding(embedding: number[], limit: number): Promise<Array<{text: string, similarity: number}>> {
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
                    [vectorString, limit]
                );

                return result.rows.map(row => ({
                    text: row.text,
                    similarity: row.similarity
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
                console.warn(`⚠ Failed to clear embeddings: ${(error as Error).message}`);
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

export async function createEmbeddingStore(): Promise<EmbeddingStore> {
    const pgvectorStore = new PgVectorStore();
    const initialized = await pgvectorStore.initialize();

    if (initialized) {
        return pgvectorStore;
    } else {
        // Fallback to in-memory store
        return new InMemoryStore();
    }
}

export async function storeEmbeddingsWithFallback(
    store: EmbeddingStore,
    chunks: string[],
    embeddings: Map<string, LlamaEmbedding>
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
    limit: number
): Promise<string[]> {
    try {
        return await store.getEmbeddings(query, limit);
    } catch (error) {
        console.warn(`⚠ Query failed on ${store.isInMemory ? "in-memory" : "PostgreSQL"} store`);
        throw error;
    }
}

// Helper function to get all embeddings
export async function getAllEmbeddingsWithFallback(
    store: EmbeddingStore
): Promise<string[]> {
    try {
        return await store.getAllEmbeddings();
    } catch (error) {
        console.warn(`⚠ Failed to get all embeddings from ${store.isInMemory ? "in-memory" : "PostgreSQL"} store`);
        throw error;
    }
}

// Export PgVectorStore for advanced use cases
export { PgVectorStore };
