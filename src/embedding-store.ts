import { LlamaEmbedding, LlamaContext } from "node-llama-cpp";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export interface EmbeddingStore {
    isReady: boolean;
    isInMemory: boolean;
    addEmbeddings(chunks: string[], embeddings: Map<string, LlamaEmbedding>): Promise<void>;
    getEmbeddings(query: string, limit: number): Promise<string[]>;
    clear(): Promise<void>;
}

// In-memory store fallback
class InMemoryStore implements EmbeddingStore {
    isReady = true;
    isInMemory = true;
    private embeddingTexts: Map<string, number[]> = new Map();
    private chunks: string[] = [];

    async addEmbeddings(chunks: string[], embeddings: Map<string, LlamaEmbedding>): Promise<void> {
        console.log("📦 Storing embeddings in memory...");
        this.chunks = chunks;
        
        for (const [text, embedding] of embeddings) {
            // Store embedding vectors
            const vector = embedding.vector ? Array.from(embedding.vector) : [];
            this.embeddingTexts.set(text, vector);
        }
        
        console.log(`✓ Stored ${this.embeddingTexts.size} embeddings in memory`);
    }

    async getEmbeddings(query: string, limit: number): Promise<string[]> {
        return Array.from(this.embeddingTexts.keys()).slice(0, limit);
    }

    async clear(): Promise<void> {
        this.embeddingTexts.clear();
        this.chunks = [];
    }
}

// ChromaDB store
class ChromaDBStore implements EmbeddingStore {
    private client: any = null;
    private collection: any = null;
    isReady = false;
    isInMemory = false;

    async initialize(): Promise<boolean> {
        try {
            const { ChromaClient } = await import("chromadb");
            
            // Use SQLite persistent storage
            const dbPath = path.join(__dirname, "..", "data", "embeddings.db");
            
            console.log("📦 Initializing ChromaDB with SQLite...");
            
            // ChromaDB with SQLite backend
            this.client = new ChromaClient({
                path: path.join(__dirname, "..", "data"),
            });

            // Get or create collection
            this.collection = await this.client.getOrCreateCollection({
                name: "pdf_embeddings",
                metadata: { "hnsw:space": "cosine" },
            });

            this.isReady = true;
            console.log(`✓ ChromaDB initialized with SQLite at: ${dbPath}`);
            return true;
        } catch (error) {
            console.warn(`⚠ ChromaDB initialization failed: ${(error as Error).message}`);
            console.warn("  Falling back to in-memory storage");
            return false;
        }
    }

    async addEmbeddings(chunks: string[], embeddings: Map<string, LlamaEmbedding>): Promise<void> {
        if (!this.isReady || !this.collection) {
            throw new Error("ChromaDB not initialized");
        }

        const ids: string[] = [];
        const documents: string[] = [];
        const metadatas: Record<string, any>[] = [];
        const vectors: number[][] = [];

        let index = 0;
        for (const [text, embedding] of embeddings) {
            ids.push(`chunk_${index}`);
            documents.push(text);
            
            // Convert embedding to vector array
            const vector = embedding.vector ? Array.from(embedding.vector) : [];
            vectors.push(vector);
            
            metadatas.push({
                source: "pdf",
                timestamp: new Date().toISOString(),
                length: text.length,
            });
            
            index++;
        }

        try {
            await this.collection.add({
                ids,
                documents,
                metadatas,
                embeddings: vectors,
            });
            
            console.log(`✓ Stored ${ids.length} embeddings in ChromaDB`);
        } catch (error) {
            console.warn(`⚠ Failed to store embeddings in ChromaDB: ${(error as Error).message}`);
            throw error;
        }
    }

    async getEmbeddings(query: string, limit: number): Promise<string[]> {
        if (!this.isReady || !this.collection) {
            throw new Error("ChromaDB not initialized");
        }

        try {
            const results = await this.collection.query({
                queryTexts: [query],
                nResults: limit,
            });

            return results.documents[0] || [];
        } catch (error) {
            console.warn(`⚠ Query failed: ${(error as Error).message}`);
            throw error;
        }
    }

    async clear(): Promise<void> {
        if (this.isReady && this.collection) {
            try {
                // Get all items and delete them
                const allItems = await this.collection.get();
                if (allItems.ids?.length > 0) {
                    await this.collection.delete({ ids: allItems.ids });
                    console.log("✓ ChromaDB collection cleared");
                }
            } catch (error) {
                console.warn(`⚠ Failed to clear collection: ${(error as Error).message}`);
            }
        }
    }
}

export async function createEmbeddingStore(): Promise<EmbeddingStore> {
    const chromaStore = new ChromaDBStore();
    const initialized = await chromaStore.initialize();

    if (initialized) {
        return chromaStore;
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
            console.warn("⚠ Failed to store in ChromaDB, using in-memory fallback");
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
        console.warn(`⚠ Query failed on ${store.isInMemory ? "in-memory" : "ChromaDB"} store`);
        throw error;
    }
}
