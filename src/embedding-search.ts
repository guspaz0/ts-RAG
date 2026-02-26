import { LlamaContext } from "node-llama-cpp";
import { EmbeddingStore } from "./embedding-store.ts";

/**
 * Search embeddings from the persistent store
 * Returns top N similar documents
 */
export async function searchEmbeddings(
    store: EmbeddingStore,
    query: string,
    embeddingContext: LlamaContext,
    limit: number = 10
): Promise<string[]> {
    try {
        // Get embeddings from persistent store
        const results = await store.getEmbeddings(query, limit);
        
        if (results.length === 0) {
            console.warn("⚠ No embeddings found in database");
            return [];
        }
        
        console.log(`✓ Retrieved ${results.length} embeddings from ${store.isInMemory ? "memory" : "ChromaDB"}`);
        return results;
    } catch (error) {
        console.warn(`⚠ Failed to search embeddings: ${(error as Error).message}`);
        return [];
    }
}

/**
 * List all stored embeddings metadata
 */
export async function listStoredEmbeddings(
    store: EmbeddingStore,
    limit: number = 20
): Promise<void> {
    try {
        // For ChromaDB, this would get collection stats
        console.log(`\n📊 Stored Embeddings (${store.isInMemory ? "In-Memory" : "ChromaDB"})`);
        console.log("─".repeat(60));
        
        // This is a placeholder - would need actual implementation per store type
        console.log(`Storage Type: ${store.isInMemory ? "Memory" : "SQLite-backed ChromaDB"}`);
        console.log(`Database Path: ${store.isInMemory ? "N/A (volatile)" : "data/embeddings.db"}`);
    } catch (error) {
        console.warn(`⚠ Failed to list embeddings: ${(error as Error).message}`);
    }
}

/**
 * Clear all stored embeddings
 */
export async function clearAllEmbeddings(store: EmbeddingStore): Promise<void> {
    try {
        console.log(`🗑️  Clearing all embeddings from ${store.isInMemory ? "memory" : "ChromaDB"}...`);
        await store.clear();
        console.log("✓ All embeddings cleared");
    } catch (error) {
        console.warn(`⚠ Failed to clear embeddings: ${(error as Error).message}`);
    }
}

/**
 * Get storage info
 */
export function getStorageInfo(store: EmbeddingStore): {
    type: string;
    persistent: boolean;
    location: string;
} {
    return {
        type: store.isInMemory ? "In-Memory" : "ChromaDB + SQLite",
        persistent: !store.isInMemory,
        location: store.isInMemory ? "RAM (volatile)" : "data/embeddings.db",
    };
}
