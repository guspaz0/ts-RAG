import { LlamaEmbedding } from "node-llama-cpp";
import { EmbeddingStore } from "./embedding-store";
import { InMemoryCatalogStore, type CatalogStore, type Catalog } from "./catalogs";

interface StoredChunk {
  text: string;
  vector: number[];
  catalogId: string | null;
}

// In-memory store fallback
export class InMemoryStore implements EmbeddingStore {
  isReady = true;
  isInMemory = true;
  catalogStore: CatalogStore = new InMemoryCatalogStore();
  private chunks: StoredChunk[] = [];

  async addEmbeddings(
    _chunks: string[],
    embeddings: Map<string, LlamaEmbedding>,
    _metadata?: Record<string, any>,
    catalogId?: string | null,
  ): Promise<void> {
    console.log("📦 Storing embeddings in memory...");

    for (const [text, embedding] of embeddings) {
      const vector = embedding.vector ? Array.from(embedding.vector) : [];
      if (vector.length > 0) {
        // Skip if this exact chunk already exists in the target catalog
        const exists = this.chunks.some(
          (c) => c.text === text && c.catalogId === catalogId,
        );
        if (!exists) {
          this.chunks.push({ text, vector, catalogId: catalogId ?? null });
        }
      }
    }

    console.log(`✓ Stored ${this.chunks.length} embeddings in memory`);
  }

  async getAllEmbeddings(catalogId?: string | null): Promise<string[]> {
    return this.chunks
      .filter((c) => c.catalogId === catalogId)
      .map((c) => c.text);
  }

  async getEmbeddings(
    _query: string,
    limit: number,
    catalogId?: string | null,
  ): Promise<string[]> {
    return this.getAllEmbeddings(catalogId).then((texts) =>
      texts.slice(0, limit),
    );
  }

  /**
   * Cosine-similarity search over the in-memory vectors so the fallback
   * store actually ranks results instead of returning nothing.
   */
  async queryByEmbedding(
    embedding: number[],
    limit: number,
    catalogId?: string | null,
  ): Promise<Array<{ text: string; similarity: number }>> {
    const results: Array<{ text: string; similarity: number }> = [];

    for (const { text, vector: vec, catalogId: cid } of this.chunks) {
      if (cid !== catalogId) continue;
      if (vec.length !== embedding.length) continue;
      let dot = 0;
      let normA = 0;
      let normB = 0;
      for (let i = 0; i < vec.length; i++) {
        const a = embedding[i]!;
        const b = vec[i]!;
        dot += a * b;
        normA += a * a;
        normB += b * b;
      }
      const similarity =
        normA > 0 && normB > 0 ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
      results.push({ text, similarity });
    }

    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  async clear(catalogId?: string | null): Promise<void> {
    if (catalogId) {
      this.chunks = this.chunks.filter((c) => c.catalogId !== catalogId);
    } else {
      this.chunks = [];
    }
  }
}
