import { LlamaEmbedding } from "node-llama-cpp";
import { EmbeddingStore } from "./embedding-store";

// In-memory store fallback
export class InMemoryStore implements EmbeddingStore {
  isReady = true;
  isInMemory = true;
  private embeddings: Map<string, number[]> = new Map();

  async addEmbeddings(
    _chunks: string[],
    embeddings: Map<string, LlamaEmbedding>,
  ): Promise<void> {
    console.log("📦 Storing embeddings in memory...");

    for (const [text, embedding] of embeddings) {
      const vector = embedding.vector ? Array.from(embedding.vector) : [];
      if (vector.length > 0) {
        this.embeddings.set(text, vector);
      }
    }

    console.log(`✓ Stored ${this.embeddings.size} embeddings in memory`);
  }

  async getAllEmbeddings(): Promise<string[]> {
    return Array.from(this.embeddings.keys());
  }

  async getEmbeddings(_query: string, limit: number): Promise<string[]> {
    return Array.from(this.embeddings.keys()).slice(0, limit);
  }

  /**
   * Cosine-similarity search over the in-memory vectors so the fallback
   * store actually ranks results instead of returning nothing.
   */
  async queryByEmbedding(
    embedding: number[],
    limit: number,
  ): Promise<Array<{ text: string; similarity: number }>> {
    const results: Array<{ text: string; similarity: number }> = [];

    for (const [text, vec] of this.embeddings) {
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

  async clear(): Promise<void> {
    this.embeddings.clear();
  }
}
