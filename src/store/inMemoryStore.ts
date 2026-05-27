import { LlamaEmbedding } from "node-llama-cpp";
import { EmbeddingStore } from "./embedding-store";

// In-memory store fallback
export class InMemoryStore implements EmbeddingStore {
  isReady = true;
  isInMemory = true;
  private embeddingTexts: Map<string, number[]> = new Map();

  async addEmbeddings(
    _chunks: string[],
    embeddings: Map<string, LlamaEmbedding>,
  ): Promise<void> {
    console.log("📦 Storing embeddings in memory...");

    for (const [text, embedding] of embeddings) {
      // Store embedding vectors
      const vector = embedding.vector ? Array.from(embedding.vector) : [];
      this.embeddingTexts.set(text, vector);
    }

    console.log(`✓ Stored ${this.embeddingTexts.size} embeddings in memory`);
  }

  async getAllEmbeddings(): Promise<string[]> {
    return Array.from(this.embeddingTexts.keys());
  }

  async getEmbeddings(_query: string, limit: number): Promise<string[]> {
    return Array.from(this.embeddingTexts.keys()).slice(0, limit);
  }

  async queryByEmbedding(_embedding: number[], _limit: number): Promise<any[]> {
    return [];
  }

  async clear(): Promise<void> {
    this.embeddingTexts.clear();
  }
}
