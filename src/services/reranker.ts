import { Llama, LlamaRankingContext } from "node-llama-cpp";

export interface RankedDocument {
  document: string;
  score: number;
}

export class Reranker {
  private context: LlamaRankingContext | null = null;
  private modelPath: string;
  private contextSize: number;

  constructor(modelPath: string, contextSize: number = 512) {
    this.modelPath = modelPath;
    this.contextSize = contextSize;
  }

  async initialize(llama: Llama): Promise<void> {
    try {
      const model = await llama.loadModel({
        modelPath: this.modelPath,
        ignoreMemorySafetyChecks: true,
      });
      this.context = await model.createRankingContext({
        contextSize: this.contextSize,
        ignoreMemorySafetyChecks: true,
      });
      console.log("✓ Reranker model loaded");
    } catch (error) {
      console.warn(
        `⚠ Failed to load reranker model: ${(error as Error).message}`,
      );
    }
  }

  async rank(
    query: string,
    documents: string[],
    topK?: number,
  ): Promise<string[]> {
    if (!this.context || documents.length === 0) {
      return topK ? documents.slice(0, topK) : documents;
    }

    try {
      const ranked = await this.context.rankAndSort(query, documents);
      const filtered = topK ? ranked.slice(0, topK) : ranked;
      return filtered.map((r) => r.document);
    } catch (error) {
      console.warn(
        `⚠ Reranking failed, using original order: ${(error as Error).message}`,
      );
      return documents.slice(0, topK);
    }
  }
}
