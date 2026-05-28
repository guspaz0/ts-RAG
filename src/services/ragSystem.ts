import { createEmbeddingStore, EmbeddingStore } from "../store/embedding-store";
import { existsSync } from "fs";
import { createQueryEngine } from "./query-engine";
import { Reranker } from "./reranker";
import { Llama, LlamaEmbeddingContext } from "node-llama-cpp";

export abstract class RagSystem {
  llama: Llama;
  embeddingStore: EmbeddingStore | null = null;
  embeddingContext: LlamaEmbeddingContext | null = null;
  embeddingModel: any;
  queryContext: any = null;
  reranker: Reranker | null = null;

  constructor(llama: any) {
    this.llama = llama;
    //void this.initialize();
  }
  async initialize() {
    try {
      this.embeddingStore = await createEmbeddingStore();
      this.embeddingModel = await this.llama.loadModel({
        modelPath: process.env["EMBEDDING_MODEL"] as string,
      });
      console.log(
        `✓ Embedding store initialized (${this.embeddingStore.isInMemory ? "In-Memory" : "PostgreSQL pgvector"})`,
      );
    } catch (error) {
      console.error("Error loading Llama model:", error);
      process.exit(1);
    }
  }
  async loadContext() {
    this.embeddingContext = await this.embeddingModel.createEmbeddingContext();

    const llmModelPath = process.env["QUERY_MODEL"] as string;
    const hasLLMModel = existsSync(llmModelPath);

    if (hasLLMModel) {
      try {
        this.queryContext = await createQueryEngine(this.llama, llmModelPath);
        if (this.queryContext) {
          console.log("✓ Language model loaded for query processing");
        }
      } catch (error) {
        console.warn(
          "⚠ Language model failed to load, will show retrieval results only",
        );
      }
    } else {
      console.warn("⚠ Language model not found at", llmModelPath);
    }

    const rerankerPath = process.env["RERANKING_MODEL"] as string;
    if (rerankerPath && existsSync(rerankerPath)) {
      this.reranker = new Reranker(rerankerPath);
      await this.reranker.initialize(this.llama);
    } else {
      console.warn("⚠ Reranker model not found at", rerankerPath);
    }
  }
}
