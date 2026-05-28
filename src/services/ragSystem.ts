import { createEmbeddingStore, EmbeddingStore } from "../store/embedding-store";
import { existsSync } from "fs";
import { createQueryEngine } from "./query-engine";
import { Reranker } from "./reranker";
import { Llama, LlamaEmbeddingContext } from "node-llama-cpp";
import path from "node:path"

export abstract class RagSystem {
  llama: Llama;
  embeddingStore: EmbeddingStore | null = null;
  embeddingContext: LlamaEmbeddingContext | null = null;
  embeddingModel: any;
  queryContext: any = null;
  reranker: Reranker | null = null;
  models: string

  constructor(llama: any) {
    this.llama = llama;
    this.models = process.env["MODELS_PATH"] as string
    //void this.initialize();
  }
  async initialize() {
    try {
      this.embeddingModel = await this.llama.loadModel({
        modelPath: path.join(this.models, process.env["EMBEDDING_MODEL"] as string),
      });

      // Detect embedding dimension from model if not specified in env
      let dimension = parseInt(process.env["EMBEDDING_DIMENSION"] || "");
      if (isNaN(dimension)) {
        const tempContext = await this.embeddingModel.createEmbeddingContext();
        const testEmbedding = await tempContext.getEmbeddingFor("test");
        dimension = testEmbedding.vector?.length ?? 384;
      }

      this.embeddingStore = await createEmbeddingStore(dimension);
      console.log(
        `✓ Embedding store initialized (${this.embeddingStore.isInMemory ? "In-Memory" : "PostgreSQL pgvector"}, ${dimension}d)`,
      );
    } catch (error) {
      console.error("Error loading Llama model:", error);
      process.exit(1);
    }
  }
  async loadContext() {
    this.embeddingContext = await this.embeddingModel.createEmbeddingContext();

    const llmModelPath = path.join(this.models, process.env["QUERY_MODEL"] as string);
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

    const rerankerPath = path.join(this.models, process.env["RERANKING_MODEL"] as string);
    if (rerankerPath && existsSync(rerankerPath)) {
      this.reranker = new Reranker(rerankerPath);
      await this.reranker.initialize(this.llama);
    } else {
      console.warn("⚠ Reranker model not found at", rerankerPath);
    }
  }
}
