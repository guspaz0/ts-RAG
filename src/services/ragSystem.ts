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
    const modelPath = path.join(this.models, process.env["EMBEDDING_MODEL"] as string);
    console.log(`📦 Loading embedding model from: ${modelPath}`);

    if (!existsSync(modelPath)) {
      console.error(`✖ Model file not found: ${modelPath}`);
      console.error(`  Check MODELS_PATH and EMBEDDING_MODEL in .env`);
      process.exit(1);
    }

    try {
      this.embeddingModel = await this.llama.loadModel({ modelPath });

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
      console.error(`✖ Failed to load embedding model from ${modelPath}:`);
      console.error(`  ${(error as Error).message}`);
      if ((error as Error).stack) {
        console.error(`  ${(error as Error).stack!.split("\n").slice(0, 5).join("\n  ")}`);
      }
      process.exit(1);
    }
  }
  async loadContext() {
    this.embeddingContext = await this.embeddingModel.createEmbeddingContext();
    // Note: The query LLM and reranker are loaded on-demand via
    // loadQueryModel() and loadReranker() to avoid loading all models
    // into memory simultaneously.
  }

  /**
   * Load the query LLM (~4.7GB). Disposes the reranker first if loaded.
   */
  async loadQueryModel(): Promise<void> {
    if (this.queryContext) return; // already loaded

    // Dispose the reranker to free memory for the LLM
    if (this.reranker) {
      console.log("🗑 Disposing reranker to free memory for language model...");
      this.reranker.dispose();
      this.reranker = null;
    }

    const llmModelPath = path.join(this.models, process.env["QUERY_MODEL"] as string);
    if (existsSync(llmModelPath)) {
      try {
        this.queryContext = await createQueryEngine(this.llama, llmModelPath);
        if (this.queryContext) {
          console.log("✓ Language model loaded for query processing");
        }
      } catch (error) {
        console.warn(
          `⚠ Language model failed to load: ${(error as Error).message}`,
        );
      }
    } else {
      console.warn("⚠ Language model not found at", llmModelPath);
    }
  }

  /**
   * Load the reranker model (~600MB). Disposes the LLM first if loaded.
   */
  async loadReranker(): Promise<void> {
    if (this.reranker) return; // already loaded

    // Dispose the LLM to free memory for the reranker
    if (this.queryContext) {
      console.log("🗑 Disposing language model to free memory for reranker...");
      try {
        this.queryContext.dispose();
      } catch {
        // ignore
      }
      this.queryContext = null;
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
