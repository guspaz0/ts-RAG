import fs from "node:fs";
import { RagSystem } from "./ragSystem.ts";
import {
  splitIntoChunks,
  embedDocuments,
  findSimilarDocuments,
} from "./pdf-embeddings";
import { queryWithContext, formatQueryResult } from "./query-engine";
import type { LlamaEmbedding, LlamaEmbeddingContext } from "node-llama-cpp";

export class MarkdownProcessor extends RagSystem {
  constructor(llama: any) {
    super(llama);
  }
  async processMarkdown(mdPath: string, query?: string | null) {
    if (!this.embeddingStore) {
      await this.initialize();
    }

    console.log(`📄 Reading Markdown from: ${mdPath}`);
    if (!mdPath) throw new Error("Markdown file not provided");

    const content = fs.readFileSync(mdPath, "utf-8");
    const mdChunks = splitIntoChunks(content);

    if (mdChunks.length === 0) {
      throw new Error("No content extracted from Markdown file");
    }

    console.log(`✓ Markdown parsed into ${mdChunks.length} chunks`);
    await this.loadContext();
    const documentEmbeddings = await embedDocuments(
      this.embeddingContext as LlamaEmbeddingContext,
      mdChunks,
    );

    if (documentEmbeddings.size === 0) {
      throw new Error("Failed to create any embeddings from the Markdown file");
    }

    console.log(
      `✓ Embeddings created successfully (${documentEmbeddings.size} chunks embedded)`,
    );

    const metadata = {
      source: "md",
      title: mdPath.split("/").pop(),
    };
    try {
      await this.embeddingStore?.addEmbeddings(
        mdChunks,
        documentEmbeddings,
        metadata,
      );
      console.log(
        `✓ Embeddings stored in ${this.embeddingStore?.isInMemory ? "memory" : "PostgreSQL pgvector"}`,
      );
    } catch (error) {
      console.warn(`⚠ Failed to store embeddings: ${(error as Error).message}`);
    }

    if (query && query !== "What is the content about?") {
      const queryEmbedding = (await this.embeddingContext?.getEmbeddingFor(
        query,
      )) as LlamaEmbedding;
      const similarDocuments = findSimilarDocuments(
        queryEmbedding,
        documentEmbeddings,
      );

      if (similarDocuments.length === 0) {
        throw new Error("No similar documents found");
      }

      console.log(`✓ Found ${similarDocuments.length} relevant chunks`);

      if (this.queryContext) {
        console.log("\n🤖 Generating answer with language model...");
        const result = await queryWithContext(
          this.queryContext,
          query,
          similarDocuments,
        );
        console.log(formatQueryResult(result));
      } else {
        console.log("\n⚠ Language model not available.");
        console.log(
          "To enable query processing with a language model, download a model",
        );
      }
    } else {
      console.log(
        `\n📄 Markdown processed successfully. Embeddings stored. No query was provided.`,
      );
    }
  }
}
