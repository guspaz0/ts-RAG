import { RagSystem } from "./ragSystem.ts";
import {
  parsePDF,
  embedDocuments,
  findSimilarDocuments,
} from "./pdf-embeddings";
import { queryWithContext, formatQueryResult } from "./query-engine";
import type { LlamaEmbedding, LlamaEmbeddingContext } from "node-llama-cpp";

export class PdfProcessor extends RagSystem {
  constructor(llama: any) {
    super(llama);
  }
  async processPDF(pdfPath: string, query?: string | null) {
    if (!this.embeddingStore) {
      await this.initialize();
    }

    console.log(`📄 Parsing PDF from: ${pdfPath}`);
    if (!pdfPath) throw new Error("Pdf file not provided");
    const pdfChunks = await parsePDF(pdfPath);

    if (pdfChunks.length === 0) {
      throw new Error("No content extracted from PDF");
    }

    console.log(`✓ PDF parsed into ${pdfChunks.length} chunks`);
    await this.loadContext();
    const documentEmbeddings = await embedDocuments(
      this.embeddingContext as LlamaEmbeddingContext,
      pdfChunks,
    );

    if (documentEmbeddings.size === 0) {
      throw new Error("Failed to create any embeddings from the PDF");
    }

    console.log(
      `✓ Embeddings created successfully (${documentEmbeddings.size} chunks embedded)`,
    );

    // Store embeddings in PostgreSQL or memory
    try {
      await this.embeddingStore?.addEmbeddings(pdfChunks, documentEmbeddings);
      console.log(
        `✓ Embeddings stored in ${this.embeddingStore?.isInMemory ? "memory" : "PostgreSQL pgvector"}`,
      );
    } catch (error) {
      console.warn(`⚠ Failed to store embeddings: ${(error as Error).message}`);
    }

    // Process query if provided
    if (query && query !== "What is the content about?") {
      // Find similar documents for the query
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

      // Process with language model if available
      if (this.queryContext) {
        console.log("\n🤖 Generating answer with language model...");
        // Pass ALL similar documents to the LLM (not limited to 3)
        const result = await queryWithContext(
          this.queryContext,
          query,
          similarDocuments,
          // No maxResults specified - will use all documents
        );

        console.log(formatQueryResult(result));
      } else {
        console.log("\n⚠ Language model not available.");
        console.log(
          "To enable query processing with a language model, download a model",
        );
      }
    } else {
      // If no query was provided, just show the PDF content
      console.log(
        `\n📄 PDF processed successfully. Embeddings stored. No query was provided.`,
      );
    }
  }
}
