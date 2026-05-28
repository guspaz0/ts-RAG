import { RagSystem } from "./ragSystem.ts";
import { formatQueryResult, queryWithContext } from "./query-engine";
import { LlamaEmbedding } from "node-llama-cpp";

// Function to process a PDF
export class QueryProcessor extends RagSystem {
  constructor(llama: any) {
    super(llama);
  }
  async processQuery(query: string, pdfPath?: string) {
    console.log(`Query: "${query}"`);

    // If no PDF provided, try to query existing embeddings in pgvector
    if (!pdfPath) {
      console.log(
        "No PDF provided. Querying existing embeddings in pgvector...",
      );

      try {
        // Get embedding for query
        if (!this.embeddingStore) {
          await this.initialize();
        }
        await this.loadContext();
        const queryEmbedding = (await this.embeddingContext?.getEmbeddingFor(
          query,
        )) as LlamaEmbedding;

        // Query pgvector for similar documents
        const similarDocuments = await this.embeddingStore?.queryByEmbedding(
          Array.from(queryEmbedding.vector || []),
          5,
        );

        if (similarDocuments?.length === 0) {
          console.warn(
            "No similar documents found in pgvector, trying in-memory store...",
          );
          // Fallback to in-memory store
          try {
            const memoryDocuments = await this.embeddingStore?.getEmbeddings(
              query,
              5,
            );
            if (memoryDocuments?.length === 0) {
              throw new Error("No similar documents found in any store");
            }
            console.log(
              `✓ Found ${memoryDocuments?.length} relevant chunks in in-memory store`,
            );
            // Show relevant context
          } catch (memoryError) {
            throw new Error(
              "Error querying in-memory store: " +
                (memoryError as Error).message,
            );
          }
          return;
        }

        console.log(
          `✓ Found ${similarDocuments?.length} relevant chunks in pgvector`,
        );

        // Process with language model if available
        if (this.queryContext) {
          console.log("\n🤖 Generating answer with language model...");
          // Get all embeddings from the database to provide full context
          let allDocuments: string[] = [];
          try {
            // Try to get all documents from pgvector first
            allDocuments =
              (await this.embeddingStore?.getAllEmbeddings()) as string[];
            console.log(
              `✓ Retrieved ${allDocuments.length} documents from database for full context`,
            );
          } catch (error) {
            console.warn(
              `⚠ Failed to retrieve all documents from database: ${(error as Error).message}`,
            );
            console.warn("  Falling back to using only similar documents...");
            // Fallback to using similar documents if database access fails
            allDocuments = similarDocuments?.map((d) => d.text) as string[];
          }

          // Re-rank documents using the reranker model
          if (this.reranker) {
            console.log("\n🔄 Re-ranking documents with cross-encoder...");
            allDocuments = await this.reranker.rank(query, allDocuments, 5);
            console.log(
              `✓ Re-ranked to top ${allDocuments.length} most relevant documents`,
            );
          }

          const result = await queryWithContext(
            this.queryContext,
            query,
            allDocuments,
          );

          console.log(formatQueryResult(result));
        } else {
          console.log("\n⚠ Language model not available.");
          console.log(
            "To enable query processing with a language model, download a model",
          );
        }
      } catch (error) {
        console.error("Error querying pgvector:", error);
        console.error("Falling back to in-memory store...");
        try {
          const similarDocuments = await this.embeddingStore?.getEmbeddings(
            query,
            5,
          );
          if (similarDocuments?.length === 0) {
            throw new Error(
              "No similar documents found in in-memory store either",
            );
          }
          console.log(
            `✓ Found ${similarDocuments?.length} relevant chunks in in-memory store`,
          );
          // Show relevant context
        } catch (fallbackError) {
          throw new Error(
            "Error querying in-memory store: " +
              (fallbackError as Error).message,
          );
        }
      }
      return;
    }
  }
}
