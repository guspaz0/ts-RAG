import { RagSystem } from "./ragSystem.ts";
import { formatQueryResult, queryWithContext } from "./query-engine";
import { expandQuery } from "./query-expander";

interface RRFResult {
  text: string;
  score: number;
  similarity: number;
}

function reciprocalRankFusion(
  results: Array<Array<{ text: string; similarity: number }>>,
  weights: number[],
  k: number = 60,
  topK: number = 30,
): RRFResult[] {
  const fused = new Map<string, { score: number; similarity: number }>();

  for (let listIdx = 0; listIdx < results.length; listIdx++) {
    const list = results[listIdx]!;
    const weight = weights[listIdx] ?? 1;

    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank]!;
      const existing = fused.get(item.text) ?? { score: 0, similarity: 0 };
      existing.score += weight / (k + rank + 1);
      existing.similarity = Math.max(existing.similarity, item.similarity);
      fused.set(item.text, existing);
    }
  }

  return Array.from(fused.entries())
    .map(([text, { score, similarity }]) => ({ text, score, similarity }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

export class QueryProcessor extends RagSystem {
  constructor(llama: any) {
    super(llama);
  }

  async processQuery(query: string, pdfPath?: string) {
    console.log(`Query: "${query}"`);

    if (!pdfPath) {
      console.log(
        "No PDF provided. Querying existing embeddings in pgvector...",
      );

      try {
        if (!this.embeddingStore) {
          await this.initialize();
        }
        await this.loadContext();

        const expanded = this.queryContext
          ? await expandQuery(this.queryContext, query, 2)
          : { original: query, variants: [], all: () => [query] };
        const queries = expanded.all();

        const embeddingResults = await Promise.allSettled(
          queries.map((q) => this.embeddingContext!.getEmbeddingFor(q)),
        );

        const validSearches: Array<{
          embedding: number[];
          weight: number;
        }> = [];

        for (let i = 0; i < embeddingResults.length; i++) {
          const result = embeddingResults[i]!;
          if (result.status === "rejected") {
            console.warn(
              `⚠ Failed to embed query variant "${queries[i]}": ${result.reason?.message ?? String(result.reason)}`,
            );
            continue;
          }
          const vec = Array.from(result.value.vector ?? []);
          if (vec.length > 0) {
            validSearches.push({
              embedding: vec,
              weight: i === 0 ? 2 : 1,
            });
          }
        }

        const searchResults = await Promise.allSettled(
          validSearches.map(({ embedding }) =>
            this.embeddingStore!.queryByEmbedding(embedding, 15),
          ),
        );

        const allResultLists: Array<
          Array<{ text: string; similarity: number }>
        > = [];
        const rrfWeights: number[] = [];

        for (let i = 0; i < searchResults.length; i++) {
          const result = searchResults[i]!;
          if (result.status === "fulfilled" && result.value.length > 0) {
            allResultLists.push(result.value);
            rrfWeights.push(validSearches[i]!.weight);
          }
        }

        let fusedDocuments: RRFResult[] = [];

        if (allResultLists.length > 1) {
          fusedDocuments = reciprocalRankFusion(allResultLists, rrfWeights);
          console.log(
            `\n🔀 Fused ${allResultLists.length} search result lists via RRF → ${fusedDocuments.length} candidates`,
          );
        } else if (allResultLists.length === 1) {
          fusedDocuments = allResultLists[0]!.map((d) => ({
            text: d.text,
            score: d.similarity,
            similarity: d.similarity,
          }));
        }

        if (fusedDocuments.length === 0) {
          console.warn(
            "No similar documents found in pgvector, trying in-memory store...",
          );
          try {
            const memoryDocuments = await this.embeddingStore?.getEmbeddings(
              query,
              10,
            );
            if (memoryDocuments?.length === 0) {
              throw new Error("No similar documents found in any store");
            }
            console.log(
              `✓ Found ${memoryDocuments?.length} relevant chunks in in-memory store`,
            );
          } catch (memoryError) {
            throw new Error(
              "Error querying in-memory store: " +
                (memoryError as Error).message,
            );
          }
          return;
        }

        console.log(
          `✓ Found ${fusedDocuments.length} relevant chunks via hybrid search`,
        );

        if (this.queryContext) {
          console.log("\n🤖 Generating answer with language model...");
          let allDocuments: string[] = fusedDocuments.map((d) => d.text);

          try {
            const allDocs =
              (await this.embeddingStore?.getAllEmbeddings()) as string[];
            if (allDocs && allDocs.length > fusedDocuments.length) {
              console.log(
                `✓ Retrieved ${allDocs.length} documents from database for full context`,
              );
              allDocuments = allDocs;
            }
          } catch (error) {
            console.warn(
              `⚠ Failed to retrieve all documents: ${(error as Error).message}`,
            );
          }

          if (this.reranker) {
            console.log("\n🔄 Re-ranking documents with cross-encoder...");
            allDocuments = await this.reranker.rank(query, allDocuments, 15);
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
            10,
          );
          if (similarDocuments?.length === 0) {
            throw new Error(
              "No similar documents found in in-memory store either",
            );
          }
          console.log(
            `✓ Found ${similarDocuments?.length} relevant chunks in in-memory store`,
          );
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
