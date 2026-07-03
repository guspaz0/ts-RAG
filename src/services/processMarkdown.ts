import fs from "node:fs";
import { RagSystem } from "./ragSystem.ts";
import {
  splitIntoChunks,
  embedDocuments,
  findSimilarDocuments,
} from "./pdf-embeddings";
import { queryWithContext, formatQueryResult } from "./query-engine";
import { expandQuery } from "./query-expander";
import type { LlamaEmbedding, LlamaEmbeddingContext } from "node-llama-cpp";

function reciprocalRankFusion(
  results: string[][],
  weights: number[],
  k: number = 60,
  topK: number = 30,
): string[] {
  const fused = new Map<string, number>();

  for (let listIdx = 0; listIdx < results.length; listIdx++) {
    const list = results[listIdx]!;
    const weight = weights[listIdx] ?? 1;

    for (let rank = 0; rank < list.length; rank++) {
      const doc = list[rank]!;
      const existing = fused.get(doc) ?? 0;
      fused.set(doc, existing + weight / (k + rank + 1));
    }
  }

  return Array.from(fused.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([doc]) => doc);
}

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
      const expanded = this.queryContext
        ? await expandQuery(this.queryContext, query, 2)
        : { original: query, variants: [], all: () => [query] };
      const queries = expanded.all();

      const embeddingResults = await Promise.allSettled(
        queries.map((q) => this.embeddingContext!.getEmbeddingFor(q)),
      );

      const validSearches: Array<{ embedding: LlamaEmbedding; weight: number }> = [];

      for (let i = 0; i < embeddingResults.length; i++) {
        const result = embeddingResults[i]!;
        if (result.status === "rejected") {
          console.warn(
            `⚠ Failed to embed query variant "${queries[i]}": ${result.reason?.message ?? String(result.reason)}`,
          );
          continue;
        }
        if (result.value.vector && result.value.vector.length > 0) {
          validSearches.push({ embedding: result.value, weight: i === 0 ? 2 : 1 });
        }
      }

      const searchResultLists: string[][] = [];

      for (const { embedding } of validSearches) {
        const results = findSimilarDocuments(embedding, documentEmbeddings);
        if (results.length > 0) {
          searchResultLists.push(results);
        }
      }

      let similarDocuments: string[];

      if (searchResultLists.length > 1) {
        const weights = validSearches.map((s) => s.weight);
        similarDocuments = reciprocalRankFusion(searchResultLists, weights);
        console.log(
          `\n🔀 Fused ${searchResultLists.length} search lists via RRF → ${similarDocuments.length} candidates`,
        );
      } else if (searchResultLists.length === 1) {
        similarDocuments = searchResultLists[0]!;
      } else {
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
      }
    } else {
      console.log(
        `\n📄 Markdown processed successfully. Embeddings stored. No query was provided.`,
      );
    }
  }
}
