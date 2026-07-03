import { LlamaContext, LlamaCompletion } from "node-llama-cpp";

export interface ExpandedQueries {
  original: string;
  variants: string[];
  all(): string[];
}

export async function expandQuery(
  queryContext: LlamaContext,
  query: string,
  numVariations: number = 2,
): Promise<ExpandedQueries> {
  try {
    const prompt = `Generate ${numVariations} alternative search queries that capture different phrasings of the same information need. Each alternative should rephrase the query using different keywords and sentence structure while preserving the core intent. Keep each query concise and focused. Return one per line, no numbering or prefixes.

Original query: ${query}
Alternative queries:`;

    const sequence = queryContext.getSequence();
    const completion = new LlamaCompletion({ contextSequence: sequence });
    const answer = await completion.generateCompletion(prompt, {
      maxTokens: 256,
      temperature: 0.8,
      topP: 0.9,
    });

    const variants = answer
      .trim()
      .split("\n")
      .map((l) => l.replace(/^[\d-.\s)]+/, "").trim())
      .filter((l) => l.length > 0 && l.length < 200)
      .slice(0, numVariations);

    if (variants.length === 0) {
      console.log("  ℹ Query expansion produced no usable variants");
      return { original: query, variants: [], all: () => [query] };
    }

    console.log(`\n🔍 Expanded query into ${variants.length + 1} variants:`);
    console.log(`   0: "${query}" (original, weighted x2)`);
    variants.forEach((v, i) => console.log(`   ${i + 1}: "${v}"`));

    return {
      original: query,
      variants,
      all: () => [query, ...variants],
    };
  } catch (error) {
    console.warn(`⚠ Query expansion failed: ${(error as Error).message}`);
    return { original: query, variants: [], all: () => [query] };
  }
}
