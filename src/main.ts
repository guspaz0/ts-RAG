import { fileURLToPath } from "node:url";
import path from "node:path";
import { existsSync } from "node:fs";
import { getLlama } from "node-llama-cpp";
import { parsePDF, embedDocuments, findSimilarDocuments } from "./pdf-embeddings.ts";
import { createQueryEngine, queryWithContext, formatQueryResult } from "./query-engine.ts";
import { createEmbeddingStore, type EmbeddingStore } from "./embedding-store.ts";

process.loadEnvFile(path.join(process.cwd(), '.env'))

const __dirname = path.dirname(
    fileURLToPath(import.meta.url)
);

// Force using Metal backend for Mac Intel + AMD Gpu inference
const llama = await getLlama({ gpu: 'metal' });

// Initialize embedding store (PostgreSQL pgvector with fallback to in-memory)
const embeddingStore: EmbeddingStore = await createEmbeddingStore();
console.log(`✓ Embedding store initialized (${embeddingStore.isInMemory ? "In-Memory" : "PostgreSQL pgvector"})`);

// Embedding model
const embeddingModel = await llama.loadModel({
    modelPath: process.env['EMBEDDING_MODEL'] as string
});
const embeddingContext = await embeddingModel.createEmbeddingContext();

// Language model for query processing (optional)
let queryContext: any = null;
const llmModelPath = process.env['QUERY_MODEL'] as string;
const hasLLMModel = existsSync(llmModelPath);

if (hasLLMModel) {
    try {
        queryContext = await createQueryEngine(llama, llmModelPath);
        if (queryContext) {
            console.log("✓ Language model loaded for query processing");
        }
    } catch (error) {
        console.warn("⚠ Language model failed to load, will show retrieval results only");
    }
} else {
    console.warn("⚠ Language model not found at", llmModelPath);
}

async function main() {
    try {
        // Get PDF path from command line arguments
        const pdfPath = process.argv[2];
        const query = process.argv[3] || "What is the content about?";

        console.log(`Query: "${query}"`);

        // If no PDF provided, try to query existing embeddings in pgvector
        if (!pdfPath) {
            console.log("No PDF provided. Querying existing embeddings in pgvector...");

            try {
                // Get embedding for query
                const queryEmbedding = await embeddingContext.getEmbeddingFor(query);

                // Query pgvector for similar documents
                const similarDocuments = await embeddingStore.queryByEmbedding(
                    Array.from(queryEmbedding.vector || []),
                    5
                );

                if (similarDocuments.length === 0) {
                    console.warn("No similar documents found in pgvector, trying in-memory store...");
                    // Fallback to in-memory store
                    try {
                        const memoryDocuments = await embeddingStore.getEmbeddings(query, 5);
                        if (memoryDocuments.length === 0) {
                            console.error("No similar documents found in any store");
                            process.exit(1);
                        }
                        console.log(`✓ Found ${memoryDocuments.length} relevant chunks in in-memory store`);
                        // Show relevant context
                        const numChunksToShow = Math.min(5, memoryDocuments.length);
                        console.log(`\n📚 Showing top ${numChunksToShow} of ${memoryDocuments.length} relevant chunks:`);
                        console.log("────────────────────────────────────────────────");
                        memoryDocuments.slice(0, numChunksToShow).forEach((doc, index) => {
                            console.log(`\n[Chunk ${index + 1}/${memoryDocuments.length}]`);
                            console.log(doc.substring(0, 300) + (doc.length > 300 ? "..." : ""));
                        });
                        if (memoryDocuments.length > numChunksToShow) {
                            console.log(`\n[... and ${memoryDocuments.length - numChunksToShow} more chunks will be passed to the model ...]`);
                        }
                        console.log("\n────────────────────────────────────────────────");
                    } catch (memoryError) {
                        console.error("Error querying in-memory store:", memoryError);
                        process.exit(1);
                    }
                    return;
                }

                console.log(`✓ Found ${similarDocuments.length} relevant chunks in pgvector`);

                // Show relevant context
                const numChunksToShow = Math.min(5, similarDocuments.length);
                console.log(`\n📚 Showing top ${numChunksToShow} of ${similarDocuments.length} relevant chunks:`);
                console.log("────────────────────────────────────────────────");
                similarDocuments.slice(0, numChunksToShow).forEach((doc, index) => {
                    console.log(`\n[Chunk ${index + 1}/${similarDocuments.length}]`);
                    console.log(doc.text.substring(0, 300) + (doc.text.length > 300 ? "..." : ""));
                });
                if (similarDocuments.length > numChunksToShow) {
                    console.log(`\n[... and ${similarDocuments.length - numChunksToShow} more chunks will be passed to the model ...]`);
                }
                console.log("\n────────────────────────────────────────────────");

                // Process with language model if available
                if (queryContext) {
                    console.log("\n🤖 Generating answer with language model...");
                    const result = await queryWithContext(
                        queryContext,
                        query,
                        similarDocuments.map(d => d.text)
                    );

                    console.log(formatQueryResult(result));
                } else {
                    console.log("\n⚠ Language model not available.");
                    console.log("To enable query processing with a language model, download a model");
                }
            } catch (error) {
                console.error("Error querying pgvector:", error);
                console.error("Falling back to in-memory store...");
                try {
                    const similarDocuments = await embeddingStore.getEmbeddings(query, 5);
                    if (similarDocuments.length === 0) {
                        console.error("No similar documents found in in-memory store either");
                        process.exit(1);
                    }
                    console.log(`✓ Found ${similarDocuments.length} relevant chunks in in-memory store`);
                    // Show relevant context
                    const numChunksToShow = Math.min(5, similarDocuments.length);
                    console.log(`\n📚 Showing top ${numChunksToShow} of ${similarDocuments.length} relevant chunks:`);
                    console.log("────────────────────────────────────────────────");
                    similarDocuments.slice(0, numChunksToShow).forEach((doc, index) => {
                        console.log(`\n[Chunk ${index + 1}/${similarDocuments.length}]`);
                        console.log(doc.substring(0, 300) + (doc.length > 300 ? "..." : ""));
                    });
                    if (similarDocuments.length > numChunksToShow) {
                        console.log(`\n[... and ${similarDocuments.length - numChunksToShow} more chunks will be passed to the model ...]`);
                    }
                    console.log("\n────────────────────────────────────────────────");
                } catch (fallbackError) {
                    console.error("Error querying in-memory store:", fallbackError);
                    process.exit(1);
                }
            }
            return;
        }

        // Parse a PDF file and create embeddings
        console.log(`📄 Parsing PDF from: ${pdfPath}`);
        if (!pdfPath) throw new Error('Pdf file not provided')
        const pdfChunks = await parsePDF(pdfPath);

        if (pdfChunks.length === 0) {
            console.error("No content extracted from PDF");
            process.exit(1);
        }

        console.log(`✓ PDF parsed into ${pdfChunks.length} chunks`);

        const documentEmbeddings = await embedDocuments(embeddingContext, pdfChunks);

        if (documentEmbeddings.size === 0) {
            console.error("Failed to create any embeddings from the PDF");

            // Store embeddings in PostgreSQL or memory
            try {
                await embeddingStore.addEmbeddings(pdfChunks, documentEmbeddings);
                console.log(`✓ Embeddings stored in ${embeddingStore.isInMemory ? "memory" : "PostgreSQL pgvector"}`);
            } catch (error) {
                console.warn(`⚠ Failed to store embeddings: ${(error as Error).message}`);
            }
            process.exit(1);
        }

        console.log(`✓ Embeddings created successfully (${documentEmbeddings.size} chunks embedded)`);

        // Store embeddings in PostgreSQL or memory
        try {
            await embeddingStore.addEmbeddings(pdfChunks, documentEmbeddings);
            console.log(`✓ Embeddings stored in ${embeddingStore.isInMemory ? "memory" : "PostgreSQL pgvector"}`);
        } catch (error) {
            console.warn(`⚠ Failed to store embeddings: ${(error as Error).message}`);
        }

        // Find similar documents for the query
        const queryEmbedding = await embeddingContext.getEmbeddingFor(query);
        const similarDocuments = findSimilarDocuments(
            queryEmbedding,
            documentEmbeddings
        );

        if (similarDocuments.length === 0) {
            console.error("No similar documents found");
            process.exit(1);
        }

        console.log(`✓ Found ${similarDocuments.length} relevant chunks`);

        // Show relevant context
        const numChunksToShow = Math.min(5, similarDocuments.length);
        console.log(`\n📚 Showing top ${numChunksToShow} of ${similarDocuments.length} relevant chunks:`);
        console.log("────────────────────────────────────────────────");
        similarDocuments.slice(0, numChunksToShow).forEach((doc, index) => {
            console.log(`\n[Chunk ${index + 1}/${similarDocuments.length}]`);
            console.log(doc.substring(0, 300) + (doc.length > 300 ? "..." : ""));
        });
        if (similarDocuments.length > numChunksToShow) {
            console.log(`\n[... and ${similarDocuments.length - numChunksToShow} more chunks will be passed to the model ...]`);
        }
        console.log("\n────────────────────────────────────────────────");

        // Process with language model if available
        if (queryContext) {
            console.log("\n🤖 Generating answer with language model...");
            // Pass ALL similar documents to the LLM (not limited to 3)
            const result = await queryWithContext(
                queryContext,
                query,
                similarDocuments
                // No maxResults specified - will use all documents
            );

            console.log(formatQueryResult(result));
        } else {
            console.log("\n⚠ Language model not available.");
            console.log("To enable query processing with a language model, download a model");
        }
    } catch (error) {
        console.error("Error:", error);
        process.exit(1);
    }
}

main();