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
        // Parse command line arguments
        let pdfPath: string | null = null;
        let query: string | null = null;

        // Process command line arguments to find --pdf and --query options
        for (let i = 2; i < process.argv.length; i++) {
            if (process.argv[i] === '--pdf') {
                pdfPath = process.argv[i + 1];
                i++; // Skip next argument as it's the value
            } else if (process.argv[i] === '--query') {
                query = process.argv[i + 1];
                i++; // Skip next argument as it's the value
            }
        }

        // If no --pdf or --query provided, check for positional arguments
        if (!pdfPath && !query) {
            pdfPath = process.argv[2];
            query = process.argv[3];
        }

        // If no query provided, set default
        if (!query) {
            query = "What is the content about?";
        }

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
                    } catch (memoryError) {
                        console.error("Error querying in-memory store:", memoryError);
                        process.exit(1);
                    }
                    return;
                }

                console.log(`✓ Found ${similarDocuments.length} relevant chunks in pgvector`);

                // Process with language model if available
                if (queryContext) {
                    console.log("\n🤖 Generating answer with language model...");
                    // Get all embeddings from the database to provide full context
                    let allDocuments: string[] = [];
                    try {
                        // Try to get all documents from pgvector first
                        allDocuments = await embeddingStore.getAllEmbeddings();
                        console.log(`✓ Retrieved ${allDocuments.length} documents from database for full context`);
                    } catch (error) {
                        console.warn(`⚠ Failed to retrieve all documents from database: ${(error as Error).message}`);
                        console.warn("  Falling back to using only similar documents...");
                        // Fallback to using similar documents if database access fails
                        allDocuments = similarDocuments.map(d => d.text);
                    }

                    const result = await queryWithContext(
                        queryContext,
                        query,
                        allDocuments
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

        // Process query if provided
        if (query && query !== "What is the content about?") {
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
        } else {
            // If no query was provided, just show the PDF content
            console.log(`\n📄 PDF processed successfully. Embeddings stored. No query was provided.`);
        }
    } catch (error) {
        console.error("Error:", error);
        process.exit(1);
    }
}

main();