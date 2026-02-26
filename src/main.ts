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
    modelPath: path.join(__dirname, "..", "models", process.env['EMBEDDING_MODEL'] as string),
});
const embeddingContext = await embeddingModel.createEmbeddingContext();

// Language model for query processing (optional)
let queryContext: any = null;
const llmModelPath = path.join(__dirname, "..", "models",  process.env['QUERY_MODEL'] as string);
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
        // Parse a PDF file and create embeddings
        const pdfPath = process.argv[2] || "./sample.pdf";
        console.log(`📄 Parsing PDF from: ${pdfPath}`);
        
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

        // Get user query
        const query = `${process.argv[3] || "What is the content about?"}`;
        console.log(`\n Query: "${query}"`);

        // Find similar documents
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