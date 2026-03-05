import { LlamaEmbedding, LlamaEmbeddingContext } from "node-llama-cpp";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"

// Maximum context size for Gemma-300M model (in tokens, roughly 1 token ≈ 4 chars)
const MAX_CONTEXT_CHARS = 1000;

function splitIntoChunks(text: string, maxChunkSize: number = MAX_CONTEXT_CHARS): string[] {
    if (!text || text.length === 0) {
        return [];
    }

    const chunks: string[] = [];
    const sentences = text.split(/(?<=[.!?])\s+/);
    let currentChunk = "";

    for (const sentence of sentences) {
        // If a single sentence is too long, split it by lines
        if (sentence.length > maxChunkSize) {
            if (currentChunk) {
                chunks.push(currentChunk.trim());
                currentChunk = "";
            }
            
            const lines = sentence.split("\n");
            for (const line of lines) {
                if (line.length > maxChunkSize) {
                    // Force split very long lines
                    for (let i = 0; i < line.length; i += maxChunkSize) {
                        const chunk = line.substring(i, i + maxChunkSize);
                        if (chunk.trim()) {
                            chunks.push(chunk.trim());
                        }
                    }
                } else if (line.trim()) {
                    chunks.push(line.trim());
                }
            }
        } else if ((currentChunk + " " + sentence).length <= maxChunkSize) {
            currentChunk += (currentChunk ? " " : "") + sentence;
        } else {
            if (currentChunk) {
                chunks.push(currentChunk.trim());
            }
            currentChunk = sentence;
        }
    }

    if (currentChunk.trim()) {
        chunks.push(currentChunk.trim());
    }

    return chunks.length > 0 ? chunks : [text];
}

export async function parsePDF(pdfPath: string): Promise<string[]> {
    try {
        const loader = new PDFLoader(pdfPath);
        const docs = await loader.load();
        const fullText = docs.map(doc => doc.pageContent).join("\n\n");
        
        // Split into chunks to respect context size limits
        const chunks = splitIntoChunks(fullText);
        console.log(`PDF parsed and split into ${chunks.length} chunks`);
        
        return chunks;
    } catch (error) {
        console.error("Error parsing PDF:", error);
        throw error;
    }
}

export async function embedDocuments(
    context: LlamaEmbeddingContext,
    documents: readonly string[]
) {
    const embeddings = new Map<string, LlamaEmbedding>();
    let processed = 0;
    let failed = 0;

    for (const document of documents) {
        try {
            if (document.length === 0) {
                continue;
            }
            const embedding = await context.getEmbeddingFor(document);
            embeddings.set(document, embedding);
            processed++;

            console.debug(
                `${processed}/${documents.length} documents embedded`, { flush: true }
            );
        } catch (error) {
            failed++;
            const errorMsg = (error as Error).message || String(error);
            console.warn(`Failed to embed chunk (${document.length} chars): ${errorMsg}`);
        }
    }

    console.log(`Embedding complete: ${processed} succeeded, ${failed} failed`);
    return embeddings;
}

export function findSimilarDocuments(
    embedding: LlamaEmbedding,
    documentEmbeddings: Map<string, LlamaEmbedding>
) {
    const similarities = new Map<string, number>();
    for (const [otherDocument, otherDocumentEmbedding] of documentEmbeddings)
        similarities.set(
            otherDocument,
            embedding.calculateCosineSimilarity(otherDocumentEmbedding)
        );

    return Array.from(similarities.keys())
        .sort((a, b) => similarities.get(b)! - similarities.get(a)!);
}
