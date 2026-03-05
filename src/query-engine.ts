import { Llama, LlamaContext } from "node-llama-cpp";

interface QueryResult {
    query: string;
    context: string;
    answer: string;
}

export async function createQueryEngine(
    llama: Llama,
    modelPath: string
): Promise<LlamaContext | null> {
    try {
        const model = await llama.loadModel({
            modelPath: modelPath,
        });
        const context = await model.createContext();
        return context;
    } catch (error) {
        console.error("Error loading language model:", error);
        return null;
    }
}

export async function queryWithContext(
    context: LlamaContext,
    query: string,
    relevantDocuments: string[],
    maxResults?: number // Optional limit, undefined = use all documents
): Promise<QueryResult> {
    try {
        // Filter out empty or very short documents
        const filteredDocuments = relevantDocuments.filter(doc => doc && doc.length > 10);

        // Use all documents or limit to maxResults if specified
        const contextDocs = maxResults ? filteredDocuments.slice(0, maxResults) : filteredDocuments;

        // Sort documents by length to prioritize more substantial content
        const sortedDocs = contextDocs.sort((a, b) => b.length - a.length);

        const contextText = sortedDocs.join("\n\n---\n\n");

        // Warn if context is very large
        const contextSize = contextText.length;
        if (contextSize > 20000) {
            console.warn(`⚠ Large context (${contextSize} chars) - model may struggle with this`);
            console.warn(`  Consider reducing maxResults or increasing model context window`);
        }

        // Build a more robust prompt with better instructions for quality answers
        const prompt = `You are an intelligent, helpful assistant that answers questions based on the provided context. Your goal is to provide accurate, comprehensive, and well-structured answers.

Guidelines for your response:
1. Use ONLY information from the provided context to answer the question
2. If the context doesn't contain sufficient information to fully answer the question, state that clearly
3. Structure your answer with clear organization (headings, bullet points, etc. if appropriate)
4. Be concise but thorough - avoid unnecessary repetition
5. If multiple relevant pieces of information are available, synthesize them coherently
6. When quoting directly from the context, cite the source appropriately
7. If the question cannot be answered from the provided context, explain why

Context:
${contextText}

Question: ${query}

Instructions: Provide a well-structured, high-quality answer using the context above. Format your response appropriately with clear structure and good readability.`;

        console.log(`\n⏳ Processing query with ${sortedDocs.length} relevant context chunks (${contextSize} chars)...`);

        try {
            // Use the sequence method to generate text
            const response = await context.getCompletionStream(prompt, {
                maxTokens: 4096,
                temperature: 0.3, // Lower temperature for more focused, deterministic responses
                topP: 0.9,
                stop: ["\n\n---\n\n"] // Prevent the model from generating additional context separators
            });

            let answer = "";
            for await (const chunk of response) {
                answer += chunk.token.text ?? "";
            }

            // Clean up the answer for better quality
            const cleanAnswer = answer.trim();

            return {
                query,
                context: contextText,
                answer: cleanAnswer || "Unable to generate response",
            };
        } catch (error) {
            // Fallback if getCompletionStream doesn't work
            const answerFallback = generateFallbackAnswer(query, sortedDocs);
            return {
                query,
                context: contextText,
                answer: answerFallback,
            };
        }
    } catch (error) {
        console.error("Error querying language model:", error);
        throw error;
    }
}

// Enhanced fallback when LLM generation fails
function generateFallbackAnswer(query: string, documents: string[]): string {
    if (documents.length === 0) {
        return "No relevant information found to answer this question.";
    }

    // Better approach to find relevant parts using semantic matching
    const queryLower = query.toLowerCase();
    const words = queryLower.split(/\s+/).filter(w => w.length > 3);

    // Score documents based on keyword relevance and length
    const scoredDocuments = documents.map((doc, idx) => {
        const docLower = doc.toLowerCase();
        const matches = words.filter(word => docLower.includes(word)).length;
        const relevanceScore = matches * 10 + (doc.length / 100); // Weight length as well
        return { doc, relevanceScore, idx };
    }).filter(r => r.relevanceScore > 0)
      .sort((a, b) => b.relevanceScore - a.relevanceScore);

    // Take the top 3 most relevant documents
    const topDocs = scoredDocuments.slice(0, 3).map(r => r.doc);

    if (topDocs.length === 0) {
        return `Based on the provided documents: ${documents[0]?.substring(0, 200)}...`;
    }

    // Create a more structured fallback answer
    const relevantContent = topDocs.join("\n\n---\n\n");
    return `Based on the most relevant information in the provided documents:\n\n${relevantContent}\n\nFor a more detailed answer, consider using the language model with a more specific query.`;
}

export function formatQueryResult(result: QueryResult): string {
    const contextLines = result.context.split("\n").length;
    const contextSize = result.context.length;

    return `
╔════════════════════════════════════════════════════════════════╗
║                    QUERY RESULT                                ║
╚════════════════════════════════════════════════════════════════╝

📝 Question: "${result.query}"

🔍 Context Used:
    ${result.context.split("\n---\n").length} chunks • ${contextSize} chars • ${contextLines} lines

💡 Answer:
────────────────────────────────────────────────────────────────
${result.answer}
────────────────────────────────────────────────────────────────
`;
}
