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
        // Use all documents or limit to maxResults if specified
        const contextDocs = maxResults ? relevantDocuments.slice(0, maxResults) : relevantDocuments;
        const contextText = contextDocs.join("\n\n---\n\n");
        
        // Warn if context is very large
        const contextSize = contextText.length;
        if (contextSize > 20000) {
            console.warn(`⚠ Large context (${contextSize} chars) - model may struggle with this`);
            console.warn(`  Consider reducing maxResults or increasing model context window`);
        }

        // Build the prompt
        const prompt = `You are a helpful assistant that answers questions based on the provided context. Use ALL the provided context to give a comprehensive and accurate answer.

Context:
${contextText}

Question: ${query}

Answer:`;

        console.log(`\n⏳ Processing query with ${contextDocs.length} context chunks (${contextSize} chars)...`);
        
        try {
            // Use the sequence method to generate text
            const response = await context.getCompletionStream(prompt, {
                maxTokens: 2048,
                temperature: 0.7,
                topP: 0.9,
            });
            
            let answer = "";
            for await (const chunk of response) {
                answer += chunk.token.text ?? "";
            }

            return {
                query,
                context: contextText,
                answer: answer.trim() || "Unable to generate response",
            };
        } catch {
            // Fallback if getCompletionStream doesn't work
            const answerFallback = generateFallbackAnswer(query, contextDocs);
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

// Fallback when LLM generation fails
function generateFallbackAnswer(query: string, documents: string[]): string {
    if (documents.length === 0) {
        return "No relevant information found to answer this question.";
    }

    // Simple keyword matching to find relevant parts
    const queryLower = query.toLowerCase();
    const words = queryLower.split(/\s+/).filter(w => w.length > 3);
    
    const relevantPassages = documents.map((doc, idx) => {
        const matches = words.filter(word => doc.toLowerCase().includes(word)).length;
        return { doc, matches, idx };
    }).filter(r => r.matches > 0)
      .sort((a, b) => b.matches - a.matches)
      .slice(0, 2)
      .map(r => r.doc);

    if (relevantPassages.length === 0) {
        return `Based on the provided documents: ${documents[0]?.substring(0, 200)}...`;
    }

    return `Based on the relevant information: ${relevantPassages.join(" ")}`;
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
────────────────────────────────────────────────────────────────
${result.context.substring(0, 1500)}${result.context.length > 1500 ? "\n\n[... full context provided to model ...]" : ""}
────────────────────────────────────────────────────────────────

💡 Answer:
────────────────────────────────────────────────────────────────
${result.answer}
────────────────────────────────────────────────────────────────
`;
}
