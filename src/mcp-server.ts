import "dotenv/config";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { createEmbeddingStore, type EmbeddingStore } from "./store/embedding-store.ts";
import { getLlama, type LlamaEmbeddingContext } from "node-llama-cpp";
import path from "node:path";

let store: EmbeddingStore;
let embeddingContext: LlamaEmbeddingContext | null = null;

const server = new McpServer(
  {
    name: "embedding-store-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

server.registerTool("search_embeddings", {
  description: "Search for similar text chunks in the embedding store using semantic similarity",
  inputSchema: z.object({
    query: z.string().describe("The text query to search for"),
    limit: z.number().min(1).max(100).default(10).describe("Maximum number of results to return"),
  }),
}, async (args) => {
  const { query, limit } = args;

  if (embeddingContext) {
    const embedding = await embeddingContext.getEmbeddingFor(query);
    const vector = embedding.vector ? Array.from(embedding.vector) : [];

    if (vector.length > 0) {
      const results = await store.queryByEmbedding(vector, limit);
      const content = results.map((r) =>
        `[${(r.similarity * 100).toFixed(1)}%] ${r.text}`
      ).join("\n\n---\n\n");
      return {
        content: [{ type: "text" as const, text: content || "No results found" }],
      };
    }
  }

  const results = await store.getEmbeddings(query, limit);
  return {
    content: [{
      type: "text" as const,
      text: results.length > 0
        ? results.map((t, i) => `[${i + 1}] ${t}`).join("\n\n---\n\n")
        : "No results found",
    }],
  };
});

server.registerTool("list_embeddings", {
  description: "List all stored embedding texts",
  inputSchema: z.object({}),
}, async () => {
  const allEmbeds = await store.getAllEmbeddings();
  const text = allEmbeds.length > 0
    ? allEmbeds.map((t, i) => `[${i + 1}] ${t}`).join("\n\n---\n\n")
    : "No embeddings stored";
  return {
    content: [{ type: "text" as const, text }],
  };
});

server.registerTool("get_store_status", {
  description: "Get embedding store type, connection status, and total embedding count",
  inputSchema: z.object({}),
}, async () => {
  const allEmbeds = await store.getAllEmbeddings();
  const info = {
    type: store.isInMemory ? "In-Memory" : "PostgreSQL (pgvector)",
    ready: store.isReady,
    embeddingModelLoaded: embeddingContext !== null,
    totalEmbeddings: allEmbeds.length,
  };
  return {
    content: [{ type: "text" as const, text: JSON.stringify(info, null, 2) }],
  };
});

async function main() {
  store = await createEmbeddingStore();

  const modelsPath = process.env["MODELS_PATH"] || "./models";
  const embeddingModelName = process.env["EMBEDDING_MODEL"];
  if (modelsPath && embeddingModelName) {
    const modelPath = path.join(modelsPath, embeddingModelName);
    try {
      const llama = await getLlama();
      const model = await llama.loadModel({ modelPath });
      embeddingContext = await model.createEmbeddingContext();
      console.error(`✓ Embedding model loaded from ${modelPath}`);
    } catch (error) {
      console.error(`⚠ Could not load embedding model: ${(error as Error).message}`);
      console.error("  Search will use text-based matching only");
    }
  } else {
    console.error("⚠ MODELS_PATH or EMBEDDING_MODEL not set, search will use text-based matching only");
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("✓ MCP server started on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
