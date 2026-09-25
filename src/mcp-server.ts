import "dotenv/config";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { createEmbeddingStore, type EmbeddingStore } from "./store/embedding-store.ts";
import { getLlama, type LlamaEmbeddingContext } from "node-llama-cpp";
import { parsePDF, splitIntoChunks, embedDocuments } from "./services/pdf-embeddings.ts";
import path from "node:path";
import fs from "node:fs";

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
  description: "Search for similar text chunks in the embedding store using semantic similarity. Optionally restrict the search to a single catalog by name.",
  inputSchema: z.object({
    query: z.string().describe("The text query to search for"),
    limit: z.number().min(1).max(100).default(10).describe("Maximum number of results to return"),
    catalog: z.string().optional().describe("Catalog name to restrict the search to (omit to search all catalogs)"),
  }),
}, async (args) => {
  const { query, limit, catalog } = args;

  let catalogId: string | null = null;
  if (catalog) {
    const found = await store.catalogStore.getCatalog(catalog);
    if (!found) {
      throw new Error(`Catalog "${catalog}" not found`);
    }
    catalogId = found.id;
  }

  if (embeddingContext) {
    const embedding = await embeddingContext.getEmbeddingFor(query);
    const vector = embedding.vector ? Array.from(embedding.vector) : [];

    if (vector.length > 0) {
      const results = await store.queryByEmbedding(vector, limit, catalogId);
      const content = results.map((r) =>
        `[${(r.similarity * 100).toFixed(1)}%] ${r.text}`
      ).join("\n\n---\n\n");
      return {
        content: [{ type: "text" as const, text: content || "No results found" }],
      };
    }
  }

  const results = await store.getEmbeddings(query, limit, catalogId);
  return {
    content: [{
      type: "text" as const,
      text: results.length > 0
        ? results.map((t, i) => `[${i + 1}] ${t}`).join("\n\n---\n\n")
        : "No results found",
    }],
  };
});

server.registerTool("create_embeddings", {
  description: "Create embeddings from a PDF file or raw text and store them in the database. The catalog parameter is required: use an existing catalog name or the string 'new:<name>' to create a new one.",
  inputSchema: z.object({
    source: z.enum(["pdf", "text"]).describe("Where the content comes from"),
    pdf_path: z.string().optional().describe("Absolute path to a PDF file (required when source='pdf')"),
    text: z.string().optional().describe("Raw text to embed (required when source='text')"),
    max_chunk_size: z.number().int().min(64).max(4096).default(1024).describe("Maximum characters per chunk"),
    catalog: z.string().describe("Catalog name to store embeddings in, or 'new:<name>' to create a new catalog first"),
  }),
}, async (args) => {
  const { source, pdf_path, text, max_chunk_size, catalog } = args;

  if (!catalog || catalog.trim().length === 0) {
    throw new Error("catalog is required");
  }

  let catalogId: string;
  let catalogName: string;
  if (catalog.startsWith("new:")) {
    const newName = catalog.slice(4).trim();
    if (!newName) throw new Error("New catalog name cannot be empty");
    const created = await store.catalogStore.createCatalog(newName);
    catalogId = created.id;
    catalogName = created.name;
  } else {
    const found = await store.catalogStore.getCatalog(catalog);
    if (!found) {
      throw new Error(`Catalog "${catalog}" not found. Use 'new:<name>' to create it.`);
    }
    catalogId = found.id;
    catalogName = found.name;
  }

  if (!embeddingContext) {
    throw new Error("Embedding model is not loaded; cannot create embeddings");
  }

  let chunks: string[];
  let sourceName: string;

  if (source === "pdf") {
    if (!pdf_path) throw new Error("pdf_path is required when source='pdf'");
    const resolved = path.resolve(pdf_path);
    if (!fs.existsSync(resolved)) {
      throw new Error(`PDF file not found: ${resolved}`);
    }
    chunks = await parsePDF(resolved);
    sourceName = path.basename(resolved);
  } else {
    if (!text || text.trim().length === 0) {
      throw new Error("text is required when source='text'");
    }
    chunks = splitIntoChunks(text, max_chunk_size);
    sourceName = "text";
  }

  if (chunks.length === 0) {
    throw new Error("No content extracted from the input");
  }

  const embeddings = await embedDocuments(embeddingContext, chunks);
  if (embeddings.size === 0) {
    throw new Error("Failed to create any embeddings");
  }

  const metadata = {
    source,
    title: sourceName,
    date: new Date().toISOString(),
    catalog: catalogName,
  };
  await store.addEmbeddings(chunks, embeddings, metadata, catalogId);

  const total = (await store.getAllEmbeddings(catalogId)).length;
  return {
    content: [{
      type: "text" as const,
      text: JSON.stringify(
        {
          stored: embeddings.size,
          totalInCatalog: total,
          storeType: store.isInMemory ? "in-memory" : "pgvector",
          source: sourceName,
          catalog: catalogName,
        },
        null,
        2,
      ),
    }],
  };
});

server.registerTool("list_embeddings", {
  description: "List stored embedding texts, optionally restricted to a single catalog by name",
  inputSchema: z.object({
    catalog: z.string().optional().describe("Catalog name to list embeddings from (omit to list all)"),
  }),
}, async (args) => {
  const { catalog } = args;
  let catalogId: string | null = null;
  if (catalog) {
    const found = await store.catalogStore.getCatalog(catalog);
    if (!found) throw new Error(`Catalog "${catalog}" not found`);
    catalogId = found.id;
  }
  const allEmbeds = await store.getAllEmbeddings(catalogId);
  const text = allEmbeds.length > 0
    ? allEmbeds.map((t, i) => `[${i + 1}] ${t}`).join("\n\n---\n\n")
    : "No embeddings stored";
  return {
    content: [{ type: "text" as const, text }],
  };
});

server.registerTool("list_catalogs", {
  description: "List all embedding catalogs with their embedding counts",
  inputSchema: z.object({}),
}, async () => {
  const catalogs = await store.catalogStore.listCatalogs();
  const text = catalogs.length > 0
    ? catalogs
        .map((c) => {
          const count = c.embedding_count !== undefined
            ? c.embedding_count
            : await store.catalogStore.countEmbeddings(c.id);
          const desc = c.description ? ` — ${c.description}` : "";
          return `• ${c.name} [${count} embeddings]${desc}`;
        })
        .join("\n")
    : "No catalogs yet";
  return {
    content: [{ type: "text" as const, text }],
  };
});

server.registerTool("create_catalog", {
  description: "Create a new embedding catalog (or return the existing one if the name is already taken)",
  inputSchema: z.object({
    name: z.string().describe("Catalog name (unique, case-insensitive)"),
    description: z.string().optional().describe("Optional description of the catalog"),
  }),
}, async (args) => {
  const catalog = await store.catalogStore.createCatalog(args.name, args.description);
  return {
    content: [{
      type: "text" as const,
      text: JSON.stringify(catalog, null, 2),
    }],
  };
});

server.registerTool("delete_catalog", {
  description: "Delete a catalog and all embeddings stored in it (cascading)",
  inputSchema: z.object({
    name: z.string().describe("Catalog name to delete"),
  }),
}, async (args) => {
  const found = await store.catalogStore.getCatalog(args.name);
  if (!found) throw new Error(`Catalog "${args.name}" not found`);
  await store.catalogStore.deleteCatalog(found.id);
  return {
    content: [{ type: "text" as const, text: `✓ Deleted catalog "${found.name}"` }],
  };
});

server.registerTool("get_store_status", {
  description: "Get embedding store type, connection status, total embedding count, and catalog list",
  inputSchema: z.object({}),
}, async () => {
  const allEmbeds = await store.getAllEmbeddings();
  const catalogs = await store.catalogStore.listCatalogs();
  const info = {
    type: store.isInMemory ? "In-Memory" : "PostgreSQL (pgvector)",
    ready: store.isReady,
    embeddingModelLoaded: embeddingContext !== null,
    totalEmbeddings: allEmbeds.length,
    catalogs: catalogs.map((c) => ({
      name: c.name,
      description: c.description,
      embeddings: c.embedding_count ?? null,
    })),
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
