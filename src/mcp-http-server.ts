import "dotenv/config";
import http from "node:http";
import path from "node:path";
import fs from "node:fs";
import { randomUUID, timingSafeEqual } from "node:crypto";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import {
  StreamableHTTPServerTransport,
} from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { getLlama, type Llama, type LlamaEmbeddingContext } from "node-llama-cpp";
import {
  createEmbeddingStore,
  type EmbeddingStore,
} from "./store/embedding-store.ts";
import { parsePDF, splitIntoChunks, embedDocuments } from "./services/pdf-embeddings.ts";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const HOST = process.env["MCP_HOST"] || "127.0.0.1";
const PORT = parseInt(process.env["MCP_PORT"] || "3000", 10);
const ENDPOINT = process.env["MCP_ENDPOINT"] || "/mcp";
// When set, clients must send: Authorization: Bearer <MCP_API_KEY>
const API_KEY = process.env["MCP_API_KEY"] || "";

function isAuthorized(req: http.IncomingMessage): boolean {
  if (!API_KEY) return true; // auth disabled when MCP_API_KEY is not set
  const header = req.headers["authorization"] ?? "";
  const match = /^Bearer\s+(.+)$/i.exec(header);
  if (!match?.[1]) return false;
  // Constant-time comparison to avoid timing attacks
  const provided = Buffer.from(match[1]);
  const expected = Buffer.from(API_KEY);
  if (provided.length !== expected.length) return false;
  return timingSafeEqual(provided, expected);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let store: EmbeddingStore | null = null;
let llama: Llama | null = null;
let embeddingContext: LlamaEmbeddingContext | null = null;
let initPromise: Promise<void> | null = null;

/** Serialize tool executions so the embedding model is never used concurrently. */
let opQueue: Promise<unknown> = Promise.resolve();
function enqueue<T>(fn: () => Promise<T>): Promise<T> {
  const run = opQueue.then(fn, fn);
  opQueue = run.catch(() => {});
  return run;
}

async function ensureInitialized(): Promise<void> {
  if (store && embeddingContext) return;
  if (!initPromise) initPromise = initialize();
  await initPromise;
}

async function initialize(): Promise<void> {
  try {
    const modelsPath = process.env["MODELS_PATH"] || "./models";
    const embeddingModelName = process.env["EMBEDDING_MODEL"];
    if (!embeddingModelName) {
      throw new Error("EMBEDDING_MODEL is not set in .env");
    }
    const modelPath = path.join(modelsPath, embeddingModelName);
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Embedding model not found at ${modelPath}`);
    }

    console.error(`⏳ Loading embedding model from ${modelPath}...`);
    llama = await getLlama({ gpu: "auto" });
    const model = await llama.loadModel({ modelPath });

    // Detect dimension from model (or use env override)
    let dimension = parseInt(process.env["EMBEDDING_DIMENSION"] || "");
    if (isNaN(dimension)) {
      const tempContext = await model.createEmbeddingContext();
      const testEmbedding = await tempContext.getEmbeddingFor("test");
      dimension = testEmbedding.vector?.length ?? 384;
    }

    store = await createEmbeddingStore(dimension);
    embeddingContext = await model.createEmbeddingContext();

    console.error(
      `✓ MCP server initialized (${store.isInMemory ? "In-Memory" : "PostgreSQL pgvector"}, ${dimension}d)`,
    );
  } catch (error) {
    initPromise = null; // allow retry on next request
    throw error;
  }
}

// ---------------------------------------------------------------------------
// MCP server + tools
// ---------------------------------------------------------------------------

/**
 * Build a fresh McpServer instance with all tools registered.
 * A separate server instance is needed per HTTP session because the SDK's
 * McpServer can only be connected to a single transport at a time.
 */
function createServer(): McpServer {
  const server = new McpServer(
    {
      name: "rag-embeddings-mcp",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    },
  );

  server.registerTool(
  "create_embeddings",
  {
    title: "Create and store embeddings",
    description:
      "Create embeddings from a PDF file or raw text and store them in the local pgvector database. " +
      "Provide either pdf_path or text (not both).",
    inputSchema: {
      source: z.enum(["pdf", "text"]).describe("Where the content comes from"),
      pdf_path: z
        .string()
        .optional()
        .describe("Absolute path to a PDF file (required when source='pdf')"),
      text: z
        .string()
        .optional()
        .describe("Raw text to embed (required when source='text')"),
      max_chunk_size: z
        .number()
        .int()
        .min(64)
        .max(4096)
        .default(1024)
        .describe("Maximum characters per chunk"),
    },
  },
  async (args) => {
    const { source, pdf_path, text, max_chunk_size } = args;
    return enqueue(async () => {
      await ensureInitialized();
      const ctx = embeddingContext!;
      const st = store!;

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

      const embeddings = await embedDocuments(ctx, chunks);
      if (embeddings.size === 0) {
        throw new Error("Failed to create any embeddings");
      }

      const metadata = {
        source,
        title: sourceName,
        date: new Date().toISOString(),
      };
      await st.addEmbeddings(chunks, embeddings, metadata);

      const total = (await st.getAllEmbeddings()).length;
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify(
              {
                stored: embeddings.size,
                totalInStore: total,
                storeType: st.isInMemory ? "in-memory" : "pgvector",
                source: sourceName,
              },
              null,
              2,
            ),
          },
        ],
      };
    });
  },
);

server.registerTool(
  "query_embeddings",
  {
    title: "Query embeddings",
    description:
      "Search the pgvector database for chunks semantically similar to the query, ranked by cosine similarity.",
    inputSchema: {
      query: z.string().describe("The natural-language query to search for"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(100)
        .default(10)
        .describe("Maximum number of results to return"),
    },
  },
  async (args) => {
    const { query, limit } = args;
    return enqueue(async () => {
      await ensureInitialized();
      const ctx = embeddingContext!;
      const st = store!;

      const embedding = await ctx.getEmbeddingFor(query);
      const vector = embedding.vector ? Array.from(embedding.vector) : [];
      if (vector.length === 0) {
        throw new Error("Failed to embed the query");
      }

      const results = await st.queryByEmbedding(vector, limit);
      if (results.length === 0) {
        return {
          content: [{ type: "text" as const, text: "No results found" }],
        };
      }

      const content = results
        .map(
          (r, i) =>
            `### Result ${i + 1} — ${(r.similarity * 100).toFixed(1)}% similar\n${r.text}`,
        )
        .join("\n\n---\n\n");
      return { content: [{ type: "text" as const, text: content }] };
    });
  },
);

server.registerTool(
  "list_embeddings",
  {
    title: "List embeddings",
    description: "List all stored embedding texts from the database.",
    inputSchema: {},
  },
  async () => {
    return enqueue(async () => {
      await ensureInitialized();
      const allEmbeds = await store!.getAllEmbeddings();
      const text =
        allEmbeds.length > 0
          ? allEmbeds.map((t, i) => `[${i + 1}] ${t}`).join("\n\n---\n\n")
          : "No embeddings stored";
      return { content: [{ type: "text" as const, text }] };
    });
  },
);

server.registerTool(
  "get_store_status",
  {
    title: "Store status",
    description:
      "Get the embedding store type, readiness, model status, and total embedding count.",
    inputSchema: {},
  },
  async () => {
    return enqueue(async () => {
      await ensureInitialized();
      const allEmbeds = await store!.getAllEmbeddings();
      const info = {
        type: store!.isInMemory ? "In-Memory" : "PostgreSQL (pgvector)",
        ready: store!.isReady,
        embeddingModelLoaded: embeddingContext !== null,
        totalEmbeddings: allEmbeds.length,
      };
      return {
        content: [{ type: "text" as const, text: JSON.stringify(info, null, 2) }],
      };
    });
  },
);

  return server;
}

// ---------------------------------------------------------------------------
// HTTP transport (Streamable HTTP, stateful sessions)
// ---------------------------------------------------------------------------

/** sessionId -> transport (one transport per MCP client session) */
const transports = new Map<string, StreamableHTTPServerTransport>();
/** transport -> McpServer (one server per session, connected to its transport) */
const serversByTransport = new WeakMap<StreamableHTTPServerTransport, McpServer>();

function readBody(req: http.IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      if (chunks.length === 0) return resolve(undefined);
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

function handleMcpRequest(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  sessionId: string | undefined,
): void {
  const method = req.method?.toUpperCase();

  if (method === "POST") {
    let transport = sessionId ? transports.get(sessionId) : undefined;

    if (sessionId && !transport) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ jsonrpc: "2.0", error: { code: -32001, message: "Session not found" }, id: null }));
      return;
    }

    if (!transport) {
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        // Register the session as soon as the server assigns a session ID
        // during the initialize handshake.
        onsessioninitialized: (sessionId) => {
          transports.set(sessionId, transport!);
        },
      });
      transport.onclose = () => {
        if (transport!.sessionId) {
          transports.delete(transport!.sessionId);
        }
      };
    }

    void (async () => {
      let body: unknown;
      try {
        body = await readBody(req);
      } catch (error) {
        console.error("Error reading request body:", error);
        if (!res.headersSent) {
          res.writeHead(400, { "Content-Type": "application/json" });
        }
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            error: { code: -32700, message: "Parse error: Invalid JSON body" },
            id: null,
          }),
        );
        return;
      }

      try {
        // Connect a per-session McpServer to this transport so incoming
        // JSON-RPC messages are routed to the registered tools and responses
        // are written back over the SSE stream. The server is created once per
        // transport (a transport maps 1:1 to a session) and reused for
        // subsequent requests on the same session.
        let mcpServer = serversByTransport.get(transport!);
        if (!mcpServer) {
          mcpServer = createServer();
          await mcpServer.connect(
            transport! as unknown as Parameters<typeof mcpServer.connect>[0],
          );
          serversByTransport.set(transport!, mcpServer);
        }
        await transport!.handleRequest(req, res, body);
      } catch (error) {
        console.error("Error handling MCP request:", error);
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "application/json" });
        }
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            error: { code: -32603, message: (error as Error).message },
            id: null,
          }),
        );
      }
    })();
    return;
  }

  // GET / DELETE require a valid session
  const transport = sessionId ? transports.get(sessionId) : undefined;
  if (!transport) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        jsonrpc: "2.0",
        error: { code: -32000, message: "Missing or invalid session" },
        id: null,
      }),
    );
    return;
  }

  void transport.handleRequest(req, res);
}

/**
 * CORS for browser-based MCP clients.
 * Set MCP_CORS_ORIGIN to the allowed origin (e.g. http://localhost:12434).
 * Defaults to "*" so local browser tools can connect; combine with MCP_API_KEY
 * for protection. Do NOT use "*" with credentials in production.
 */
const CORS_ORIGIN = process.env["MCP_CORS_ORIGIN"] || "*";

function corsHeaders(req: http.IncomingMessage): Record<string, string> {
  const origin = req.headers["origin"];
  // Echo the request origin only if it is explicitly allowed; otherwise use "*"
  const allow =
    CORS_ORIGIN === "*" ? "*" : origin && CORS_ORIGIN === origin ? origin : CORS_ORIGIN;
  const headers: Record<string, string> = {
    "Access-Control-Allow-Origin": allow,
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers":
      "Content-Type, Authorization, Accept, Mcp-Session-Id, Last-Event-ID",
    "Access-Control-Expose-Headers": "Mcp-Session-Id",
    "Access-Control-Max-Age": "86400",
  };
  return headers;
}

function startHttpServer(): http.Server {
  const httpServer = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);

    if (url.pathname !== ENDPOINT) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end(`Not found. MCP endpoint is ${ENDPOINT}`);
      return;
    }

    // Log every request so browser-side failures can be diagnosed
    console.error(
      `→ ${req.method} ${url.pathname} origin=${req.headers["origin"] ?? "-"} ` +
        `session=${(req.headers["mcp-session-id"] as string) ?? "-"} ` +
        `auth=${req.headers["authorization"] ? "yes" : "no"}`,
    );

    // CORS preflight: browsers send OPTIONS before any request that carries
    // custom headers (e.g. Authorization). Must be answered before auth.
    if (req.method?.toUpperCase() === "OPTIONS") {
      res.writeHead(204, corsHeaders(req));
      res.end();
      return;
    }

    // Attach CORS headers to every real response (including 401s, so the
    // browser can surface the error instead of a generic "Failed to fetch")
    const cors = corsHeaders(req);
    for (const [key, value] of Object.entries(cors)) {
      res.setHeader(key, value);
    }

    if (!isAuthorized(req)) {
      res.writeHead(401, {
        "Content-Type": "application/json",
        "WWW-Authenticate": 'Bearer realm="mcp"',
      });
      res.end(
        JSON.stringify({
          jsonrpc: "2.0",
          error: { code: -32001, message: "Unauthorized: missing or invalid Bearer token" },
          id: null,
        }),
      );
      return;
    }

    const sessionId = req.headers["mcp-session-id"] as string | undefined;
    handleMcpRequest(req, res, sessionId);
  });

  httpServer.listen(PORT, HOST, () => {
    console.error(`✓ MCP Streamable HTTP server listening on http://${HOST}:${PORT}${ENDPOINT}`);
  });

  return httpServer;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main(): void {
  const httpServer = startHttpServer();

  const shutdown = async (signal: string) => {
    console.error(`\n🛑 Received ${signal}, shutting down...`);
    try {
      for (const transport of transports.values()) {
        await transport.close();
      }
      httpServer.close();
      process.exit(0);
    } catch (error) {
      console.error("Error during shutdown:", error);
      process.exit(1);
    }
  };

  process.on("SIGINT", () => void shutdown("SIGINT"));
  process.on("SIGTERM", () => void shutdown("SIGTERM"));
}

main();
