# ts-RAG — Retrieval Augmented Generation System in TypeScript

A RAG (Retrieval Augmented Generation) system written in TypeScript that ingests PDF and Markdown documents, generates vector embeddings using local models via `node-llama-cpp`, stores them in PostgreSQL with pgvector (with a volatile in-memory fallback), and answers natural-language queries by retrieving semantically similar chunks. Query from the command line or via an MCP server.

## Architecture

```
Document (PDF/MD)
       │
       ▼
  main.ts ──► PdfProcessor / MarkdownProcessor
       │
       ├─ 1. parsePDF() / readFile() → split into chunks (~1024 characters)
       ├─ 2. LlamaEmbeddingContext.getEmbeddingFor() → 768d vectors
       ├─ 3. EmbeddingStore.addEmbeddings() → pgvector (or InMemoryStore)
       │
       ▼
  QueryProcessor
       ├─ 1. Generates the query embedding
       ├─ 2. pgvector <-> (cosine distance) → top 5 chunks
       ├─ 3. (Optional) cross-encoder reranker reorders results
       └─ 4. LlamaCompletion.generateCompletion() → answer with context
```

## MCP Server

The system exposes its capabilities as an **MCP server** (Model Context Protocol) so it can be integrated with clients such as Claude Desktop, OpenCode, or any MCP-compatible client. Two transport modes are available:

### 1. stdio (local)

`src/mcp-server.ts` — stdio transport for local clients that spawn the process directly:

```bash
npm run mcp-server
```

Example configuration (Claude Desktop / OpenCode):

```json
{
  "mcpServers": {
    "rag-embeddings": {
      "command": "npm",
      "args": ["run", "mcp-server"],
      "cwd": "/path/to/ts-RAG"
    }
  }
}
```

### 2. Streamable HTTP (network)

`src/mcp-http-server.ts` — HTTP transport with SSE sessions, intended for remote or browser-based clients:

```bash
npm run mcp-http
# → http://127.0.0.1:3000/mcp
```

Additional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HOST` | `127.0.0.1` | Listen interface |
| `MCP_PORT` | `3000` | HTTP port |
| `MCP_ENDPOINT` | `/mcp` | MCP endpoint path |
| `MCP_API_KEY` | *(empty)* | When set, requires `Authorization: Bearer <key>` (constant-time comparison) |
| `MCP_CORS_ORIGIN` | `*` | Allowed CORS origin for browser clients |

HTTP server features:

- **Lazy initialization**: the embedding model and store are loaded on the first request, not at startup
- **Operation serialization**: tool calls are queued to avoid concurrent use of the model
- **Per-client sessions**: each client gets its own `Mcp-Session-Id` and server instance
- **Graceful shutdown**: closes all active sessions on SIGINT/SIGTERM

### Exposed tools

Both servers register the same set of tools:

| Tool | Description |
|------|-------------|
| `create_embeddings` | Creates embeddings from a PDF (`pdf_path`) or plain text (`text`) and stores them in the database |
| `query_embeddings` | Finds chunks semantically similar to a query, ranked by cosine similarity |
| `list_embeddings` | Lists all texts stored in the store |
| `get_store_status` | Store status (type, ready, model loaded, total embeddings) |

In the stdio server, if the embedding model cannot be loaded, `search_embeddings` degrades to text-based matching.

## Main components

| File | Purpose |
|------|---------|
| `src/main.ts` | Entry point: loads `.env`, initializes `node-llama-cpp` with Metal GPU, orchestrates the full flow |
| `src/cli.ts` | Interactive interface and argument parsing (`--pdf`, `--query`) |
| `src/services/ragSystem.ts` | Abstract base class: loads models, detects dimension, initializes store |
| `src/services/processPdf.ts` | Processes PDFs: parsing → embeddings → storage → optional query |
| `src/services/processMarkdown.ts` | Processes Markdown files (same flow as PDF) |
| `src/services/processQuery.ts` | Queries existing embeddings: generates embedding → search → rerank → answer |
| `src/services/pdf-embeddings.ts` | Parsing with LangChain PDFLoader, chunking, embedding generation, cosine similarity |
| `src/services/query-engine.ts` | Answer engine: context prompt, LLM generation, keyword fallback |
| `src/services/reranker.ts` | Cross-encoder reranker that reorders chunks by relevance |
| `src/services/embedding-search.ts` | Embedding search and management utilities |
| `src/services/cleanup.service.ts` | Graceful shutdown: captures SIGINT/SIGTERM and stops PostgreSQL cleanly |
| `src/store/embedding-store.ts` | Store factory: tries PostgreSQL, falls back to InMemoryStore |
| `src/store/pgVectorStore.ts` | PostgreSQL + pgvector implementation: connection pool, `embeddings` table with `vector(768)` column, IVFFLAT index, upsert, cosine similarity search |
| `src/store/pgDaemon.ts` | Embedded PostgreSQL server: starts `initdb` and `postgres` as child processes |
| `src/store/inMemoryStore.ts` | In-memory fallback with `Map<string, number[]>` |
| `src/mcp-server.ts` | MCP server over stdio: exposes the embedding tools to local clients |
| `src/mcp-http-server.ts` | MCP server over Streamable HTTP: per-client sessions, optional Bearer auth, CORS, and lazy initialization |

## Storage

### PostgreSQL (primary)

`embeddings` table with an IVFFLAT index for ANN search:

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL UNIQUE,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

The `<->` operator (cosine distance) is used to retrieve the most similar chunks.

### In-memory (fallback)

Volatile `Map<string, number[]>` — data is lost when the process exits.

### Embedded PostgreSQL

If `POSTGRES_HOST=127.0.0.1`, the system automatically starts a local PostgreSQL server as a child process using `initdb` + `postgres`.

### Docker

`docker-compose.yml` runs `pgvector/pgvector:0.8.2-pg18-trixie` on port 5432.

## Required models

GGUF models must be placed in `MODELS_PATH` (configurable in `.env`):

| Variable | Model | Purpose |
|----------|-------|---------|
| `EMBEDDING_MODEL` | embeddinggemma-300M-Q8_0.gguf | 768d embedding generation |
| `QUERY_MODEL` | gemma-3-4b-it-Q4_K_M.gguf | LLM for RAG answers |
| `RERANKING_MODEL` | bge-reranker-v2-m3-Q8_0.gguf | Cross-encoder reranking |

## Environment variables (`.env`)

```
POSTGRES_PASSWORD=...
POSTGRES_USER=...
POSTGRES_HOST=...
POSTGRES_PORT=5434
POSTGRES_DATABASE=embeddings
POSTGRES_DATA_DIR=/path/to/data
MODELS_PATH=/path/to/models
EMBEDDING_MODEL=embeddinggemma-300M-Q8_0.gguf
EMBEDDING_DIMENSION=768
RERANKING_MODEL=bge-reranker-v2-m3-Q8_0.gguf
QUERY_MODEL=gemma-3-4b-it-Q4_K_M.gguf
```

## Usage

```bash
# Process a PDF and answer a query
npm start path/to/document.pdf "What does the document say about X?"

# Query existing embeddings (no PDF)
npm start -- --query "What is the capital of France?"

# Interactive mode (menu with 4 options)
npm start
```

## Available scripts

| Command | Description |
|---------|-------------|
| `npm start` | Runs with `vite-node` and Metal GPU optimizations |
| `npm run dev` | Development mode with `ts-node` |
| `npm run build` | TypeScript compilation to JavaScript |
| `npm run mcp-server` | Starts the MCP server over stdio |
| `npm run mcp-http` | Starts the MCP server over Streamable HTTP |
| `npm test` | Placeholder (no tests implemented) |

## Fault-tolerance strategy

The system incorporates multiple layers of gradual degradation:

1. **Database**: if PostgreSQL is unavailable → `InMemoryStore`
2. **Query model**: if the LLM cannot be loaded → keyword-based answer
3. **Reranker**: if it fails → results are returned without reranking
4. **Embeddings**: if a chunk cannot be embedded → that chunk is skipped
5. **Embedded PostgreSQL**: if `initdb` fails → tries external connection → falls back to memory
