# JS Embeddings - RAG System

A TypeScript project for building a Retrieval Augmented Generation (RAG) system that parses PDF files, creates embeddings, stores them persistently in PostgreSQL with pgvector, and answers questions using an AI model powered by Llama.cpp.

## Features

- **PDF Parsing**: Extract text content from PDF files using LangChain's PDFLoader
- **Embeddings Creation**: Generate embeddings from document text using embedding models (e.g., BGE-small)
- **Persistent Vector Storage**: Store embeddings in PostgreSQL with pgvector extension for scalable similarity search
- **Graceful Fallback**: If PostgreSQL is unavailable, embeddings automatically fall back to in-memory storage
- **Similarity Search**: Find relevant document chunks based on vector similarity using cosine distance
- **RAG (Retrieval Augmented Generation)**: Answer questions using a language model with retrieved context
- **Flexible Model Support**: Use different Llama.cpp models for embeddings and generation
- **High Performance**: IVFFLAT vector indexing for fast nearest-neighbor searches

## Architecture

```mermaid
┌─────────────────────────────────┐
│   PDF Document Processing       │
│   (pdf-embeddings.ts)           │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Embedding Generation          │
│   (node-llama-cpp)              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Embedding Storage Decision    │
│   (embedding-store.ts)          │
└────┬─────────────────────┬──────┘
     │                     │
  (Try)               (Fallback)
     │                     │
     ▼                     ▼
┌──────────────┐    ┌─────────────┐
│ PostgreSQL   │    │  In-Memory  │
│ + pgvector   │    │   Storage   │
│              │    │             │
│ • IVFFLAT    │    │ • Fast      │
│ • Persistent │    │ • Non-persist
│ • Scalable   │    │             │
└──────────────┘    └─────────────┘
```

## Prerequisites

- Node.js 18+ and npm
- **PostgreSQL 13+** with pgvector extension
- Two Llama.cpp compatible models:
  1. **Embedding Model** (required): `models/bge-small-en-v1.5-Q8_0.gguf` 
  2. **Language Model** (optional): `models/neural-chat-7b-v3-3-Q4_K_M.gguf` or similar

## Installation

### 1. Install PostgreSQL & pgvector

**macOS:**
```bash
brew install postgresql@15
brew install pgvector
brew services start postgresql@15
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install build-essential postgresql-server-dev-15
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
sudo systemctl start postgresql
```

See [PGVECTOR_SETUP.md](./PGVECTOR_SETUP.md) for detailed PostgreSQL setup.

### 2. Create Database

```bash
psql -U postgres
```

```sql
CREATE DATABASE embeddings;
\c embeddings
CREATE EXTENSION vector;
```

### 3. Install Node Dependencies

```bash
npm install
```

This creates `node_modules/` with all required packages including the `pg` client for PostgreSQL.

### 4. Configure Environment (Optional)

Create a `.env` file in your project root:

```bash
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=postgres
PG_DATABASE=embeddings
```

### 5. Download Required Models

**Embedding Model** (Go to [Hugging Face](https://huggingface.co/) and download):
```bash
# BGE-small embedding model
# Place at: models/bge-small-en-v1.5-Q8_0.gguf
```
## Usage

### Basic RAG Query

Run the application with a PDF file and query:
```bash
npm start /path/to/your/file.pdf "Your question here"
```

### Query Existing Embeddings

You can also query existing embeddings without providing a PDF file:
```bash
npm start
```
This will search through existing embeddings in PostgreSQL/pgvector and return relevant chunks.

### Examples

```bash
# Answer a question about a PDF
npm start documents/paper.pdf "What are the main findings?"

# Default query if not provided
npm start documents/guide.pdf

# With relative path
npm start ./my-document.pdf "Summarize the introduction"

# Query existing embeddings (no PDF needed)
npm start
```

### Development Mode

Watch mode with auto-reload:
```bash
npm run dev
```

### Build

## Workflow & Architecture

### Phase 1: Embedding Creation & Storage
1. Parse PDF and extract text content
2. Split text into manageable chunks (~1000 characters)
3. Generate embeddings for each chunk using the embedding model
4. **Store embeddings persistently in PostgreSQL with pgvector**
   - If PostgreSQL unavailable, automatic fallback to in-memory storage
   - Uses IVFFLAT vector indexing for fast similarity searches
   - Embeddings stored with metadata in `embeddings` table

### Phase 2: Query Processing
1. Generate embedding for the user's question
2. Search PostgreSQL pgvector for semantically similar chunks using vector similarity (cosine distance)
3. Retrieve top-K most relevant chunks as context
4. If no embeddings found in PostgreSQL, fallback to in-memory storage

### Phase 3: Answer Generation
1. Send the user's question + relevant context to the language model
2. Model generates an answer based on the provided context
3. Return formatted response to user

## Storage Architecture

```
┌─────────────────────────────────────────┐
│    PDF Documents → Embeddings           │
└────────────────┬────────────────────────┘
                 │
         ┌───────▼──────────┐
         │  Try PostgreSQL  │   
         │   with pgvector  │
         └───────┬──────────┘
              ✓  │  ✗
       ┌────────┘└─────────┐
       │                   │
    ┌──▼──────┐       ┌────▼────┐
    │PostgreSQL│      │Memory    │
    │pgvector  │      │Store     │
    └──────────┘      └──────────┘
```

## Architecture

```
┌─────────────────────────────────┐
│   PDF Document Processing       │
│   (pdf-embeddings.ts)           │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Embedding Generation          │
│   (node-llama-cpp)              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Embedding Storage Decision    │
│   (embedding-store.ts)          │
└────┬─────────────────────┬──────┘
     │                     │
  (Try)               (Fallback)
     │                     │
     ▼                     ▼
┌──────────────┐    ┌─────────────┐
│ PostgreSQL   │    │  In-Memory  │
│ + pgvector   │    │   Storage   │
│              │    │             │
│ • IVFFLAT    │    │ • Fast      │
│ • Persistent │    │ • Non-persist
│ • Scalable   │    │             │
└──────────────┘    └─────────────┘
```

## Project Structure

```
js-embeddings/
├── src/
│   ├── main.ts                 # Entry point, CLI handling
│   ├── pdf-embeddings.ts       # PDF parsing and embedding functions
│   ├── query-engine.ts         # RAG and question answering
│   ├── embedding-store.ts      # PostgreSQL pgvector + Memory storage management
│   └── embedding-search.ts     # Search and retrieval utilities
├── models/
│   ├── bge-small-en-v1.5-Q8_0.gguf         # Embedding model
│   └── neural-chat-7b-v3-3-Q4_K_M.gguf    # Language model (optional)
├── package.json
├── tsconfig.json
├── PGVECTOR_SETUP.md           # PostgreSQL/pgvector setup guide
└── README.md
```

## API Reference

### Embedding Storage (`embedding-store.ts`)

**`createEmbeddingStore(): Promise<EmbeddingStore>`**
- Creates embedding store (PostgreSQL with pgvector if available, in-memory fallback)
- Automatically checks PostgreSQL connectivity
- Configurable via environment variables (PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE)

**`storeEmbeddingsWithFallback(store, chunks, embeddings)`**
- Stores embeddings in PostgreSQL or memory with automatic fallback

**`PgVectorStore.queryByEmbedding(embedding, limit)`** (Advanced)
- Query PostgreSQL directly using vector similarity
- Returns results with cosine similarity scores
- Gracefully handles query failures by returning empty array for fallback

### Embedding Search (`embedding-search.ts`)

**`searchEmbeddings(store, query, limit): Promise<string[]>`**
- Searches stored embeddings for similar documents
- Returns top-N results

**`getStorageInfo(store): Object`**
- Returns storage type and location information

**`clearAllEmbeddings(store)`**
- Clears all stored embeddings installation script using cosine similarity
3. Retrieve top-K most relevant chunks as context

### Phase 3: Answer Generation
1. Send the user's question + relevant context to the language model
2. Model generates an answer based on the provided context
3. Return formatted response to user

## Project Structure

```
.
├── src/
│   ├── main.ts              # Entry point, CLI handling
│   ├── pdf-embeddings.ts    # PDF parsing and embedding functions
│   └── query-engine.ts      # RAG and question answering engine
├── models/
│   ├── bge-small-en-v1.5-Q8_0.gguf    # Embedding model
│   └── neural-chat-7b-v3-3-Q4_K_M.gguf # Language model (optional)
├── package.json
└── tsconfig.json
```

## API Functions

### PDF and Embeddings (`pdf-embeddings.ts`)

**`parsePDF(pdfPath: string): Promise<string[]>`**
- Parses a PDF file and splits content into chunks
- Returns array of text chunks

**`embedDocuments(context: LlamaEmbeddingContext, documents: readonly string[]): Promise<Map<string, LlamaEmbedding>>`**
- Creates embeddings for document chunks
- Handles errors gracefully

**`findSimilarDocuments(embedding: LlamaEmbedding, documentEmbeddings: Map): string[]`**
- Finds chunks similar to a query embedding
- Returns sorted by similarity score

### Query Engine (`query-engine.ts`)

**`createQueryEngine(llama: Llama, modelPath: string): Promise<LlamaContext | null>`**
- Loads a language model for question answering
- Returns null if model loading fails

**`queryWithContext(context: LlamaContext, query: string, documents: string[], maxResults: number): Promise<QueryResult>`**
- Generates answers using retrieved context
- Falls back to keyword matching if generation fails

**`formatQueryResult(result: QueryResult): string`**
- Formats the RAG result for display

## Dependencies

- **node-llama-cpp**: Llama.cpp bindings for Node.js
- **@langchain/community**: Document loaders
- **typescript**: Language
- **ts-node**: TypeScript runtime

## Configuration

### Chunk Size
Modify in `src/pdf-embeddings.ts`:
```typescript
const MAX_CONTEXT_CHARS = 1000; // Adjust based on model limits
```

### Model Selection
Change models in `src/main.ts`:
```typescript
const embeddingModelPath = "path/to/embedding/model.gguf";
const llmModelPath = "path/to/language/model.gguf";
```

## Troubleshooting

### "Model not found" Error
- Ensure model files are in the `models/` directory
- Check file paths in `main.ts`

### "Context too long" Error
- Reduce `MAX_CONTEXT_CHARS` in `pdf-embeddings.ts`
- Use a smaller model or a model with larger context window

### Language model not processing
- The system will fall back to showing relevant chunks
- Download a language model for full RAG functionality
- Ensure model is in GGUF format

### Query fails with no results
- If querying existing embeddings and getting no results, verify that embeddings exist in the database
- The system will automatically fall back to in-memory storage if PostgreSQL is unavailable

## Performance Tips

1. **Use GPU Acceleration**: Metal on macOS, CUDA on NVIDIA
2. **Adjust Context Size**: Larger chunks = better context but slower processing
3. **Limit Retrieved Results**: Use fewer chunks as context for faster generation
4. **Model Selection**: Smaller models (7B) for speed, larger (13B+) for quality

## Example Output

```
📄 Parsing PDF from: /path/to/document.pdf
✓ PDF parsed into 45 chunks
✓ Embeddings created successfully (45 chunks embedded)

❓ Query: "What is the main topic?"

✓ Found 3 relevant chunks

📚 Most relevant document chunks:
────────────────────────────────────────────────
[Chunk 1]
The main topic of this document is...

[Chunk 2]
Building on this, we can see...
────────────────────────────────────────────────

╔════════════════════════════════════════════════════════════════╗
║                    QUERY RESULT                                ║
╚════════════════════════════════════════════════════════════════╝

📝 Question: "What is the main topic?"
💡 Answer:
The main topic is... [generated by language model]
```

## License

ISC
