# JS Embeddings - RAG System

A TypeScript project for building a Retrieval Augmented Generation (RAG) system that parses PDF files, creates embeddings, stores them persistently with ChromaDB/SQLite, and answers questions using an AI model powered by Llama.cpp.

## Features

- **PDF Parsing**: Extract text content from PDF files using LangChain's PDFLoader
- **Embeddings Creation**: Generate embeddings from document text using embedding models (e.g., BGE-small)
- **Persistent Storage**: Store embeddings in ChromaDB with SQLite backend for persistence
- **Graceful Fallback**: If ChromaDB fails, embeddings are stored in memory automatically
- **Similarity Search**: Find relevant document chunks based on semantic similarity to queries
- **RAG (Retrieval Augmented Generation)**: Answer questions using a language model with retrieved context
- **Flexible Model Support**: Use different Llama.cpp models for embeddings and generation

## Prerequisites

- Node.js 18+ and npm
- Two Llama.cpp compatible models:
  1. **Embedding Model** (required): `models/bge-small-en-v1.5-Q8_0.gguf` 
  2. **Language Model** (optional): `models/neural-chat-7b-v3-3-Q4_K_M.gguf` or similar

## Installation

1. Install dependencies (including ChromaDB):
```bash
npm install
# or use the install script
bash install.sh
```

This creates:
- `node_modules/` - All npm packages including ChromaDB
- `data/` - Directory for persistent SQLite database (chromadb.db)

2. Download required models (as before)

**Embedding Model** (Go to [Hugging Face](https://huggingface.co/) and download):
```bash
# BGE-small embedding model
# Place at: models/bge-small-en-v1.5-Q8_0.gguf
```

**Language Model** (Optional, for question answering):
```bash
# Examples:
# - neural-chat-7b-v3-3-Q4_K_M.gguf (recommended for most cases)
# - mistral-7b-instruct-v0.2.Q4_K_M.gguf
# - zephyr-7b-beta.Q4_K_M.gguf
# Place at: models/neural-chat-7b-v3-3-Q4_K_M.gguf
```

## Usage

### Basic RAG Query

Run the application with a PDF file and query:
```bash
npm start /path/to/your/file.pdf "Your question here"
```

### Examples

```bash
# Answer a question about a PDF
npm start documents/paper.pdf "What are the main findings?"

# Default query if not provided
npm start documents/guide.pdf

# With relative path
npm start ./my-document.pdf "Summarize the introduction"
```

### Development Mode

Watch mode with auto-reload:
```bash
npm run dev
```

### Build

Compile TypeScript to JavaScript:
```bash & Storage
1. Parse PDF and extract text content
2. Split text into manageable chunks (~1000 characters)
3. Generate embeddings for each chunk using the embedding model
4. **Store embeddings persistently in ChromaDB (SQLite backend)**
   - If ChromaDB fails, automatic fallback to in-memory storage
   - SQLite database saved in `data/embeddings.db` for persistence across sessions

### Phase 2: Query Processing
1. Generate embedding for the user's question
2. Search ChromaDB/memory for semantically similar chunks
3. Retrieve top-K most relevant chunks as context

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
         ┌───────▼────────┐
         │  Try ChromaDB   │   # Entry point, CLI handling
│   ├── pdf-embeddings.ts       # PDF parsing and embedding functions
│   ├── query-engine.ts         # RAG and question answering
│   ├── embedding-store.ts      # ChromaDB + Memory storage management
│   └── embedding-search.ts     # Search and retrieval utilities
├── data/
│   └── embeddings.db           # SQLite database (persistent embeddings)
├── models/
│   ├── bge-small-en-v1.5-Q8_0.gguf         # Embedding model
│   └── neural-chat-7b-v3-3-Q4_K_M.gguf    # Language model (optional)
├── package.json
├── tsconfig.json
├── Embedding Storage (`embedding-store.ts`)

**`createEmbeddingStore(): Promise<EmbeddingStore>`**
- Creates embedding store (ChromaDB if available, Memory fallback)
- Checks database connectivity automatically

**`storeEmbeddingsWithFallback(store, chunks, embeddings)`**
- Stores embeddings in ChromaDB or memory with automatic fallback

### Embedding Search (`embedding-search.ts`)

**`searchEmbeddings(store, query, context, limit): Promise<string[]>`**
- Searches stored embeddings for similar documents
- Returns top-N results

**`getStorageInfo(store): Object`**
- Returns storage type and location information

**`clearAllEmbeddings(store)`**
- Clears all stored embeddings installation script
└── README.md──────────────────┘
                 │
      ┌──────────┴──────────┐
      │ On Failure/Error    │
      └──────┬───────────────┘
             │
    ┌────────▼──────────┐
    │ In-Memory Store   │
    │ [VOLATILE - Session │
    │  only]             │
    └────────────────────┘
```using cosine similarity
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

**`embedDocuments(context: LlamaContext, documents: readonly string[]): Promise<Map<string, LlamaEmbedding>>`**
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
