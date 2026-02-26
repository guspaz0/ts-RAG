# JS Embeddings - Usage Guide

## Quick Start

### 1. Check Configuration
```bash
npm run check
```

This will verify that all required components are installed and configured.

### 2. Install with ChromaDB Support
```bash
bash install.sh
# or manually:
npm install
mkdir -p data
```

This installs ChromaDB (for persistent embeddings) and creates the `data/` directory for SQLite storage.

### 3. Verify Dependencies
```bash
npm install
```

### 3. Download Models

Download the following GGUF models and place them in the `models/` directory:

#### Required: Embedding Model
- **Model**: BGE-small-en-v1.5 (quantized Q8_0)
- **Download**: https://huggingface.co/BAAI/bge-small-en-v1.5/tree/main
- **File**: `bge-small-en-v1.5-Q8_0.gguf`
- **Size**: ~130MB
- **Purpose**: Generates embeddings for document chunks and queries

#### Optional: Language Model  
For full RAG (Retrieval Augmented Generation) capabilities, download one of:

- **Neural Chat 7B** (Recommended)
  - https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF
  - File: `neural-chat-7b-v3-3-Q4_K_M.gguf`
  - Size: ~5GB
  - Good balance of speed and quality

- **Mistral 7B Instruct**
  - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
  - File: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
  - Size: ~5GB
  - Better instruction following

- **Zephyr 7B Beta**
  - https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF
  - File: `zephyr-7b-beta.Q4_K_M.gguf`
  - Size: ~5GB
  - Great for chat and Q&A

### 4. Run Your First Query
```bash
npm start path/to/your/document.pdf "What is the main topic?"
```

**Storage Behavior:**
- ✓ Embeddings automatically stored in ChromaDB
- ✓ SQLite database created in `data/embeddings.db`
- ✓ Embeddings persist across sessions
- ✓ If ChromaDB fails, falls back to in-memory storage

## Persistent Storage

### ChromaDB + SQLite

Your embeddings are now automatically stored in a persistent SQLite database:

```
First Run:
  PDF → Embeddings → ChromaDB → data/embeddings.db
  ✓ Embeddings saved
  
Second Run:
  PDF → Same collection in database
  ✓ Can reuse embeddings
```

### Database Location
```
project-root/
└── data/
    └── embeddings.db  # SQLite database with all embeddings
```

### Check Storage Info
```bash
# View storage status
npm start document.pdf "query" | grep -i "chromadb\|memory"
```

### Manage Embeddings

Clear all stored embeddings:
```bash
# Delete the database - will be recreated on next run
rm data/embeddings.db
```

Backup your embeddings:
```bash
cp data/embeddings.db data/embeddings.db.backup
```

See [STORAGE.md](./STORAGE.md) for detailed storage documentation.

### 5. Run Your First Query

## Detailed Usage Examples

### Example 1: Simple Document Analysis
```bash
npm start ./documents/paper.pdf
```
Default query: "What is the content about?"

### Example 2: Specific Question
```bash
npm start ./documents/technical-guide.pdf "Explain the installation process"
```

### Example 3: Multi-word Query
```bash
npm start "./important document.pdf" "What are the key findings in Chapter 3?"
```

### Example 4: Development Mode
```bash
npm run dev
```
Watches files for changes and restarts automatically.

## Output Format

When you run a query, you'll see:

```
📄 Parsing PDF from: /path/to/document.pdf
✓ PDF parsed into 45 chunks
✓ Embeddings created successfully (45 chunks embedded)

❓ Query: "What is the main topic?"
✓ Found 3 relevant chunks

📚 Most relevant document chunks:
────────────────────────────────────────────────
[Chunk 1]
...snippet of content...

[Chunk 2]
...snippet of content...

[Chunk 3]
...snippet of content...
────────────────────────────────────────────────

╔════════════════════════════════════════════════════════════════╗
║                    QUERY RESULT                                ║
╚════════════════════════════════════════════════════════════════╝

📝 Question: "What is the main topic?"
💡 Answer:
...AI-generated answer based on context...
────────────────────────────────────────────────────────────────
```

## Advanced Configuration

### Adjusting Chunk Size
If you get "context too long" errors, reduce the chunk size in `src/pdf-embeddings.ts`:

```typescript
// Reduce from 1000 to 500 characters per chunk
const MAX_CONTEXT_CHARS = 500;
```

### Using Different Models
Modify `src/main.ts` to use different models:

```typescript
const llmModelPath = path.join(__dirname, "..", "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf");
```

### Changing GPU Provider
In `src/main.ts`, change the `gpu` option:

```typescript
// For NVIDIA GPUs
const llama = await getLlama({ gpu: 'cuda' });

// For AMD GPUs
const llama = await getLlama({ gpu: 'rocm' });

// Disable GPU (CPU only)
const llama = await getLlama({ gpu: 'cpu' });
```

### Adjusting Language Model Parameters
In `src/query-engine.ts`, modify generation parameters:

```typescript
// Generation parameters in queryWithContext function:
{
    maxTokens: 512,      // Max tokens to generate
    temperature: 0.7,    // Randomness (0-1, higher = more creative)
    topP: 0.9,          // Nucleus sampling
}
```

## Troubleshooting

### Issue: "Cannot find module 'node-llama-cpp'"
```bash
# Solution: Install dependencies
npm install
```

### Issue: "Model not found" Error
```bash
# Check what's in the models directory
ls -la models/

# Verify model paths in src/main.ts
# Download missing models from Hugging Face
```

### Issue: "Input is longer than context size"
```bash
# Solution 1: Reduce chunk size
# Edit MAX_CONTEXT_CHARS in src/pdf-embeddings.ts

# Solution 2: Use a model with larger context window
# Example: mistral models have 32k context
```

### Issue: Slow Performance
- Use a smaller model (e.g., 7B instead of 13B)
- Enable GPU acceleration
- Reduce MAX_CONTEXT_CHARS for faster processing
- Use fewer chunks in query results (modify maxResults in main.ts)

### Issue: Inaccurate Answers
- Use a larger/better language model
- Provide more context chunks (increase maxResults)
- Rephrase your question more specifically
- Ensure the model has sufficient instruction knowledge

## Performance Tips

1. **GPU Acceleration**
   - Metal (macOS): Usually auto-enabled
   - CUDA (NVIDIA): Install NVIDIA GPU support
   - ROCm (AMD): Install ROCm drivers

2. **Model Size vs Speed**
   - 7B parameters: ~5GB, ~50ms per token (reasonably fast)
   - 13B parameters: ~8GB, ~100ms per token (slower)
   - 70B parameters: ~45GB, too slow for most use cases

3. **Quantization**
   - Q4_K_M: Good balance (smaller, fast)
   - Q5_K_M: Better quality, larger
   - Q6_K: Best quality, very large
   - Q8_0: Highest precision, largest

4. **Batch Processing**
   - Process multiple PDFs and cache embeddings
   - Reuse embeddings across multiple queries
   - Creates a knowledge base effect

## API Integration

You can use this as a library in your own code:

```typescript
import { parsePDF, embedDocuments, findSimilarDocuments } from "./src/pdf-embeddings.ts";
import { queryWithContext } from "./src/query-engine.ts";

// Your custom code here
```

## Environment Variables (Optional)

Create a `.env` file:
```bash
# Model paths
EMBEDDING_MODEL_PATH=./models/bge-small-en-v1.5-Q8_0.gguf
LLM_MODEL_PATH=./models/neural-chat-7b-v3-3-Q4_K_M.gguf

# GPU
USE_GPU=true
GPU_TYPE=metal

# Processing
MAX_CHUNK_SIZE=1000
MAX_CONTEXT_RESULTS=3
```

## Building for Production

```bash
# Compile TypeScript
npm run build

# Output will be in dist/ directory
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review README.md for more details
3. Check JavaScript console for detailed error messages
4. Ensure all models are properly downloaded and placed

Enjoy using JS Embeddings RAG System! 🚀
