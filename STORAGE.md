# ChromaDB & SQLite Storage Guide

## Overview

JS Embeddings now supports persistent embedding storage using:
- **ChromaDB**: Vector database optimized for embeddings
- **SQLite**: Lightweight relational database for persistence
- **In-Memory Fallback**: Automatic fallback if ChromaDB fails

## Architecture

### Default Flow (ChromaDB + SQLite)

```
┌──────────────────────────────┐
│   Process PDF & Create       │
│   Embeddings                 │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   Try ChromaDB Connection    │
│   - Check if available       │
│   - Initialize client        │
└──────────┬───────────────────┘
           │
        ┌──┴──┐
        │     │
    ✓ Yes   ✗ No
        │     │
        │     ▼
        │  ┌────────────────────┐
        │  │ Use In-Memory      │
        │  │ Store (Fallback)   │
        │  └────────────────────┘
        │     │
        │     │
        ▼     ▼
    ┌──────────────────────────┐
    │  Store Embeddings        │
    │  - Maps embeddings to    │
    │    vector database       │
    └──────────────────────────┘
```

### Persistence

**ChromaDB + SQLite:**
- Embeddings persist across sessions
- Database stored in `data/embeddings.db`
- Survives application restarts
- Can be backed up and restored

**In-Memory Fallback:**
- Embeddings lost on application exit
- Fast but temporary
- Good for testing/development

## Usage

### Automatic Storage

The system automatically stores embeddings:

```bash
npm start document.pdf "your question"
```

**What happens:**
1. ✓ PDF is parsed
2. ✓ Embeddings are created
3. ✓ Automatically stored in ChromaDB or memory
4. ✓ Used for answering questions

### Manual Storage Operations

Create a script like `manage-embeddings.ts`:

```typescript
import { createEmbeddingStore } from "./src/embedding-store.ts";
import { getStorageInfo, clearAllEmbeddings } from "./src/embedding-search.ts";

const store = await createEmbeddingStore();

// Check storage
const info = getStorageInfo(store);
console.log("Storage:", info);
// Output: Storage: {
//   type: "ChromaDB + SQLite",
//   persistent: true,
//   location: "data/embeddings.db"
// }

// Clear all embeddings
await clearAllEmbeddings(store);
```

## Database Details

### SQLite Database

**Location:** `data/embeddings.db`

**Size:** Depends on number of embeddings
- ~100 PDF chunks: 10-50 MB
- ~1000 PDF chunks: 100-500 MB

**Inspection:**
```bash
# List all tables
sqlite3 data/embeddings.db ".tables"

# View schema
sqlite3 data/embeddings.db ".schema"

# Export data
sqlite3 data/embeddings.db ".dump" > backup.sql
```

### Backup & Restore

**Backup embeddings:**
```bash
cp data/embeddings.db data/embeddings.db.backup
```

**Restore from backup:**
```bash
cp data/embeddings.db.backup data/embeddings.db
```

**Clear database:**
```bash
rm data/embeddings.db
# Next run will create fresh database
```

## Error Handling

### Scenario 1: ChromaDB Connection Failed
```
⚠ ChromaDB initialization failed: Connection refused
  Falling back to in-memory storage
✓ In-Memory Store initialized
```
**What to do:**
- Embeddings will be stored in RAM (temporary)
- Restart application to reload from disk when ChromaDB works

### Scenario 2: SQLite Database Corrupted
```
⚠ Failed to store embeddings in ChromaDB
```
**What to do:**
1. Delete corrupted database: `rm data/embeddings.db`
2. Restart application
3. New database created automatically

### Scenario 3: Disk Space Full
**What to do:**
1. Backup current database: `cp data/embeddings.db data/embeddings.db.backup`
2. Clear database: `rm data/embeddings.db`
3. Free up disk space
4. Restart application

## Best Practices

### For Development

1. Use in-memory storage (ephemeral)
2. Faster iteration
3. No persistence needed

```typescript
// Use in-memory for testing
const store = new InMemoryStore();
```

### For Production

1. Enable ChromaDB with SQLite
2. Set up regular backups
3. Monitor database size
4. Clean old embeddings periodically

```bash
# Backup frequently
0 2 * * * cp data/embeddings.db data/embeddings.db.$(date +%Y%m%d)
```

### Maintenance

**Regular cleanup:**
```bash
# Keep only last 7 backups
ls -t data/embeddings.db.* | tail -n +8 | xargs rm
```

**Monitor size:**
```bash
du -sh data/embeddings.db
```

## Performance Tips

### ChromaDB Optimization

1. **Batch Processing**: Add multiple PDFs before querying
2. **Index**: Automatically optimized for cosine similarity search
3. **Query Speed**: O(log n) for similarity search

### Memory Usage

- **Embeddings**: ~1KB per chunk (depends on model)
- **Database Overhead**: ~2-3x embedding size

### Scaling

For large document collections:
1. Split PDFs into multiple processes
2. Use batch insertion
3. Regular index optimization
4. Archive old embeddings periodically

## Troubleshooting

### Q: Where is my data stored?
**A:** `data/embeddings.db` (SQLite) or in-memory (temporary)

### Q: How do I backup?
**A:** `cp data/embeddings.db backup.db`

### Q: Can I use existing ChromaDB database?
**A:** Yes, point to the existing database path in code

### Q: What if database gets corrupted?
**A:** Delete it and let the system create a fresh one

### Q: How do I export embeddings?
**A:** Use `sqlite3` CLI to dump database to SQL

## Advanced: Custom Storage Backend

Implement `EmbeddingStore` interface for custom backends:

```typescript
interface EmbeddingStore {
    isReady: boolean;
    isInMemory: boolean;
    addEmbeddings(chunks: string[], embeddings: Map<string, LlamaEmbedding>): Promise<void>;
    getEmbeddings(query: string, limit: number): Promise<string[]>;
    clear(): Promise<void>;
}
```

Example: PostgreSQL backend, Redis cache, etc.

## Support

For issues with ChromaDB or SQLite:
1. Check `data/` directory permissions
2. Verify disk space available
3. Check SQLite database integrity: `sqlite3 data/embeddings.db "PRAGMA integrity_check;"`
4. Review application logs for detailed errors
