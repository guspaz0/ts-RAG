# ts-RAG — Sistema de Retrieval Augmented Generation en TypeScript

Sistema RAG (Retrieval Augmented Generation) escrito en TypeScript que ingiere documentos PDF y Markdown, genera embeddings vectoriales usando modelos locales vía `node-llama-cpp`, los almacena en PostgreSQL con pgvector (con fallback a memoria volátil) y responde consultas en lenguaje natural recuperando fragmentos semánticamente similares. Consulta por terminal de comandos o Mcp server.

## Arquitectura

```
Documento (PDF/MD)
       │
       ▼
  main.ts ──► PdfProcessor / MarkdownProcessor
       │
       ├─ 1. parsePDF() / readFile() → división en fragmentos (~1024 caracteres)
       ├─ 2. LlamaEmbeddingContext.getEmbeddingFor() → vectores 768d
       ├─ 3. EmbeddingStore.addEmbeddings() → pgvector (o InMemoryStore)
       │
       ▼
  QueryProcessor
       ├─ 1. Genera embedding de la consulta
       ├─ 2. pgvector <-> (distancia coseno) → top 5 fragmentos
       ├─ 3. (Opcional) reranker cross-encoder reordena resultados
       └─ 4. LlamaCompletion.generateCompletion() → respuesta con contexto
```

## Componentes principales

| Archivo | Función |
|---------|---------|
| `src/main.ts` | Punto de entrada: carga `.env`, inicializa `node-llama-cpp` con GPU Metal, orquesta el flujo completo |
| `src/cli.ts` | Interfaz interactiva y parsing de argumentos (`--pdf`, `--query`) |
| `src/services/ragSystem.ts` | Clase base abstracta: carga modelos, detecta dimensión, inicializa store |
| `src/services/processPdf.ts` | Procesa PDFs: parseo → embeddings → almacenamiento → consulta opcional |
| `src/services/processMarkdown.ts` | Procesa archivos Markdown (mismo flujo que PDF) |
| `src/services/processQuery.ts` | Consulta embeddings existentes: genera embedding → busca → rerank → responde |
| `src/services/pdf-embeddings.ts` | Parseo con LangChain PDFLoader, división en fragmentos, generación de embeddings, similitud coseno |
| `src/services/query-engine.ts` | Motor de respuestas: prompt con contexto, generación con LLM, fallback por palabras clave |
| `src/services/reranker.ts` | Cross-encoder reranker que reordena fragmentos por relevancia |
| `src/services/embedding-search.ts` | Utilidades de búsqueda y gestión de embeddings |
| `src/services/cleanup.service.ts` | Apagado graceful: captura señales SIGINT/SIGTERM y detiene PostgreSQL limpiamente |
| `src/store/embedding-store.ts` | Fábrica de stores: intenta PostgreSQL, fallback a InMemoryStore |
| `src/store/pgVectorStore.ts` | Implementación PostgreSQL + pgvector: pool de conexiones, tabla `embeddings` con columna `vector(768)`, índice IVFFLAT, upsert, búsqueda por similitud coseno |
| `src/store/pgDaemon.ts` | Servidor PostgreSQL embebido: inicia `initdb` y `postgres` como proceso hijo |
| `src/store/inMemoryStore.ts` | Fallback en memoria con `Map<string, number[]>` |

## Almacenamiento

### PostgreSQL (primario)

Tabla `embeddings` con índice IVFFLAT para búsqueda ANN:

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL UNIQUE,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Se usa el operador `<->` (distancia coseno) para recuperar los fragmentos más similares.

### In-memory (fallback)

`Map<string, number[]>` volatile — los datos se pierden al terminar el proceso.

### PostgreSQL embebido

Si `POSTGRES_HOST=127.0.0.1`, el sistema inicia automáticamente un servidor PostgreSQL local como proceso hijo usando `initdb` + `postgres`.

### Docker

`docker-compose.yml` levanta `pgvector/pgvector:0.8.2-pg18-trixie` en puerto 5432.

## Modelos requeridos

Los modelos en formato GGUF deben estar en `MODELS_PATH` (configurable en `.env`):

| Variable | Modelo | Propósito |
|----------|--------|-----------|
| `EMBEDDING_MODEL` | embeddinggemma-300M-Q8_0.gguf | Generación de embeddings 768d |
| `QUERY_MODEL` | gemma-3-4b-it-Q4_K_M.gguf | LLM para respuestas RAG |
| `RERANKING_MODEL` | bge-reranker-v2-m3-Q8_0.gguf | Reranking cross-encoder |

## Variables de entorno (`.env`)

```
POSTGRES_PASSWORD=...
POSTGRES_USER=...
POSTGRES_HOST=...
POSTGRES_PORT=5434
POSTGRES_DATABASE=embeddings
POSTGRES_DATA_DIR=/ruta/data
MODELS_PATH=/ruta/modelos
EMBEDDING_MODEL=embeddinggemma-300M-Q8_0.gguf
EMBEDDING_DIMENSION=768
RERANKING_MODEL=bge-reranker-v2-m3-Q8_0.gguf
QUERY_MODEL=gemma-3-4b-it-Q4_K_M.gguf
```

## Uso

```bash
# Procesar PDF y responder consulta
npm start ruta/documento.pdf "¿Qué dice el documento sobre X?"

# Consultar embeddings existentes (sin PDF)
npm start -- --query "¿Cuál es la capital de Francia?"

# Modo interactivo (menú con 4 opciones)
npm start
```

## Scripts disponibles

| Comando | Descripción |
|---------|-------------|
| `npm start` | Ejecuta con `vite-node` y optimizaciones Metal GPU |
| `npm run dev` | Modo desarrollo con `ts-node` |
| `npm run build` | Compilación TypeScript a JavaScript |
| `npm test` | Placeholder (sin tests implementados) |

## Estrategia de tolerancia a fallos

El sistema incorpora múltiples capas de degradación gradual:

1. **Base de datos**: si PostgreSQL no está disponible → `InMemoryStore`
2. **Modelo de consulta**: si el LLM no se puede cargar → respuesta por palabras clave
3. **Reranker**: si falla → se omiten los resultados sin reranking
4. **Embeddings**: si un fragmento no se puede embedding → se omite ese fragmento
5. **PostgreSQL embebido**: si `initdb` falla → se intenta conexión externa → fallback a memoria
