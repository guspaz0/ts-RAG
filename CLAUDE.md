# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase Overview

This is a TypeScript-based Retrieval Augmented Generation (RAG) system that processes PDF documents, creates vector embeddings, and enables question-answering capabilities using LLMs. The system uses PostgreSQL with pgvector for persistent storage of embeddings and includes a graceful fallback to in-memory storage.

## Key Architecture Components

- **main.ts**: Entry point that orchestrates the entire workflow - PDF processing, embedding creation, storage, and query processing
- **pdf-embeddings.ts**: Handles PDF parsing and embedding generation using node-llama-cpp
- **embedding-store.ts**: Manages storage of embeddings with PostgreSQL/pgvector as primary storage and in-memory fallback
- **query-engine.ts**: Processes queries and generates answers using language models when available

## Key Features

- PDF parsing with LangChain's PDFLoader
- Embedding generation using BGE-small model
- Persistent vector storage in PostgreSQL with pgvector
- Graceful fallback to in-memory storage when PostgreSQL is unavailable
- Query processing that can work with existing embeddings (no PDF required)
- Similarity search using cosine distance
- RAG (Retrieval Augmented Generation) with language model support

## Common Development Commands

- `npm start <pdf-path> "<query>"` - Process PDF and answer query
- `npm start` - Query existing embeddings without PDF
- `npm run dev` - Development watch mode with auto-reload
- `npm run build` - TypeScript compilation
- `npm test` - Currently not implemented (placeholder)

## Development Workflow

1. Set up PostgreSQL with pgvector extension
2. Download required models to `models/` directory
3. Configure environment variables in `.env` file
4. Run with `npm start` or `npm run dev`
5. Use `npm run build` for production compilation