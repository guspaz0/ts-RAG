#!/usr/bin/env node

import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

console.log("═══════════════════════════════════════════════════════");
console.log("  JS Embeddings - Configuration Check");
console.log("═══════════════════════════════════════════════════════\n");

const checks = {
  "Node.js": checkNodeVersion(),
  "Dependencies": checkDependencies(),
  "Embedding Model": checkEmbeddingModel(),
  "Language Model": checkLanguageModel(),
};

let allGood = true;
for (const [name, status] of Object.entries(checks)) {
  const icon = status.ok ? "✓" : "✗";
  const color = status.ok ? "\x1b[32m" : "\x1b[33m";
  const reset = "\x1b[0m";
  
  console.log(`${color}${icon}${reset} ${name}: ${status.message}`);
  if (!status.ok && status.details) {
    console.log(`  → ${status.details}`);
  }
  
  if (!status.ok && status.required) {
    allGood = false;
  }
}

console.log("\n═══════════════════════════════════════════════════════");

if (allGood) {
  console.log("\n✓ All required components are configured!");
  console.log("\nYou can now run:");
  console.log("  npm start /path/to/your/file.pdf \"Your question here\"");
} else {
  console.log("\n⚠ Some required components are missing.");
  console.log("\nTo get started:");
  console.log("  1. Download the BGE embedding model:");
  console.log("     models/bge-small-en-v1.5-Q8_0.gguf");
  console.log("  2. (Optional) Download a language model:");
  console.log("     models/neural-chat-7b-v3-3-Q4_K_M.gguf");
}

console.log("═══════════════════════════════════════════════════════\n");

function checkNodeVersion() {
  const version = process.version;
  const majorVersion = parseInt(version.slice(1).split(".")[0]);
  
  if (majorVersion >= 18) {
    return {
      ok: true,
      message: `${version} (required: 18+)`,
      required: true,
    };
  }
  
  return {
    ok: false,
    message: `${version} (required: 18+)`,
    details: "Please upgrade Node.js to version 18 or higher",
    required: true,
  };
}

function checkDependencies() {
  try {
    require("node-llama-cpp");
    require("@langchain/community");
    return {
      ok: true,
      message: "All dependencies installed",
      required: true,
    };
  } catch {
    return {
      ok: false,
      message: "Some dependencies missing",
      details: "Run: npm install",
      required: true,
    };
  }
}

function checkEmbeddingModel() {
  const modelPath = path.join(__dirname, "..", "models", "bge-small-en-v1.5-Q8_0.gguf");
  
  if (existsSync(modelPath)) {
    return {
      ok: true,
      message: "Found at models/bge-small-en-v1.5-Q8_0.gguf",
      required: true,
    };
  }
  
  return {
    ok: false,
    message: "Not found",
    details: "Download from: https://huggingface.co/BAAI/bge-small-en-v1.5",
    required: true,
  };
}

function checkLanguageModel() {
  const modelPath = path.join(__dirname, "..", "models", "neural-chat-7b-v3-3-Q4_K_M.gguf");
  
  if (existsSync(modelPath)) {
    return {
      ok: true,
      message: "Found at models/neural-chat-7b-v3-3-Q4_K_M.gguf",
      required: false,
    };
  }
  
  return {
    ok: false,
    message: "Not found (optional, for RAG functionality)",
    details: "Download: https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF",
    required: false,
  };
}
