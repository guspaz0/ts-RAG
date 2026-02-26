#!/usr/bin/env bash

echo "🚀 Installing JS Embeddings Dependencies..."

echo ""
echo "📦 Installing npm packages..."
npm install

echo ""
echo "📁 Creating data directory for ChromaDB..."
mkdir -p data

echo ""
echo "✓ Installation complete!"
echo ""
echo "Summary:"
echo "  ✓ npm dependencies installed"
echo "  ✓ data/ directory created for persistent ChromaDB storage"
echo ""
echo "Next steps:"
echo "  1. Download embedding model: bge-small-en-v1.5-Q8_0.gguf"
echo "  2. (Optional) Download language model: neural-chat-7b-v3-3-Q4_K_M.gguf"
echo "  3. Place models in models/ directory"
echo "  4. Run: npm start /path/to/your/document.pdf"
echo ""
