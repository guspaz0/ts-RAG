#!/bin/bash

# Quick setup guide for pgvector backend

set -e

echo "🚀 JS Embeddings - pgvector Quick Setup"
echo "========================================"
echo ""

# Step 1: Check Python installation
echo "Step 1: Checking prerequisites..."
if command -v psql &> /dev/null; then
    PG_VERSION=$(psql --version | awk '{print $3}' | cut -d. -f1)
    echo "✓ PostgreSQL ${PG_VERSION} found"
else
    echo "✗ PostgreSQL not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install postgresql@15 pgvector
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install postgresql postgresql-contrib
    fi
fi

# Step 2: Start PostgreSQL
echo ""
echo "Step 2: Starting PostgreSQL..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew services start postgresql@15 2>/dev/null || brew services restart postgresql@15
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo systemctl start postgresql
fi
echo "✓ PostgreSQL running"

# Step 3: Create database
echo ""
echo "Step 3: Setting up database..."
bash ./setup-pgvector.sh

# Step 4: Install dependencies
echo ""
echo "Step 4: Installing npm dependencies..."
npm install

# Step 5: Compile TypeScript
echo ""
echo "Step 5: Compiling TypeScript..."
npm run build

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download embedding models:"
echo "   mkdir -p models"
echo "   # Download bge-small-en-v1.5-Q8_0.gguf from HuggingFace"
echo ""
echo "2. Run the application:"
echo "   npm start path/to/document.pdf"
echo ""
echo "3. With a question:"
echo "   npm start path/to/document.pdf \"What is this about?\""
echo ""
