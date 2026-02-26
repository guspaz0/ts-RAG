#!/bin/bash

# PostgreSQL & pgvector setup script for js-embeddings

set -e

echo "🚀 Setting up PostgreSQL & pgvector for JS Embeddings"
echo "======================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${YELLOW}⚠️  PostgreSQL is not installed${NC}"
    echo ""
    echo "Please install PostgreSQL:"
    echo "  macOS:   brew install postgresql@15"
    echo "  Linux:   sudo apt-get install postgresql postgresql-contrib"
    echo "  Windows: https://www.postgresql.org/download/windows/"
    exit 1
fi

# Detect PostgreSQL version
PG_VERSION=$(psql --version | awk '{print $3}' | cut -d. -f1)
echo -e "${GREEN}✓ PostgreSQL ${PG_VERSION} found${NC}"

# Check if PostgreSQL is running
if ! pg_isready > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  PostgreSQL is not running${NC}"
    echo ""
    echo "Starting PostgreSQL..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start postgresql@$PG_VERSION
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start postgresql
    fi
    sleep 2
fi

echo -e "${GREEN}✓ PostgreSQL is running${NC}"

# Check if pgvector extension is available
echo ""
echo "Checking pgvector extension..."

# Try to create pgvector extension
PGPASSWORD=${PG_PASSWORD:-postgres} psql -U ${PG_USER:-postgres} -h ${PG_HOST:-localhost} \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" postgres 2>/dev/null || {
    echo -e "${YELLOW}⚠️  pgvector extension not found${NC}"
    echo ""
    echo "Installing pgvector..."
    echo "  macOS:   brew install pgvector"
    echo "  Linux:   git clone https://github.com/pgvector/pgvector.git"
    echo "           cd pgvector && make && sudo make install"
    echo ""
    exit 1
}

echo -e "${GREEN}✓ pgvector extension available${NC}"

# Create database
echo ""
echo "Creating embeddings database..."

PGPASSWORD=${PG_PASSWORD:-postgres} psql -U ${PG_USER:-postgres} -h ${PG_HOST:-localhost} \
    -tc "SELECT 1 FROM pg_database WHERE datname = 'embeddings'" | grep -q 1 || {
    PGPASSWORD=${PG_PASSWORD:-postgres} psql -U ${PG_USER:-postgres} -h ${PG_HOST:-localhost} \
        -c "CREATE DATABASE embeddings;" postgres
    echo -e "${GREEN}✓ Database 'embeddings' created${NC}"
} && echo -e "${GREEN}✓ Database 'embeddings' already exists${NC}"

# Create pgvector extension in embeddings database
echo ""
echo "Setting up pgvector extension..."

PGPASSWORD=${PG_PASSWORD:-postgres} psql -U ${PG_USER:-postgres} -h ${PG_HOST:-localhost} \
    -d embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;" > /dev/null

echo -e "${GREEN}✓ pgvector extension enabled${NC}"

# Summary
echo ""
echo -e "${GREEN}✅ PostgreSQL setup complete!${NC}"
echo ""
echo "Database Connection Info:"
echo "  Host:     ${PG_HOST:-localhost}"
echo "  Port:     ${PG_PORT:-5432}"
echo "  User:     ${PG_USER:-postgres}"
echo "  Database: embeddings"
echo ""
echo "Next steps:"
echo "  1. Set environment variables (optional):"
echo "     export PG_HOST=${PG_HOST:-localhost}"
echo "     export PG_PORT=${PG_PORT:-5432}"
echo "     export PG_USER=${PG_USER:-postgres}"
echo "     export PG_PASSWORD=${PG_PASSWORD:-postgres}"
echo "     export PG_DATABASE=embeddings"
echo ""
echo "  2. Or create a .env file in project root"
echo ""
echo "  3. Start the application:"
echo "     npm start document.pdf"
echo ""
echo -e "${BLUE}For more info, see PGVECTOR_SETUP.md${NC}"
