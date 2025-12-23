#!/bin/bash
# Urban Transit Tool - Quick Start Script

set -e  # Exit on error

echo "======================================"
echo " Urban Transit Tool - Docker Setup"
echo "======================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker Desktop from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed!"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please review .env and update passwords if needed"
    echo ""
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data cache output_cache logs
echo "✅ Directories created"
echo ""

# Pull images
echo "📦 Pulling Docker images..."
docker-compose pull
echo "✅ Images pulled"
echo ""

# Build application
echo "🔨 Building application..."
docker-compose build
echo "✅ Application built"
echo ""

# Start services
echo "🚀 Starting services..."
docker-compose up -d
echo ""

# Wait for database to be ready
echo "⏳ Waiting for database to initialize..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "======================================"
    echo " 🎉 Setup Complete!"
    echo "======================================"
    echo ""
    echo "📊 Services:"
    echo "  - Dashboard: http://localhost:8501"
    echo "  - Database:  localhost:5432"
    echo "  - pgAdmin:   http://localhost:5050 (start with: docker-compose --profile tools up -d)"
    echo ""
    echo "📝 Default Credentials:"
    echo "  - Database User: urban_admin"
    echo "  - Database Password: urban_transit_2024 (change in .env)"
    echo ""
    echo "🔍 View Logs:"
    echo "  docker-compose logs -f"
    echo ""
    echo "🛑 Stop Services:"
    echo "  docker-compose stop"
    echo ""
    echo "📖 Full Documentation:"
    echo "  See DOCKER_SETUP.md for complete guide"
    echo ""
else
    echo ""
    echo "❌ Services failed to start!"
    echo "Check logs with: docker-compose logs"
    exit 1
fi
