#!/bin/bash

# Setup script for Electric Vehicle Analysis project
set -e

echo "🚗 Setting up Electric Vehicle Analysis Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.11 or later."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_status "Python version: $python_version"
}

# Check if Node.js is installed
check_node() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18 or later."
        exit 1
    fi
    
    node_version=$(node --version)
    print_status "Node.js version: $node_version"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Some features may not work."
        return 1
    fi
    
    docker_version=$(docker --version)
    print_status "Docker version: $docker_version"
    return 0
}

# Create virtual environment
setup_python_env() {
    print_step "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    print_status "Python dependencies installed"
}

# Setup frontend
setup_frontend() {
    print_step "Setting up frontend dependencies..."
    
    cd frontend
    npm install
    print_status "Frontend dependencies installed"
    cd ..
}

# Create necessary directories
create_directories() {
    print_step "Creating necessary directories..."
    
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/external
    mkdir -p models/trained
    mkdir -p models/configs
    mkdir -p logs
    mkdir -p uploads
    mkdir -p mlruns
    
    print_status "Directories created"
}

# Setup environment file
setup_env_file() {
    print_step "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_status "Environment file created from template"
        print_warning "Please edit .env file with your configuration"
    else
        print_status "Environment file already exists"
    fi
}

# Download sample data
download_sample_data() {
    print_step "Downloading sample dataset..."
    
    if [ ! -f "data/raw/electric_vehicles.csv" ]; then
        # Download Washington State EV dataset
        curl -o data/raw/electric_vehicles.csv \
            "https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD"
        print_status "Sample dataset downloaded"
    else
        print_status "Sample dataset already exists"
    fi
}

# Setup pre-commit hooks
setup_pre_commit() {
    print_step "Setting up pre-commit hooks..."
    
    source venv/bin/activate
    pip install pre-commit
    pre-commit install
    print_status "Pre-commit hooks installed"
}

# Run initial data processing
process_initial_data() {
    print_step "Processing initial dataset..."
    
    source venv/bin/activate
    python scripts/process_data.py
    print_status "Initial data processing completed"
}

# Setup database (if Docker is available)
setup_database() {
    if check_docker; then
        print_step "Setting up database with Docker..."
        docker-compose up -d db redis
        sleep 10
        print_status "Database containers started"
        
        # Wait for database to be ready
        echo "Waiting for database to be ready..."
        until docker-compose exec -T db pg_isready -U postgres; do
            sleep 2
        done
        print_status "Database is ready"
    else
        print_warning "Docker not available. Please setup PostgreSQL manually."
    fi
}

# Main setup function
main() {
    print_status "Starting setup process..."
    
    # Check prerequisites
    check_python
    check_node
    
    # Setup project
    create_directories
    setup_env_file
    setup_python_env
    setup_frontend
    setup_pre_commit
    download_sample_data
    setup_database
    process_initial_data
    
    print_status "Setup completed successfully! 🎉"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Start the application: docker-compose up"
    echo "3. Access the frontend at: http://localhost:3000"
    echo "4. Access the API docs at: http://localhost:8000/docs"
    echo "5. Access MLflow UI at: http://localhost:5000"
}

# Run main function
main