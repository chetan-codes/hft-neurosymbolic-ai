#!/bin/bash

# HFT Neurosymbolic AI System - Complete Setup and Startup Script
# This script sets up and starts the entire HFT neurosymbolic AI system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="hft_neurosymbolic"
DOCKER_COMPOSE_FILE="docker-compose.yml"
LOG_FILE="setup.log"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check available memory (macOS compatible)
    if command_exists sysctl; then
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "System has less than 8GB RAM. Performance may be degraded."
        fi
    else
        print_warning "Could not check memory. Please ensure you have at least 8GB RAM."
    fi
    
    # Check available disk space (macOS compatible)
    DISK_GB=$(df -g . | awk 'NR==2 {print $4}')
    if [ "$DISK_GB" -lt 10 ]; then
        print_error "Insufficient disk space. Need at least 10GB free space."
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data models logs config benchmarks
    mkdir -p config/grafana/dashboards config/grafana/datasources
    
    print_success "Directories created"
}

# Function to build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build the main application image
    docker-compose build hft_app
    
    print_success "Docker images built successfully"
}

# Function to start services
start_services() {
    print_status "Starting HFT Neurosymbolic AI System services..."
    
    # Start all services
    docker-compose up -d
    
    print_success "Services started successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for Dgraph
    print_status "Waiting for Dgraph..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8080/health >/dev/null 2>&1; then
            print_success "Dgraph is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Dgraph may not be fully ready"
    fi
    
    # Wait for Neo4j
    print_status "Waiting for Neo4j..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:7474 >/dev/null 2>&1; then
            print_success "Neo4j is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Neo4j may not be fully ready"
    fi
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if docker-compose exec redis redis-cli ping >/dev/null 2>&1; then
            print_success "Redis is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Redis may not be fully ready"
    fi
    
    # Wait for Jena
    print_status "Waiting for Jena Fuseki..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:3030 >/dev/null 2>&1; then
            print_success "Jena Fuseki is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Jena Fuseki may not be fully ready"
    fi
    
    # Wait for main application
    print_status "Waiting for HFT application..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            print_success "HFT application is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "HFT application may not be fully ready"
    fi
}

# Function to run initial data ingestion
run_initial_ingestion() {
    print_status "Running initial data ingestion..."
    
    # Generate synthetic data and convert to RDF
    docker-compose exec hft_app python yahoo_finance_to_rdf.py \
        --synthetic \
        --symbols AAPL GOOGL MSFT AMZN TSLA \
        --days 30 \
        --output initial_data.ttl
    
    print_success "Initial data ingestion completed"
}

# Function to run system verification
run_verification() {
    print_status "Running system verification..."
    
    # Run the verification script
    docker-compose exec hft_app python setup_verification_docker.py
    
    print_success "System verification completed"
}

# Function to display service status
show_status() {
    print_status "HFT Neurosymbolic AI System Status:"
    echo
    
    # Show Docker containers status
    docker-compose ps
    
    echo
    print_status "Service URLs:"
    echo "  HFT Application:     http://localhost:8000"
    echo "  Streamlit Dashboard: http://localhost:8501"
    echo "  Dgraph:             http://localhost:8080"
    echo "  Neo4j Browser:      http://localhost:7474"
    echo "  Jena Fuseki:        http://localhost:3030"
    echo "  Prometheus:         http://localhost:9090"
    echo "  Grafana:            http://localhost:3000"
    echo "  Redis CLI:          docker-compose exec redis redis-cli"
    
    echo
    print_status "Default Credentials:"
    echo "  Neo4j: neo4j / hft_password_2025"
    echo "  Grafana: admin / hft_admin_2025"
    
    echo
    print_status "Useful Commands:"
    echo "  View logs:          docker-compose logs -f"
    echo "  Stop services:      docker-compose down"
    echo "  Restart services:   docker-compose restart"
    echo "  Access shell:       docker-compose exec hft_app bash"
}

# Function to display help
show_help() {
    echo "HFT Neurosymbolic AI System Setup Script"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  setup       Complete setup and start (default)"
    echo "  start       Start services only"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show service logs"
    echo "  clean       Clean up all data and containers"
    echo "  help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0 setup    # Complete setup and start"
    echo "  $0 start    # Start services only"
    echo "  $0 status   # Show current status"
}

# Function to stop services
stop_services() {
    print_status "Stopping HFT Neurosymbolic AI System..."
    docker-compose down
    print_success "Services stopped"
}

# Function to restart services
restart_services() {
    print_status "Restarting HFT Neurosymbolic AI System..."
    docker-compose restart
    print_success "Services restarted"
}

# Function to show logs
show_logs() {
    print_status "Showing service logs..."
    docker-compose logs -f
}

# Function to clean up
clean_up() {
    print_warning "This will remove all data and containers. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v
        docker system prune -f
        rm -rf data/* models/* logs/*
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main execution
main() {
    case "${1:-setup}" in
        "setup")
            print_status "Starting HFT Neurosymbolic AI System setup..."
            check_requirements
            create_directories
            build_images
            start_services
            wait_for_services
            run_initial_ingestion
            run_verification
            show_status
            print_success "HFT Neurosymbolic AI System setup completed!"
            ;;
        "start")
            start_services
            wait_for_services
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            wait_for_services
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "clean")
            clean_up
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 