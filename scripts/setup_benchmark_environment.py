#!/usr/bin/env python3
"""
Setup script for benchmark environment
"""

import subprocess
import sys
import os
import asyncio
import httpx
import time
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing benchmark dependencies...")
    
    packages = [
        "httpx",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "asyncio"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def check_system_health():
    """Check if both systems are running"""
    print("\n🏥 Checking system health...")
    
    systems = {
        "Neurosymbolic AI": "http://localhost:8000/health",
        "Jena Fuseki": "http://localhost:3030/hft_jena/query"
    }
    
    healthy_systems = []
    
    for name, url in systems.items():
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                print(f"✅ {name}: Healthy")
                healthy_systems.append(name)
            else:
                print(f"❌ {name}: Unhealthy (HTTP {response.status_code})")
        except Exception as e:
            print(f"❌ {name}: Unreachable ({e})")
    
    return healthy_systems

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "benchmark_results",
        "performance_charts",
        "load_test_charts",
        "benchmark_charts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")

def create_sample_data():
    """Create sample data for testing"""
    print("\n📊 Creating sample data...")
    
    # Create sample stock data for Jena
    sample_data = """
@prefix hft: <http://hft.example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://hft.example.org/company/AAPL> rdf:type hft:Company ;
    hft:symbol "AAPL" ;
    hft:name "Apple Inc." ;
    hft:price 150.25 ;
    hft:volume 1000000 ;
    hft:timestamp "2023-01-01T00:00:00Z"^^xsd:dateTime ;
    hft:rsi 65.5 ;
    hft:ma_short 148.50 ;
    hft:ma_long 145.75 .

<http://hft.example.org/company/MSFT> rdf:type hft:Company ;
    hft:symbol "MSFT" ;
    hft:name "Microsoft Corporation" ;
    hft:price 300.75 ;
    hft:volume 800000 ;
    hft:timestamp "2023-01-01T00:00:00Z"^^xsd:dateTime ;
    hft:rsi 70.2 ;
    hft:ma_short 298.50 ;
    hft:ma_long 295.25 .

<http://hft.example.org/company/GOOGL> rdf:type hft:Company ;
    hft:symbol "GOOGL" ;
    hft:name "Alphabet Inc." ;
    hft:price 2500.50 ;
    hft:volume 500000 ;
    hft:timestamp "2023-01-01T00:00:00Z"^^xsd:dateTime ;
    hft:rsi 55.8 ;
    hft:ma_short 2480.25 ;
    hft:ma_long 2450.75 .
"""
    
    with open("sample_stock_data.ttl", "w") as f:
        f.write(sample_data)
    
    print("✅ Created sample_stock_data.ttl")

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\n🧪 Running quick test...")
    
    try:
        # Test Neurosymbolic AI
        response = httpx.post(
            "http://localhost:8000/api/v1/trading/signal",
            json={
                "symbol": "AAPL",
                "timeframe": "1m",
                "strategy": "neurosymbolic"
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            print("✅ Neurosymbolic AI test passed")
        else:
            print(f"❌ Neurosymbolic AI test failed: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Neurosymbolic AI test failed: {e}")
    
    try:
        # Test Jena Fuseki
        sparql_query = """
        PREFIX hft: <http://hft.example.org/>
        SELECT ?symbol ?price WHERE {
            ?company hft:symbol ?symbol .
            ?company hft:price ?price .
        } LIMIT 5
        """
        
        response = httpx.post(
            "http://localhost:3030/hft_jena/query",
            data={"query": sparql_query, "output": "json"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10.0
        )
        
        if response.status_code == 200:
            print("✅ Jena Fuseki test passed")
        else:
            print(f"❌ Jena Fuseki test failed: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Jena Fuseki test failed: {e}")

def main():
    """Main setup function"""
    print("🚀 Setting up benchmark environment...")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Create sample data
    create_sample_data()
    
    # Check system health
    healthy_systems = check_system_health()
    
    if len(healthy_systems) < 2:
        print("\n⚠️  WARNING: Not all systems are healthy!")
        print("Please ensure both Neurosymbolic AI and Jena Fuseki are running.")
        print("You can start them with:")
        print("  docker-compose up -d")
        print("  python -m uvicorn main:app --host 0.0.0.0 --port 8000")
        return False
    
    # Run quick test
    run_quick_test()
    
    print("\n" + "=" * 50)
    print("✅ Benchmark environment setup complete!")
    print("=" * 50)
    print("\nTo run benchmarks:")
    print("  python scripts/real_benchmark_validator.py")
    print("  python scripts/realtime_performance_monitor.py")
    print("  python scripts/load_testing_framework.py")
    print("  python scripts/master_benchmark_runner.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
