#!/usr/bin/env python3
"""
Example usage of Yahoo Finance to RDF converter
Demonstrates how to generate ~1M RDF triples from stock data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_processing.yahoo_finance_to_rdf import YahooFinanceToRDF
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def example_synthetic_data():
    """Example: Generate synthetic data and convert to RDF"""
    print("=== Example 1: Synthetic Data Generation ===")
    
    # Initialize converter
    converter = YahooFinanceToRDF()
    
    # Generate synthetic data for multiple stocks over 2 years
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
    df = converter.generate_synthetic_data(symbols, days=730)  # 2 years
    
    print(f"Generated {len(df)} data points")
    print(f"Data shape: {df.shape}")
    print(f"Symbols: {df['Symbol'].unique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert to RDF
    converter.convert_to_rdf(df)
    
    # Save to Turtle format
    converter.save_rdf("synthetic_stock_data.ttl", "turtle")
    
    # Print statistics
    stats = converter.get_statistics()
    print(f"\nRDF Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    return converter

def example_real_data():
    """Example: Fetch real data from Yahoo Finance and convert to RDF"""
    print("\n=== Example 2: Real Yahoo Finance Data ===")
    
    # Initialize converter
    converter = YahooFinanceToRDF()
    
    # Fetch real data for popular stocks
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    df = converter.fetch_real_data(symbols, period="1y")
    
    if not df.empty:
        print(f"Fetched {len(df)} data points")
        print(f"Data shape: {df.shape}")
        print(f"Symbols: {df['Symbol'].unique()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Convert to RDF
        converter.convert_to_rdf(df)
        
        # Save to different formats
        converter.save_rdf("real_stock_data.ttl", "turtle")
        converter.save_rdf("real_stock_data.xml", "xml")
        converter.save_rdf("real_stock_data.json", "json-ld")
        
        # Print statistics
        stats = converter.get_statistics()
        print(f"\nRDF Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")
        
        return converter
    else:
        print("Failed to fetch real data")
        return None

def example_large_dataset():
    """Example: Generate a large dataset to reach ~1M triples"""
    print("\n=== Example 3: Large Dataset Generation (~1M triples) ===")
    
    # Initialize converter
    converter = YahooFinanceToRDF()
    
    # Generate data for many stocks over a long period
    symbols = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM",
        "ORCL", "INTC", "AMD", "QCOM", "AVGO", "TXN", "MU", "ADI", "KLAC", "LRCX",
        "ASML", "TSM", "SMCI", "PLTR", "SNOW", "DDOG", "CRWD", "ZS", "NET", "OKTA"
    ]
    
    # Generate 3 years of data for 30 stocks
    df = converter.generate_synthetic_data(symbols, days=1095)  # 3 years
    
    print(f"Generated {len(df)} data points")
    print(f"Data shape: {df.shape}")
    print(f"Symbols: {len(df['Symbol'].unique())}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert to RDF
    converter.convert_to_rdf(df)
    
    # Save to Turtle format
    converter.save_rdf("large_stock_dataset.ttl", "turtle")
    
    # Print statistics
    stats = converter.get_statistics()
    print(f"\nRDF Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Estimate triples per data point
    triples_per_point = stats['total_triples'] / len(df)
    print(f"\nTriples per data point: {triples_per_point:.1f}")
    
    return converter

def query_example(converter):
    """Example: Query the generated RDF data"""
    print("\n=== Example 4: RDF Query Examples ===")
    
    from rdflib import RDF, Literal, Namespace
    from rdflib.namespace import XSD, DCTERMS
    
    # Define namespaces
    STOCK = Namespace("http://example.org/stock/")
    
    g = converter.g
    
    # Query 1: Find all companies
    print("\n1. All companies:")
    for company in g.subjects(RDF.type, STOCK.Company):
        symbol = g.value(company, STOCK.symbol)
        print(f"  - {symbol}")
    
    # Query 2: Find price observations for a specific date
    print("\n2. Price observations for a specific date:")
    target_date = Literal("2024-01-15", datatype=XSD.date)
    for price_obs in g.subjects(DCTERMS.date, target_date):
        company = g.value(price_obs, STOCK.forCompany)
        close_price = g.value(price_obs, STOCK.closePrice)
        if company and close_price:
            symbol = g.value(company, STOCK.symbol)
            print(f"  - {symbol}: ${close_price}")
    
    # Query 3: Find bullish days
    print("\n3. Bullish days (first 10):")
    bullish_count = 0
    for price_obs in g.subjects(STOCK.priceDirection, Literal("UP")):
        if bullish_count >= 10:
            break
        company = g.value(price_obs, STOCK.forCompany)
        date = g.value(price_obs, DCTERMS.date)
        if company and date:
            symbol = g.value(company, STOCK.symbol)
            print(f"  - {symbol} on {date}")
            bullish_count += 1
    
    # Query 4: Count total price observations
    total_observations = len(list(g.subjects(RDF.type, STOCK.PriceObservation)))
    print(f"\n4. Total price observations: {total_observations:,}")

def main():
    """Run all examples"""
    print("Yahoo Finance to RDF Converter - Examples")
    print("=" * 50)
    
    # Run examples
    converter1 = example_synthetic_data()
    converter2 = example_real_data()
    converter3 = example_large_dataset()
    
    # Query the large dataset
    if converter3:
        query_example(converter3)
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Generated files:")
    print("  - synthetic_stock_data.ttl")
    print("  - real_stock_data.ttl")
    print("  - real_stock_data.xml")
    print("  - real_stock_data.json")
    print("  - large_stock_dataset.ttl")

if __name__ == "__main__":
    main() 