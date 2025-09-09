#!/usr/bin/env python3
"""
Yahoo Finance CSV to RDF Converter
Converts stock price data to RDF triples using RDFLib
Generates approximately 1M triples for HFT neurosymbolic analysis
"""

import pandas as pd
import rdflib
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD, DCTERMS, FOAF
import yfinance as yf
import datetime
import random
import os
import argparse
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define RDF namespaces
STOCK = Namespace("http://example.org/stock/")
PRICE = Namespace("http://example.org/price/")
COMPANY = Namespace("http://example.org/company/")
MARKET = Namespace("http://example.org/market/")
INDICATOR = Namespace("http://example.org/indicator/")

class YahooFinanceToRDF:
    """Convert Yahoo Finance data to RDF triples"""
    
    def __init__(self):
        self.g = Graph()
        self.setup_namespaces()
        self.triple_count = 0
        
    def setup_namespaces(self):
        """Bind all namespaces to the graph"""
        self.g.bind("stock", STOCK)
        self.g.bind("price", PRICE)
        self.g.bind("company", COMPANY)
        self.g.bind("market", MARKET)
        self.g.bind("indicator", INDICATOR)
        self.g.bind("rdf", RDF)
        self.g.bind("rdfs", RDFS)
        self.g.bind("xsd", XSD)
        self.g.bind("dcterms", DCTERMS)
        self.g.bind("foaf", FOAF)
        
    def generate_synthetic_data(self, symbols: List[str], days: int = 365) -> pd.DataFrame:
        """Generate synthetic stock data for testing"""
        logger.info(f"Generating synthetic data for {len(symbols)} symbols over {days} days")
        
        data = []
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        for symbol in symbols:
            # Generate realistic price movements
            base_price = random.uniform(50, 500)
            current_price = base_price
            
            current_date = start_date
            while current_date <= end_date:
                # Simulate daily price movement
                change_pct = random.uniform(-0.05, 0.05)  # ±5% daily change
                current_price *= (1 + change_pct)
                
                # Generate volume
                volume = random.randint(1000000, 10000000)
                
                # Generate OHLC data
                high = current_price * random.uniform(1.0, 1.03)
                low = current_price * random.uniform(0.97, 1.0)
                open_price = random.uniform(low, high)
                close_price = current_price
                
                data.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Symbol': symbol,
                    'Open': round(open_price, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(close_price, 2),
                    'Adj_Close': round(close_price * random.uniform(0.98, 1.02), 2),
                    'Volume': volume
                })
                
                current_date += datetime.timedelta(days=1)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} data points")
        return df
    
    def fetch_real_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Fetch real data from Yahoo Finance"""
        logger.info(f"Fetching real data for {len(symbols)} symbols")
        
        all_data = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    hist.reset_index(inplace=True)
                    hist['Symbol'] = symbol
                    # Handle different column structures from Yahoo Finance
                    if 'Adj Close' in hist.columns:
                        hist = hist.rename(columns={'Adj Close': 'Adj_Close'})
                    elif 'Adj_Close' not in hist.columns:
                        hist['Adj_Close'] = hist['Close']  # Use Close as Adj_Close if not available
                    
                    # Ensure we have the expected columns
                    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'Symbol']
                    for col in expected_columns:
                        if col not in hist.columns:
                            if col == 'Date':
                                hist['Date'] = hist.index
                            else:
                                hist[col] = 0  # Default value for missing columns
                    
                    hist = hist[expected_columns]
                    all_data.append(hist)
                    logger.info(f"Fetched data for {symbol}: {len(hist)} records")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total records fetched: {len(df)}")
            return df
        else:
            logger.error("No data fetched from Yahoo Finance")
            return pd.DataFrame()
    
    def create_company_triples(self, symbol: str) -> None:
        """Create triples for company information"""
        company_uri = COMPANY[symbol]
        
        # Company type
        self.g.add((company_uri, RDF.type, STOCK.Company))
        self.g.add((company_uri, RDFS.label, Literal(f"{symbol} Corporation", lang="en")))
        self.g.add((company_uri, STOCK.symbol, Literal(symbol)))
        self.g.add((company_uri, STOCK.exchange, Literal("NASDAQ")))
        self.g.add((company_uri, STOCK.sector, Literal("Technology")))
        
        self.triple_count += 5
    
    def create_price_triples(self, row: pd.Series) -> None:
        """Create triples for price data"""
        symbol = row['Symbol']
        date = row['Date']
        
        # Convert date to string format
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # Create URIs
        price_uri = PRICE[f"{symbol}_{date_str.replace('-', '_')}"]
        company_uri = COMPANY[symbol]
        date_uri = URIRef(f"http://example.org/date/{date_str}")
        
        # Price observation
        self.g.add((price_uri, RDF.type, STOCK.PriceObservation))
        self.g.add((price_uri, STOCK.forCompany, company_uri))
        self.g.add((price_uri, STOCK.observationDate, date_uri))
        self.g.add((price_uri, DCTERMS.date, Literal(date_str, datatype=XSD.date)))
        
        # Price values
        self.g.add((price_uri, STOCK.openPrice, Literal(row['Open'], datatype=XSD.decimal)))
        self.g.add((price_uri, STOCK.highPrice, Literal(row['High'], datatype=XSD.decimal)))
        self.g.add((price_uri, STOCK.lowPrice, Literal(row['Low'], datatype=XSD.decimal)))
        self.g.add((price_uri, STOCK.closePrice, Literal(row['Close'], datatype=XSD.decimal)))
        self.g.add((price_uri, STOCK.adjustedClosePrice, Literal(row['Adj_Close'], datatype=XSD.decimal)))
        self.g.add((price_uri, STOCK.volume, Literal(row['Volume'], datatype=XSD.integer)))
        
        # Currency information
        self.g.add((price_uri, STOCK.currency, Literal("USD")))
        
        # Price change calculations
        if row['Close'] > row['Open']:
            self.g.add((price_uri, STOCK.priceDirection, Literal("UP")))
        elif row['Close'] < row['Open']:
            self.g.add((price_uri, STOCK.priceDirection, Literal("DOWN")))
        else:
            self.g.add((price_uri, STOCK.priceDirection, Literal("FLAT")))
        
        # Price change amount
        price_change = row['Close'] - row['Open']
        self.g.add((price_uri, STOCK.priceChange, Literal(price_change, datatype=XSD.decimal)))
        
        # Price change percentage
        if row['Open'] != 0:
            price_change_pct = (price_change / row['Open']) * 100
            self.g.add((price_uri, STOCK.priceChangePercent, Literal(price_change_pct, datatype=XSD.decimal)))
        
        # High-Low range
        high_low_range = row['High'] - row['Low']
        self.g.add((price_uri, STOCK.highLowRange, Literal(high_low_range, datatype=XSD.decimal)))
        
        self.triple_count += 15
    
    def create_market_analysis_triples(self, df: pd.DataFrame) -> None:
        """Create triples for market analysis and indicators"""
        logger.info("Creating market analysis triples")
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].sort_values('Date').reset_index(drop=True)
            
            if len(symbol_data) < 20:  # Need at least 20 days for moving averages
                continue
                
            # Calculate moving averages
            ma_20 = symbol_data['Close'].rolling(window=20).mean()
            ma_50 = symbol_data['Close'].rolling(window=50).mean()
            
            for idx, row in symbol_data.iterrows():
                if pd.isna(ma_20.iloc[idx]) or pd.isna(ma_50.iloc[idx]):
                    continue
                    
                date = row['Date']
                # Convert date to string format
                if hasattr(date, 'strftime'):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                price_uri = PRICE[f"{symbol}_{date_str.replace('-', '_')}"]
                
                # Moving average triples
                self.g.add((price_uri, INDICATOR.movingAverage20, Literal(ma_20.iloc[idx], datatype=XSD.decimal)))
                self.g.add((price_uri, INDICATOR.movingAverage50, Literal(ma_50.iloc[idx], datatype=XSD.decimal)))
                
                # Moving average crossovers
                if ma_20.iloc[idx] > ma_50.iloc[idx]:
                    self.g.add((price_uri, INDICATOR.maCrossover, Literal("BULLISH")))
                else:
                    self.g.add((price_uri, INDICATOR.maCrossover, Literal("BEARISH")))
                
                # Price vs moving average
                if row['Close'] > ma_20.iloc[idx]:
                    self.g.add((price_uri, INDICATOR.priceVsMA20, Literal("ABOVE")))
                else:
                    self.g.add((price_uri, INDICATOR.priceVsMA20, Literal("BELOW")))
                
                self.triple_count += 5
    
    def create_volume_analysis_triples(self, df: pd.DataFrame) -> None:
        """Create triples for volume analysis"""
        logger.info("Creating volume analysis triples")
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol]
            avg_volume = symbol_data['Volume'].mean()
            
            for _, row in symbol_data.iterrows():
                date = row['Date']
                # Convert date to string format
                if hasattr(date, 'strftime'):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                price_uri = PRICE[f"{symbol}_{date_str.replace('-', '_')}"]
                
                # Volume analysis
                if row['Volume'] > avg_volume * 1.5:
                    self.g.add((price_uri, INDICATOR.volumeSignal, Literal("HIGH")))
                elif row['Volume'] < avg_volume * 0.5:
                    self.g.add((price_uri, INDICATOR.volumeSignal, Literal("LOW")))
                else:
                    self.g.add((price_uri, INDICATOR.volumeSignal, Literal("NORMAL")))
                
                # Volume ratio
                volume_ratio = row['Volume'] / avg_volume
                self.g.add((price_uri, INDICATOR.volumeRatio, Literal(volume_ratio, datatype=XSD.decimal)))
                
                self.triple_count += 2
    
    def create_temporal_triples(self, df: pd.DataFrame) -> None:
        """Create triples for temporal relationships"""
        logger.info("Creating temporal relationship triples")
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].sort_values('Date')
            
            for i in range(len(symbol_data) - 1):
                current_date = symbol_data.iloc[i]['Date']
                next_date = symbol_data.iloc[i + 1]['Date']
                
                # Convert dates to string format
                if hasattr(current_date, 'strftime'):
                    current_date_str = current_date.strftime('%Y-%m-%d')
                else:
                    current_date_str = str(current_date)
                
                if hasattr(next_date, 'strftime'):
                    next_date_str = next_date.strftime('%Y-%m-%d')
                else:
                    next_date_str = str(next_date)
                
                current_uri = PRICE[f"{symbol}_{current_date_str.replace('-', '_')}"]
                next_uri = PRICE[f"{symbol}_{next_date_str.replace('-', '_')}"]
                
                # Temporal relationships
                self.g.add((current_uri, STOCK.precedes, next_uri))
                self.g.add((next_uri, STOCK.follows, current_uri))
                
                # Day of week
                if hasattr(current_date, 'strftime'):
                    day_of_week = current_date.strftime('%A')
                    month = current_date.month
                    year = current_date.year
                else:
                    # Fallback for string dates
                    date_obj = datetime.datetime.strptime(current_date_str, '%Y-%m-%d')
                    day_of_week = date_obj.strftime('%A')
                    month = date_obj.month
                    year = date_obj.year
                
                self.g.add((current_uri, STOCK.dayOfWeek, Literal(day_of_week)))
                
                # Month and year
                self.g.add((current_uri, STOCK.month, Literal(month, datatype=XSD.integer)))
                self.g.add((current_uri, STOCK.year, Literal(year, datatype=XSD.integer)))
                
                self.triple_count += 5
    
    def convert_to_rdf(self, df: pd.DataFrame) -> None:
        """Convert DataFrame to RDF triples"""
        logger.info("Starting RDF conversion")
        
        # Create company triples
        for symbol in df['Symbol'].unique():
            self.create_company_triples(symbol)
        
        # Create price triples
        for _, row in df.iterrows():
            self.create_price_triples(row)
        
        # Create additional analysis triples
        self.create_market_analysis_triples(df)
        self.create_volume_analysis_triples(df)
        self.create_temporal_triples(df)
        
        logger.info(f"Conversion complete. Total triples: {self.triple_count}")
    
    def save_rdf(self, filename: str, format: str = "turtle") -> None:
        """Save RDF graph to file"""
        logger.info(f"Saving RDF to {filename} in {format} format")
        self.g.serialize(destination=filename, format=format)
        logger.info(f"RDF saved successfully")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated RDF"""
        stats = {
            "total_triples": len(self.g),
            "unique_subjects": len(set(s for s, p, o in self.g)),
            "unique_predicates": len(set(p for s, p, o in self.g)),
            "unique_objects": len(set(o for s, p, o in self.g)),
            "namespaces": len(list(self.g.namespaces()))
        }
        return stats

def main():
    parser = argparse.ArgumentParser(description="Convert Yahoo Finance data to RDF triples")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                       help="Stock symbols to process")
    parser.add_argument("--period", default="1y", help="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead of fetching from Yahoo Finance")
    parser.add_argument("--output", default="stock_data.ttl", help="Output RDF file")
    parser.add_argument("--format", default="turtle", choices=["turtle", "xml", "json-ld", "nt"], help="RDF output format")
    parser.add_argument("--days", type=int, default=365, help="Number of days for synthetic data")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = YahooFinanceToRDF()
    
    # Get data
    if args.synthetic:
        logger.info("Generating synthetic data")
        df = converter.generate_synthetic_data(args.symbols, args.days)
    else:
        logger.info("Fetching data from Yahoo Finance")
        df = converter.fetch_real_data(args.symbols, args.period)
    
    if df.empty:
        logger.error("No data available for conversion")
        return
    
    # Convert to RDF
    converter.convert_to_rdf(df)
    
    # Save RDF
    converter.save_rdf(args.output, args.format)
    
    # Print statistics
    stats = converter.get_statistics()
    logger.info("RDF Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"Successfully generated ~{stats['total_triples']:,} RDF triples")

if __name__ == "__main__":
    main() 