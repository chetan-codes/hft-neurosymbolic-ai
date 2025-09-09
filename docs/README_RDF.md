# Yahoo Finance to RDF Converter

A Python script to ingest Yahoo Finance CSV data (or synthetic data) and convert it to approximately 1M RDF triples using RDFLib for HFT neurosymbolic analysis.

## Features

- **Real Data Fetching**: Fetch live stock data from Yahoo Finance API
- **Synthetic Data Generation**: Generate realistic synthetic stock data for testing
- **RDF Conversion**: Convert stock data to RDF triples with rich semantic relationships
- **Multiple Output Formats**: Support for Turtle, XML, JSON-LD, and N-Triples formats
- **Comprehensive Analysis**: Generate triples for price data, technical indicators, volume analysis, and temporal relationships
- **Scalable**: Generate datasets with ~1M triples for large-scale analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_rdf.txt
```

2. Or install manually:
```bash
pip install rdflib pandas yfinance numpy
```

## Usage

### Basic Usage

```bash
# Generate synthetic data and convert to RDF
python yahoo_finance_to_rdf.py --synthetic --symbols AAPL GOOGL MSFT --days 365

# Fetch real data from Yahoo Finance
python yahoo_finance_to_rdf.py --symbols AAPL GOOGL MSFT --period 1y

# Generate large dataset (~1M triples)
python yahoo_finance_to_rdf.py --synthetic --symbols AAPL GOOGL MSFT AMZN TSLA META NVDA NFLX --days 1095
```

### Command Line Options

- `--symbols`: Stock symbols to process (default: AAPL, GOOGL, MSFT, AMZN, TSLA)
- `--period`: Data period for real data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- `--synthetic`: Generate synthetic data instead of fetching from Yahoo Finance
- `--output`: Output RDF file name (default: stock_data.ttl)
- `--format`: RDF output format (turtle, xml, json-ld, nt)
- `--days`: Number of days for synthetic data (default: 365)

### Example Usage

```python
from utils.data_processing.yahoo_finance_to_rdf import YahooFinanceToRDF

# Initialize converter
converter = YahooFinanceToRDF()

# Generate synthetic data
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
df = converter.generate_synthetic_data(symbols, days=365)

# Convert to RDF
converter.convert_to_rdf(df)

# Save to different formats
converter.save_rdf("stock_data.ttl", "turtle")
converter.save_rdf("stock_data.xml", "xml")

# Get statistics
stats = converter.get_statistics()
print(f"Total triples: {stats['total_triples']:,}")
```

## RDF Schema

The script generates RDF triples using the following namespaces:

- `stock:` - Stock-related concepts and properties
- `price:` - Price observations and data
- `company:` - Company information
- `market:` - Market-related concepts
- `indicator:` - Technical indicators and analysis

### Example Triples

```
# Company information
<http://example.org/company/AAPL> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/stock/Company> .
<http://example.org/company/AAPL> <http://example.org/stock/symbol> "AAPL" .

# Price observation
<http://example.org/price/AAPL_2024_01_15> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/stock/PriceObservation> .
<http://example.org/price/AAPL_2024_01_15> <http://example.org/stock/closePrice> "150.25"^^<http://www.w3.org/2001/XMLSchema#decimal> .
<http://example.org/price/AAPL_2024_01_15> <http://example.org/stock/priceDirection> "UP" .

# Technical indicators
<http://example.org/price/AAPL_2024_01_15> <http://example.org/indicator/movingAverage20> "148.50"^^<http://www.w3.org/2001/XMLSchema#decimal> .
<http://example.org/price/AAPL_2024_01_15> <http://example.org/indicator/maCrossover> "BULLISH" .
```

## Generated Triples Breakdown

For each stock data point, the script generates approximately 27 triples:

1. **Company Information** (5 triples per company)
   - Company type, label, symbol, exchange, sector

2. **Price Data** (15 triples per observation)
   - Open, high, low, close, adjusted close prices
   - Volume, currency, price direction
   - Price change amount and percentage
   - High-low range

3. **Technical Indicators** (5 triples per observation)
   - 20-day and 50-day moving averages
   - Moving average crossovers
   - Price vs moving average relationships

4. **Volume Analysis** (2 triples per observation)
   - Volume signals (high, normal, low)
   - Volume ratios

5. **Temporal Relationships** (5 triples per observation)
   - Precedes/follows relationships
   - Day of week, month, year

## Scaling to 1M Triples

To generate approximately 1M triples:

```bash
# Option 1: Many stocks over long period
python yahoo_finance_to_rdf.py --synthetic --symbols AAPL GOOGL MSFT AMZN TSLA META NVDA NFLX ADBE CRM ORCL INTC AMD QCOM AVGO TXN MU ADI KLAC LRCX ASML TSM SMCI PLTR SNOW DDOG CRWD ZS NET OKTA --days 1095

# Option 2: Fewer stocks over very long period
python yahoo_finance_to_rdf.py --synthetic --symbols AAPL GOOGL MSFT --days 3650
```

**Calculation**: 
- 30 stocks × 1095 days × 27 triples per observation = ~887,000 triples
- 3 stocks × 3650 days × 27 triples per observation = ~295,000 triples

## Example Scripts

Run the example script to see all features in action:

```bash
python example_usage.py
```

This will:
1. Generate synthetic data for 8 stocks over 2 years
2. Fetch real data from Yahoo Finance for 5 stocks
3. Generate a large dataset with 30 stocks over 3 years (~1M triples)
4. Demonstrate RDF querying capabilities

## Output Files

The script generates:
- `stock_data.ttl` - Turtle format (default)
- `stock_data.xml` - RDF/XML format
- `stock_data.json` - JSON-LD format
- `stock_data.nt` - N-Triples format

## RDF Querying Examples

```python
from rdflib import Graph, Literal

# Load the generated RDF
g = Graph()
g.parse("stock_data.ttl", format="turtle")

# Find all companies
for company in g.subjects(RDF.type, STOCK.Company):
    symbol = g.value(company, STOCK.symbol)
    print(f"Company: {symbol}")

# Find bullish days
for price_obs in g.subjects(STOCK.priceDirection, Literal("UP")):
    company = g.value(price_obs, STOCK.forCompany)
    date = g.value(price_obs, DCTERMS.date)
    print(f"Bullish: {company} on {date}")

# Find high volume days
for price_obs in g.subjects(INDICATOR.volumeSignal, Literal("HIGH")):
    company = g.value(price_obs, STOCK.forCompany)
    volume = g.value(price_obs, STOCK.volume)
    print(f"High volume: {company} - {volume}")
```

## Integration with HFT Neurosymbolic System

This RDF converter is designed to work with the HFT neurosymbolic system:

1. **Knowledge Graph**: Provides structured financial data for reasoning
2. **Symbolic AI**: Enables logical queries and rule-based analysis
3. **Neural Integration**: Can be used with graph neural networks
4. **Temporal Analysis**: Supports time-series reasoning and pattern recognition

## Performance Considerations

- **Memory Usage**: Large datasets may require significant RAM
- **Processing Time**: 1M triples generation takes ~2-5 minutes
- **File Size**: Turtle format is most compact (~50-100MB for 1M triples)
- **Query Performance**: Consider using a triplestore for large datasets

## Troubleshooting

1. **Yahoo Finance API Issues**: Use `--synthetic` flag for testing
2. **Memory Errors**: Reduce number of symbols or days
3. **File Size**: Use Turtle format for smaller files
4. **Network Issues**: Check internet connection for real data fetching

## License

This script is part of the HFT Neurosymbolic project and follows the same licensing terms. 