#!/usr/bin/env python3
"""
Load minimal test data into Neo4j for HFT system
"""

import asyncio
import logging
from neo4j import AsyncGraphDatabase
import json
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jDataLoader:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="hft_password_2025"):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def close(self):
        await self.driver.close()
    
    async def load_test_data(self):
        """Load minimal test data for AAPL and TSLA"""
        async with self.driver.session() as session:
            try:
                # Clear existing data
                await session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared existing Neo4j data")
                
                # Generate test data for AAPL
                await self._load_symbol_data(session, "AAPL", 100, 150.0)
                
                # Generate test data for TSLA
                await self._load_symbol_data(session, "TSLA", 100, 200.0)
                
                logger.info("Successfully loaded test data into Neo4j")
                
            except Exception as e:
                logger.error(f"Failed to load test data: {e}")
                raise
    
    async def _load_symbol_data(self, session, symbol, num_records, base_price):
        """Load price data for a symbol"""
        try:
            current_price = base_price
            current_time = datetime.now() - timedelta(days=num_records)
            
            for i in range(num_records):
                # Generate realistic price movement
                change_percent = random.uniform(-0.05, 0.05)  # ±5% daily change
                current_price *= (1 + change_percent)
                volume = random.randint(1000000, 10000000)
                
                # Create price node
                await session.run("""
                    CREATE (p:Price {
                        symbol: $symbol,
                        price: $price,
                        volume: $volume,
                        timestamp: $timestamp,
                        date: $date
                    })
                """, {
                    "symbol": symbol,
                    "price": round(current_price, 2),
                    "volume": volume,
                    "timestamp": current_time.isoformat(),
                    "date": current_time.strftime("%Y-%m-%d")
                })
                
                current_time += timedelta(days=1)
            
            logger.info(f"Loaded {num_records} records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            raise

async def main():
    loader = Neo4jDataLoader()
    try:
        await loader.load_test_data()
    finally:
        await loader.close()

if __name__ == "__main__":
    asyncio.run(main())
