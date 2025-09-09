#!/usr/bin/env python3
"""
Market Data Adapters for HFT Neurosymbolic AI System
Supports yfinance, CCXT, and alternative data sources
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import aiohttp
import json

# Data source imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from hft_components.structured_logging import logger, with_correlation_id, performance_logger

logger = logging.getLogger(__name__)

class MarketDataAdapter(ABC):
    """Abstract base class for market data adapters"""
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = "1d",
        period: str = "1y"
    ) -> Dict[str, Any]:
        """Get historical market data"""
        pass
    
    @abstractmethod
    async def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if adapter is available"""
        pass

class YahooFinanceAdapter(MarketDataAdapter):
    """Yahoo Finance data adapter"""
    
    def __init__(self):
        self.name = "yfinance"
        self.available = YFINANCE_AVAILABLE
        
    def is_available(self) -> bool:
        return self.available
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = "1d",
        period: str = "1y"
    ) -> Dict[str, Any]:
        """Get historical data from Yahoo Finance"""
        if not self.available:
            return {"error": "yfinance not available"}
        
        try:
            with with_correlation_id():
                logger.info("Fetching Yahoo Finance data", symbol=symbol, timeframe=timeframe, period=period)
                
                # Map timeframe to yfinance interval
                interval_map = {
                    "1m": "1m",
                    "5m": "5m", 
                    "15m": "15m",
                    "30m": "30m",
                    "1h": "1h",
                    "1d": "1d",
                    "1wk": "1wk",
                    "1mo": "1mo"
                }
                
                interval = interval_map.get(timeframe, "1d")
                
                # Create ticker
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                hist = ticker.history(period=period, interval=interval)
                
                if hist.empty:
                    return {"error": f"No data found for {symbol}"}
                
                # Convert to our format
                data = []
                for date, row in hist.iterrows():
                    data.append({
                        "timestamp": date.isoformat(),
                        "date": date.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                        "adj_close": float(row["Adj Close"]) if "Adj Close" in row else float(row["Close"])
                    })
                
                # Get additional info
                info = ticker.info
                
                result = {
                    "symbol": symbol,
                    "source": "yfinance",
                    "timeframe": timeframe,
                    "period": period,
                    "data_points": len(data),
                    "data": data,
                    "metadata": {
                        "company_name": info.get("longName", ""),
                        "sector": info.get("sector", ""),
                        "industry": info.get("industry", ""),
                        "market_cap": info.get("marketCap", 0),
                        "currency": info.get("currency", "USD")
                    }
                }
                
                logger.info("Yahoo Finance data fetched successfully", 
                           symbol=symbol, data_points=len(data))
                
                return result
                
        except Exception as e:
            logger.error("Failed to fetch Yahoo Finance data", symbol=symbol, error=str(e))
            return {"error": str(e)}
    
    async def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time data from Yahoo Finance"""
        if not self.available:
            return {"error": "yfinance not available"}
        
        try:
            with with_correlation_id():
                logger.info("Fetching real-time Yahoo Finance data", symbol=symbol)
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price
                hist = ticker.history(period="1d", interval="1m")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
                
                result = {
                    "symbol": symbol,
                    "source": "yfinance",
                    "timestamp": datetime.now().isoformat(),
                    "current_price": current_price,
                    "previous_close": info.get("previousClose", 0),
                    "open": info.get("open", 0),
                    "day_high": info.get("dayHigh", 0),
                    "day_low": info.get("dayLow", 0),
                    "volume": info.get("volume", 0),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "dividend_yield": info.get("dividendYield", 0)
                }
                
                logger.info("Real-time Yahoo Finance data fetched", symbol=symbol, price=current_price)
                return result
                
        except Exception as e:
            logger.error("Failed to fetch real-time Yahoo Finance data", symbol=symbol, error=str(e))
            return {"error": str(e)}

class CCXTAdapter(MarketDataAdapter):
    """CCXT cryptocurrency data adapter"""
    
    def __init__(self, exchange: str = "binance"):
        self.name = "ccxt"
        self.exchange_name = exchange
        self.available = CCXT_AVAILABLE
        self.exchange = None
        
        if self.available:
            try:
                exchange_class = getattr(ccxt, exchange)
                self.exchange = exchange_class({
                    'apiKey': '',  # Add API keys if needed
                    'secret': '',
                    'sandbox': True,  # Use sandbox for testing
                    'enableRateLimit': True,
                })
            except Exception as e:
                logger.error(f"Failed to initialize {exchange} exchange", error=str(e))
                self.available = False
    
    def is_available(self) -> bool:
        return self.available and self.exchange is not None
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = "1d",
        period: str = "1y"
    ) -> Dict[str, Any]:
        """Get historical cryptocurrency data"""
        if not self.available:
            return {"error": "CCXT not available"}
        
        try:
            with with_correlation_id():
                logger.info("Fetching CCXT data", symbol=symbol, exchange=self.exchange_name, timeframe=timeframe)
                
                # Map timeframe to CCXT timeframe
                timeframe_map = {
                    "1m": "1m",
                    "5m": "5m",
                    "15m": "15m", 
                    "30m": "30m",
                    "1h": "1h",
                    "4h": "4h",
                    "1d": "1d",
                    "1wk": "1w",
                    "1mo": "1M"
                }
                
                tf = timeframe_map.get(timeframe, "1d")
                
                # Calculate limit based on period
                period_days = {
                    "1d": 1,
                    "1wk": 7,
                    "1mo": 30,
                    "3mo": 90,
                    "6mo": 180,
                    "1y": 365
                }
                
                days = period_days.get(period, 365)
                limit = min(days * (24 if "h" in timeframe else 1), 1000)  # CCXT limit
                
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                
                if not ohlcv:
                    return {"error": f"No data found for {symbol}"}
                
                # Convert to our format
                data = []
                for candle in ohlcv:
                    timestamp = datetime.fromtimestamp(candle[0] / 1000)
                    data.append({
                        "timestamp": timestamp.isoformat(),
                        "date": timestamp.strftime("%Y-%m-%d"),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    })
                
                result = {
                    "symbol": symbol,
                    "source": f"ccxt_{self.exchange_name}",
                    "timeframe": timeframe,
                    "period": period,
                    "data_points": len(data),
                    "data": data,
                    "metadata": {
                        "exchange": self.exchange_name,
                        "market_type": "cryptocurrency"
                    }
                }
                
                logger.info("CCXT data fetched successfully", 
                           symbol=symbol, exchange=self.exchange_name, data_points=len(data))
                
                return result
                
        except Exception as e:
            logger.error("Failed to fetch CCXT data", symbol=symbol, error=str(e))
            return {"error": str(e)}
    
    async def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time cryptocurrency data"""
        if not self.available:
            return {"error": "CCXT not available"}
        
        try:
            with with_correlation_id():
                logger.info("Fetching real-time CCXT data", symbol=symbol, exchange=self.exchange_name)
                
                # Get ticker data
                ticker = self.exchange.fetch_ticker(symbol)
                
                result = {
                    "symbol": symbol,
                    "source": f"ccxt_{self.exchange_name}",
                    "timestamp": datetime.now().isoformat(),
                    "current_price": ticker.get("last", 0),
                    "bid": ticker.get("bid", 0),
                    "ask": ticker.get("ask", 0),
                    "high": ticker.get("high", 0),
                    "low": ticker.get("low", 0),
                    "volume": ticker.get("baseVolume", 0),
                    "change": ticker.get("change", 0),
                    "percentage": ticker.get("percentage", 0)
                }
                
                logger.info("Real-time CCXT data fetched", symbol=symbol, price=ticker.get("last", 0))
                return result
                
        except Exception as e:
            logger.error("Failed to fetch real-time CCXT data", symbol=symbol, error=str(e))
            return {"error": str(e)}

class AlternativeDataAdapter(MarketDataAdapter):
    """Alternative data adapter for news, sentiment, etc."""
    
    def __init__(self):
        self.name = "alternative"
        self.available = True  # Always available for basic implementation
    
    def is_available(self) -> bool:
        return self.available
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = "1d",
        period: str = "1y"
    ) -> Dict[str, Any]:
        """Get alternative data (placeholder implementation)"""
        # This would integrate with news APIs, sentiment analysis, etc.
        return {
            "symbol": symbol,
            "source": "alternative",
            "timeframe": timeframe,
            "period": period,
            "data_points": 0,
            "data": [],
            "metadata": {
                "data_type": "news_sentiment",
                "note": "Alternative data integration placeholder"
            }
        }
    
    async def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time alternative data"""
        return {
            "symbol": symbol,
            "source": "alternative",
            "timestamp": datetime.now().isoformat(),
            "sentiment_score": 0.0,
            "news_count": 0,
            "social_mentions": 0
        }

class DataAdapterManager:
    """Manager for all data adapters"""
    
    def __init__(self):
        self.adapters = {
            "yfinance": YahooFinanceAdapter(),
            "ccxt_binance": CCXTAdapter("binance"),
            "ccxt_coinbase": CCXTAdapter("coinbasepro"),
            "alternative": AlternativeDataAdapter()
        }
        
        # Log available adapters
        available_adapters = [name for name, adapter in self.adapters.items() if adapter.is_available()]
        logger.info("Data adapters initialized", available_adapters=available_adapters)
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapters"""
        return [name for name, adapter in self.adapters.items() if adapter.is_available()]
    
    async def get_data(
        self, 
        symbol: str, 
        source: str = "yfinance",
        timeframe: str = "1d",
        period: str = "1y",
        data_type: str = "historical"
    ) -> Dict[str, Any]:
        """Get data from specified adapter"""
        
        if source not in self.adapters:
            return {"error": f"Unknown data source: {source}"}
        
        adapter = self.adapters[source]
        
        if not adapter.is_available():
            return {"error": f"Data source {source} is not available"}
        
        try:
            with with_correlation_id():
                start_time = time.time()
                
                if data_type == "historical":
                    result = await adapter.get_historical_data(symbol, timeframe, period)
                elif data_type == "realtime":
                    result = await adapter.get_realtime_data(symbol)
                else:
                    return {"error": f"Unknown data type: {data_type}"}
                
                duration = time.time() - start_time
                performance_logger.log_performance(
                    "data_fetch",
                    duration * 1000,
                    source=source,
                    symbol=symbol,
                    data_type=data_type
                )
                
                return result
                
        except Exception as e:
            logger.error("Data fetch failed", source=source, symbol=symbol, error=str(e))
            return {"error": str(e)}
    
    async def get_multi_source_data(
        self, 
        symbol: str, 
        sources: List[str] = None,
        timeframe: str = "1d",
        period: str = "1y"
    ) -> Dict[str, Any]:
        """Get data from multiple sources"""
        
        if sources is None:
            sources = self.get_available_adapters()
        
        results = {}
        
        # Fetch data from all sources concurrently
        tasks = []
        for source in sources:
            task = self.get_data(symbol, source, timeframe, period, "historical")
            tasks.append(task)
        
        source_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, source in enumerate(sources):
            result = source_results[i]
            if isinstance(result, Exception):
                results[source] = {"error": str(result)}
            else:
                results[source] = result
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "period": period,
            "sources": results,
            "timestamp": datetime.now().isoformat()
        }
