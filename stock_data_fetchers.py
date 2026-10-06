import requests
from yfinance import Ticker
from yfinance.exceptions import YFException
from abc import ABC, abstractmethod
from blpapi import Session, Service, Request
import blpapi
from langchain_core.tools import tool

class BaseStockDataFetcher(ABC):
    @abstractmethod
    def get_stock_data(self, ticker: str):
        pass

    @abstractmethod
    def get_stock_news(self, ticker: str):
        pass

    @abstractmethod
    def get_stock_analyst_price_targets(self, ticker: str):
        pass

    @abstractmethod
    def get_stock_quarterly_data(self, ticker: str):
        pass

    @abstractmethod
    def get_stock_calendar(self, ticker: str):
        pass

class YahooFinanceStockDataFetcher(BaseStockDataFetcher):
    def get_stock_data(self, ticker: str):
        try:
            stock = Ticker(ticker)
            return stock.info
        except YFException as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return None

    def get_stock_news(self, ticker: str):
        try:
            stock = Ticker(ticker)
            return stock.news
        except YFException as e:
            print(f"Error fetching stock news for {ticker}: {e}")
            return None

    def get_stock_analyst_price_targets(self, ticker: str):
        try:
            stock = Ticker(ticker)
            return stock.analyst_price_targets
        except YFException as e:
            print(f"Error fetching analyst price targets for {ticker}: {e}")
            return None

    def get_stock_quarterly_data(self, ticker: str):
        try:
            stock = Ticker(ticker)
            return {
                "quarterly_financials": stock.quarterly_financials,
                "quarterly_earnings": stock.quarterly_earnings,
                "quarterly_balance_sheet": stock.quarterly_balance_sheet,
                "quarterly_cashflow": stock.quarterly_cashflow,
                "quarterly_income_stmt": stock.quarterly_income_stmt,
            }
        except YFException as e:
            print(f"Error fetching quarterly data for {ticker}: {e}")
            return None

    def get_stock_calendar(self, ticker: str):
        try:
            stock = Ticker(ticker)
            return stock.calendar
        except YFException as e:
            print(f"Error fetching stock calendar for {ticker}: {e}")
            return None

class AlphaVantageStockDataFetcher(BaseStockDataFetcher):
    # Just an example code. Actually, I don't know if this will work with the real Alpha Vantage API
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_stock_data(self, ticker: str):
        try:
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return None

    def get_stock_news(self, ticker: str):
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching stock news for {ticker}: {e}")
            return None

    def get_stock_analyst_price_targets(self, ticker: str):
        # Alpha Vantage does not provide analyst price targets directly
        print(f"Analyst price targets not available for {ticker} via Alpha Vantage")
        return None

    def get_stock_quarterly_data(self, ticker: str):
        try:
            url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("quarterlyReports", None)
        except requests.RequestException as e:
            print(f"Error fetching quarterly data for {ticker}: {e}")
            return None

    def get_stock_calendar(self, ticker: str):
        # Alpha Vantage does not provide a stock calendar directly
        print(f"Stock calendar not available for {ticker} via Alpha Vantage")
        return None

class BloombergStockDataFetcher(BaseStockDataFetcher):
    # Just an example code. Actually, I don't know if this will work with the real Bloomberg API
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_stock_data(self, ticker: str):
        try:
            session = Session()
            if not session.start():
                print(f"Error starting Bloomberg session for {ticker}")
                return None
            if not session.openService("//blp/refdata"):
                print(f"Error opening Bloomberg service for {ticker}")
                return None
            service = session.getService("//blp/refdata")
            request = service.createRequest("ReferenceDataRequest")
            request.getElement("securities").appendValue(ticker)
            request.getElement("fields").appendValue("PX_LAST")
            session.sendRequest(request)
            while True:
                event = session.nextEvent()
                for msg in event:
                    if msg.hasElement("securityData"):
                        return msg.getElement("securityData").getValueAsElement(0).getElement("fieldData").getElement("PX_LAST").getValue()
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return None

    def get_stock_news(self, ticker: str):
        try:
            session = Session()
            if not session.start():
                print(f"Error starting Bloomberg session for {ticker}")
                return None
            if not session.openService("//blp/news"):
                print(f"Error opening Bloomberg news service for {ticker}")
                return None
            service = session.getService("//blp/news")
            request = service.createRequest("NewsRequest")
            request.getElement("securities").appendValue(ticker)
            session.sendRequest(request)
            while True:
                event = session.nextEvent()
                for msg in event:
                    if msg.hasElement("news"):
                        return msg.getElement("news").getValueAsElement(0).getElement("headline").getValue()
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        except Exception as e:
            print(f"Error fetching stock news for {ticker}: {e}")
            return None
        print(f"Stock news not available for {ticker} via Bloomberg")
        return None

    def get_stock_analyst_price_targets(self, ticker: str):
        try:
            session = Session()
            if not session.start():
                print(f"Error starting Bloomberg session for {ticker}")
                return None
            if not session.openService("//blp/analyst"):
                print(f"Error opening Bloomberg analyst service for {ticker}")
                return None
            service = session.getService("//blp/analyst")
            request = service.createRequest("AnalystPriceTargetRequest")
            request.getElement("securities").appendValue(ticker)
            session.sendRequest(request)
            while True:
                event = session.nextEvent()
                for msg in event:
                    if msg.hasElement("analystPriceTargets"):
                        return msg.getElement("analystPriceTargets").getValueAsElement(0).getElement("targetPrice").getValue()
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        except Exception as e:
            print(f"Error fetching analyst price targets for {ticker}: {e}")
            return None
        print(f"Analyst price targets not available for {ticker} via Bloomberg")
        return None

    def get_stock_quarterly_data(self, ticker: str):
        try:
            session = Session()
            if not session.start():
                print(f"Error starting Bloomberg session for {ticker}")
                return None
            if not session.openService("//blp/quarterly"):
                print(f"Error opening Bloomberg quarterly data service for {ticker}")
                return None
            service = session.getService("//blp/quarterly")
            request = service.createRequest("QuarterlyDataRequest")
            request.getElement("securities").appendValue(ticker)
            session.sendRequest(request)
            while True:
                event = session.nextEvent()
                for msg in event:
                    if msg.hasElement("quarterlyData"):
                        return msg.getElement("quarterlyData").getValueAsElement(0).getElement("data").getValue()
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        except Exception as e:
            print(f"Error fetching quarterly data for {ticker}: {e}")
            return None
        print(f"Quarterly data not available for {ticker} via Bloomberg")
        return None

    def get_stock_calendar(self, ticker: str):
        try:
            session = Session()
            if not session.start():
                print(f"Error starting Bloomberg session for {ticker}")
                return None
            if not session.openService("//blp/calendar"):
                print(f"Error opening Bloomberg stock calendar service for {ticker}")
                return None
            service = session.getService("//blp/calendar")
            request = service.createRequest("StockCalendarRequest")
            request.getElement("securities").appendValue(ticker)
            session.sendRequest(request)
            while True:
                event = session.nextEvent()
                for msg in event:
                    if msg.hasElement("stockCalendar"):
                        return msg.getElement("stockCalendar").getValueAsElement(0).getElement("data").getValue()
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        except Exception as e:
            print(f"Error fetching stock calendar for {ticker}: {e}")
            return None
        print(f"Stock calendar not available for {ticker} via Bloomberg")
        return None
