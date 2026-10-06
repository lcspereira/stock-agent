import asyncio
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastmcp import FastMCP, Context
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from stock_data_fetchers import YahooFinanceStockDataFetcher
from yfinance.exceptions import YFException
from bs4 import BeautifulSoup
import requests


mcp = FastMCP("stock-agent-tools")
stock_data_fetcher = YahooFinanceStockDataFetcher()


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of yfinance/pandas values into JSON-safe objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    # pandas/numpy objects usually expose one of these conversion methods.
    for method_name in ("to_dict", "tolist", "item"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _to_jsonable(method())
            except Exception:
                pass

    return str(value)


def _require(value: Any, ticker: str) -> Any:
    if value is None:
        raise RuntimeError(f"No data returned for {ticker}")
    return value


def _sync_get_stock_data(ticker: str) -> dict[str, Any]:
    return _to_jsonable(_require(stock_data_fetcher.get_stock_data(ticker), ticker))


@mcp.tool
async def get_stock_data(ticker: str) -> dict[str, Any]:
    """Fetches current stock data for the given ticker symbol using yfinance."""
    try:
        return await asyncio.to_thread(_sync_get_stock_data, ticker)
    except YFException as exc:
        return {"error": f"Error fetching stock data for {ticker}: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error fetching stock data for {ticker}: {exc}"}


def _sync_get_stock_calendar(ticker: str) -> dict[str, Any]:
    return _to_jsonable(_require(stock_data_fetcher.get_stock_calendar(ticker), ticker))


@mcp.tool
async def get_stock_calendar(ticker: str) -> dict[str, Any]:
    """Fetches the stock calendar for the given ticker symbol using yfinance."""
    try:
        return await asyncio.to_thread(_sync_get_stock_calendar, ticker)
    except YFException as exc:
        return {"error": f"Error fetching stock calendar for {ticker}: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error fetching stock calendar for {ticker}: {exc}"}


def _sync_get_stock_analyst_price_targets(ticker: str) -> dict[str, Any]:
    return _to_jsonable(
        _require(stock_data_fetcher.get_stock_analyst_price_targets(ticker), ticker)
    )


@mcp.tool
async def get_stock_analyst_price_targets(ticker: str) -> dict[str, Any]:
    """Fetches analyst price targets for the given ticker symbol using yfinance."""
    try:
        return await asyncio.to_thread(_sync_get_stock_analyst_price_targets, ticker)
    except YFException as exc:
        return {"error": f"Error fetching analyst price targets for {ticker}: {exc}"}
    except Exception as exc:
        return {
            "error": f"Unexpected error fetching analyst price targets for {ticker}: {exc}"
        }


def _sync_get_stock_quarterly_data(ticker: str) -> dict[str, Any]:
    return _to_jsonable(
        _require(stock_data_fetcher.get_stock_quarterly_data(ticker), ticker)
    )


@mcp.tool
async def get_stock_quarterly_data(ticker: str) -> dict[str, Any]:
    """Fetches quarterly financial data for the given ticker symbol using yfinance."""
    try:
        return await asyncio.to_thread(_sync_get_stock_quarterly_data, ticker)
    except YFException as exc:
        return {"error": f"Error fetching quarterly data for {ticker}: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error fetching quarterly data for {ticker}: {exc}"}


def _fetch_article(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": ""})
    page_content = response.text
    soup = BeautifulSoup(page_content, "html.parser")
    article_content = soup.find(class_="body").get_text(separator="\n").strip() if soup.find(class_="body") else soup.get_text(separator="\n").strip()
    return article_content


def _sync_get_stock_news(ticker: str) -> list[dict[str, Any]]:
    raw_news = _require(stock_data_fetcher.get_stock_news(ticker), ticker)

    content: list[dict[str, Any]] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for news_item in raw_news:
        if "content" not in news_item:
            title = news_item.get("title", "")
            url = news_item.get("canonicalUrl", {}).get("url", "")
            published = news_item.get("published")
        else:
            article = news_item.get("content", {})
            title = article.get("title", "")
            url = article.get("canonicalUrl", {}).get("url", "")
            published = article.get("pubDate")

        if not url:
            continue

        try:
            article_content = _fetch_article(url)
        except Exception:
            article_content = ""

        content.append(
            {
                "title": title,
                "link": url,
                "published": _to_jsonable(published),
                "content_blocks": splitter.split_text(article_content),
            }
        )

    return _to_jsonable(content)


@mcp.tool
async def get_stock_news(ticker: str, ctx: Context) -> list[dict[str, Any]]:
    """Fetches latest news and split article content blocks for a ticker symbol."""
    try:
        return await asyncio.to_thread(_sync_get_stock_news, ticker)
    except YFException as exc:
        return [{"error": f"Error fetching news for {ticker}: {exc}"}]
    except Exception as exc:
        return [{"error": f"Unexpected error fetching news for {ticker}: {exc}"}]


if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=5678)
