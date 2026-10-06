from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from yfinance import Ticker
from yfinance.exceptions import YFException
from pprint import pprint
from datetime import datetime, timedelta
from langgraph.checkpoint.memory import InMemorySaver
import os
from stock_data_fetchers import YahooFinanceStockDataFetcher

stock_data_fetcher = YahooFinanceStockDataFetcher()

def get_stock_data(ticker: str) -> dict:
    """Fetches the current stock price for the given ticker symbol using yfinance."""
    print (f"Fetching stock data for {ticker}...")
    try:
        return stock_data_fetcher.get_stock_data(ticker)
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {e}"

def get_stock_calendar(ticker: str) -> dict:
    """Fetches the stock calendar for the given ticker symbol using yfinance."""
    print (f"Fetching stock calendar for {ticker}...")
    try:
        return stock_data_fetcher.get_stock_calendar(ticker)
    except Exception as e:
        return f"Error fetching stock calendar for {ticker}: {e}"

def get_stock_analyst_price_targets(ticker: str) -> dict:
    """Fetches the analyst price targets for the given ticker symbol using yfinance."""
    print (f"Fetching analyst price targets for {ticker}...")
    try:
        return stock_data_fetcher.get_stock_analyst_price_targets(ticker)
    except Exception as e:
        return f"Error fetching analyst price targets for {ticker}: {e}"

def get_stock_quarterly_data(ticker: str) -> dict:
    """Fetches the quarterly financial data for the given ticker symbol using yfinance."""
    print (f"Fetching quarterly data for {ticker}...")
    try:
        return stock_data_fetcher.get_stock_quarterly_data(ticker)
    except YFException as e:
        return f"Error fetching quarterly data for {ticker}: {e}"

def get_stock_news(ticker: str) -> list:
    """Fetches the latest news articles related to the given ticker symbol using yfinance."""
    print (f"Fetching news for {ticker}...")
    try:
        content = []
        for news in stock_data_fetcher.get_stock_news(ticker):
            try:
                if "content" not in news:
                    loader = WebBaseLoader(news["clickThroughUrl"]["url"])
                    article_content = loader.load()[0].page_content
                    content.append({
                        "title": news["title"],
                        "link": news["clickThroughUrl"]["url"],
                        "published": news["published"],
                        "content": article_content
                    })
                else:
                    loader = WebBaseLoader(news['content']["clickThroughUrl"]["url"])
                    article_content = loader.load()[0].page_content
                    content.append({
                        "title": news['content']["title"],
                        "link": news['content']["clickThroughUrl"]["url"],
                        "published": news['content']["pubDate"],
                        "content": article_content
                    })
            except Exception as e:
                continue  # Skip articles that cannot be loaded

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for item in content:
            item["content_blocks"] = text_splitter.split_text(item["content"])
            del item["content"]
    except Exception as e:
        return f"Error fetching news for {ticker}: {e}"

    return content

def call_fundamentals_agent(query):
    """Call the fundamentals agent with the given query."""
    print(f"Calling fundamentals agent with query: {query}")
    fundamentals_agent = create_deep_agent(
        tools=[get_stock_data, get_stock_analyst_price_targets, get_stock_quarterly_data],
        model=ChatOllama(
            model="glm-5.2",
            temperature=0.0,
            base_url="https://ollama.com",
            api_key=os.getenv("OLLAMA_API_KEY"),
        ),
        system_prompt="You are a helpful assistant that assess financial trends and clearly distinguish data from interpretation.",
        memory=["./AGENTS.md"],
        checkpointer=InMemorySaver(),
    )
    return fundamentals_agent.invoke({
        "messages": [{
            "role": "user",
            "content": query
        }]
    }, config={"configurable": {"thread_id": "fundamentals"}})

def call_catalyst_analyst_agent(query):
    """Call the catalyst analyst agent with the given query."""
    print(f"Calling catalyst analyst agent with query: {query}")
    catalyst_analyst_agent = create_deep_agent(
        tools=[get_stock_calendar],
        model=ChatOllama(
            model="glm-5.2",
            temperature=0.0,
            base_url="https://ollama.com",
            api_key=os.getenv("OLLAMA_API_KEY"),
        ),
        system_prompt="You are a helpful assistant that analyzes catalyst events and provides insights based on the information available.",
        memory=["./AGENTS.md"],
        checkpointer=InMemorySaver(),
    )
    return catalyst_analyst_agent.invoke({
        "messages": [{
            "role": "user",
            "content": query
        }]
    }, config={"configurable": {"thread_id": "catalyst-analyst"}})

def call_news_agent(query):
    """Call the news agent with the given query."""
    print(f"Calling news agent with query: {query}")
    news_agent = create_deep_agent(
        tools=[get_stock_news],
        model=ChatOllama(
            model="glm-5.2",
            temperature=0.3,
            base_url="https://ollama.com",
            api_key=os.getenv("OLLAMA_API_KEY"),
        ),
        system_prompt="You are a helpful assistant that analyzes stock news and provides insights based on the information available.",
        memory=["./AGENTS.md"],
        checkpointer=InMemorySaver(),
    )
    return news_agent.invoke({
        "messages": [{
            "role": "user",
            "content": query
        }]
    }, config={"configurable": {"thread_id": "news"}})

def main():
    
    main_agent = create_deep_agent(
        tools=[call_fundamentals_agent, call_catalyst_analyst_agent, call_news_agent],
        model=ChatOllama(
            model="glm-5.2",
            temperature=0.4,
            base_url="https://ollama.com",
            api_key=os.getenv("OLLAMA_API_KEY"),
        ),
        system_prompt="You are a helpful assistant that analyzes stock data and provides insights based on the information available.",
        memory=["./AGENTS.md"],
        checkpointer=InMemorySaver(),
    )

    while True:
        query = input("Enter a stock-related query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        start_time = datetime.now()
        result = main_agent.invoke({
            "messages": [{
                "role": "user",
                "content": query
            }]
        }, config={"configurable": {"thread_id": "1"}})
        end_time = datetime.now()
        pprint(result["messages"][-1].content_blocks)
        print(f"Response time: {(end_time - start_time).total_seconds():.2f} seconds")


if __name__ == "__main__":
    main()
