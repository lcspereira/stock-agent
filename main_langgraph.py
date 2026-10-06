from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from datetime import datetime
import os
from bs4 import BeautifulSoup

from stock_data_fetchers import YahooFinanceStockDataFetcher

stock_data_fetcher = YahooFinanceStockDataFetcher()


def _model(temperature: float) -> ChatOllama:
    return ChatOllama(
        model="glm-5.2",
        base_url="https://ollama.com",
        api_key=os.getenv("OLLAMA_API_KEY"),
        temperature=temperature,
    )


@tool
def get_stock_data(ticker: str) -> dict:
    """Fetches the current stock price for the given ticker symbol using yfinance."""
    print(f"Fetching stock data for {ticker}...")
    return stock_data_fetcher.get_stock_data(ticker) or {}


@tool
def get_stock_calendar(ticker: str) -> dict:
    """Fetches the stock calendar for the given ticker symbol using yfinance."""
    print(f"Fetching stock calendar for {ticker}...")
    return stock_data_fetcher.get_stock_calendar(ticker) or {}


@tool
def get_stock_analyst_price_targets(ticker: str) -> dict:
    """Fetches the analyst price targets for the given ticker symbol using yfinance."""
    print(f"Fetching analyst price targets for {ticker}...")
    return stock_data_fetcher.get_stock_analyst_price_targets(ticker) or {}


@tool
def get_stock_quarterly_data(ticker: str) -> dict:
    """Fetches the quarterly financial data for the given ticker symbol using yfinance."""
    print(f"Fetching quarterly data for {ticker}...")
    return stock_data_fetcher.get_stock_quarterly_data(ticker) or {}


@tool
def get_stock_news(ticker: str) -> list:
    """Fetches the latest news articles related to the given ticker symbol using yfinance."""
    print(f"Fetching news for {ticker}...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    content = []
    for news in stock_data_fetcher.get_stock_news(ticker) or []:
        article = news.get("content", news)
        url = article.get("canonicalUrl", {}).get("url")
        if not url:
            continue
        page = WebBaseLoader(url).load()[0].page_content
        body = BeautifulSoup(page, "html.parser").find(class_="body")
        text = body.get_text(separator="\n").strip() if body else page
        content.append({
            "title": article.get("title"),
            "link": url,
            "published": article.get("pubDate", article.get("published")),
            "content_blocks": text_splitter.split_text(text),
        })
    return content


def _build_system_prompt() -> str:
    base = "You are a helpful assistant that analyzes stock data and provides insights based on the information available."
    agents_file = "./AGENTS.md"
    if os.path.exists(agents_file):
        with open(agents_file, "r") as f:
            return f.read().strip() + "\n\n" + base
    return base


def build_agent_graph(model: BaseChatModel, tools: list, system_prompt: str):
    llm_with_tools = model.bind_tools(tools)

    def assistant_node(state: MessagesState):
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools=tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=MemorySaver())


def _run_agent(graph, query: str, thread_id: str) -> str:
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].text


@tool
def call_fundamentals_agent(query: str) -> str:
    """Calls the fundamentals agent to assess a stock's financial trends, price targets and quarterly data."""
    graph = build_agent_graph(
        _model(0.0),
        [get_stock_data, get_stock_analyst_price_targets, get_stock_quarterly_data],
        "You are a fundamentals analyst. Assess financial trends and clearly distinguish data from interpretation.",
    )
    return _run_agent(graph, query, "fundamentals")


@tool
def call_catalyst_analyst_agent(query: str) -> str:
    """Calls the catalyst analyst agent to analyze upcoming catalyst events such as earnings and dividends."""
    graph = build_agent_graph(
        _model(0.0),
        [get_stock_calendar],
        "You are a catalyst analyst. You analyze catalyst events for a stock and explain their potential impact.",
    )
    return _run_agent(graph, query, "catalyst-analyst")


@tool
def call_news_agent(query: str) -> str:
    """Calls the news agent to analyze the latest news for a stock."""
    graph = build_agent_graph(
        _model(0.3),
        [get_stock_news],
        "You are a news analyst. You analyze stock news and summarize how it may affect the stock.",
    )
    return _run_agent(graph, query, "news")


def build_stock_graph():
    return build_agent_graph(
        _model(0.4),
        [call_fundamentals_agent, call_catalyst_analyst_agent, call_news_agent],
        _build_system_prompt(),
    )


def main():
    graph = build_stock_graph()
    config = {"configurable": {"thread_id": "1"}}

    while True:
        query = input("Enter a stock-related query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        start_time = datetime.now()
        result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        end_time = datetime.now()
        pprint(result["messages"][-1].content_blocks)
        print(f"Response time: {(end_time - start_time).total_seconds():.2f} seconds")


if __name__ == "__main__":
    main()
