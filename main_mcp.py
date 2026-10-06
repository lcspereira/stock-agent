from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from pprint import pprint
from datetime import datetime
import asyncio
import os

from main_langgraph import _build_system_prompt


def _model(temperature: float) -> ChatOllama:
    return ChatOllama(
        model="glm-5.2",
        base_url="https://ollama.com",
        api_key=os.getenv("OLLAMA_API_KEY"),
        temperature=temperature,
    )


async def _run_agent(agent, query: str, thread_id: str) -> str:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].text


def build_specialist_tools(mcp_tools: list):
    by_name = {t.name: t for t in mcp_tools}

    def pick(*names: str) -> list:
        return [by_name[n] for n in names]

    @tool
    async def call_fundamentals_agent(query: str) -> str:
        """Calls the fundamentals agent to assess a stock's financial trends, price targets and quarterly data."""
        agent = create_agent(
            model=_model(0.0),
            tools=pick("get_stock_data", "get_stock_analyst_price_targets", "get_stock_quarterly_data"),
            system_prompt="You are a fundamentals analyst. Assess financial trends and clearly distinguish data from interpretation.",
            checkpointer=InMemorySaver(),
        )
        return await _run_agent(agent, query, "fundamentals")

    @tool
    async def call_catalyst_analyst_agent(query: str) -> str:
        """Calls the catalyst analyst agent to analyze upcoming catalyst events such as earnings and dividends."""
        agent = create_agent(
            model=_model(0.0),
            tools=pick("get_stock_calendar"),
            system_prompt="You are a catalyst analyst. You analyze catalyst events for a stock and explain their potential impact.",
            checkpointer=InMemorySaver(),
        )
        return await _run_agent(agent, query, "catalyst-analyst")

    @tool
    async def call_news_agent(query: str) -> str:
        """Calls the news agent to analyze the latest news for a stock."""
        agent = create_agent(
            model=_model(0.3),
            tools=pick("get_stock_news"),
            system_prompt="You are a news analyst. You analyze stock news and summarize how it may affect the stock.",
            checkpointer=InMemorySaver(),
        )
        return await _run_agent(agent, query, "news")

    return [call_fundamentals_agent, call_catalyst_analyst_agent, call_news_agent]


async def main():
    client = MultiServerMCPClient({
        "stock": {
            "transport": "http",
            "url": "http://localhost:5678/mcp",
        }
    })
    mcp_tools = await client.get_tools()
    agent = create_agent(
        model=_model(0.4),
        tools=build_specialist_tools(mcp_tools),
        system_prompt=_build_system_prompt(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "1"}}
    while True:
        query = input("Enter your stock-related query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        start_time = datetime.now()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )
        end_time = datetime.now()
        pprint(result["messages"][-1].content_blocks)
        print(f"Response time: {(end_time - start_time).total_seconds():.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
