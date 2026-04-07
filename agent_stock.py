"""
Simple Stock Analyst ReAct Agent
=================================
A bare-bones agent to understand the core loop.

Install:
    pip install anthropic yfinance pandas pandas-ta

Run:
    export ANTHROPIC_API_KEY=your_key
    python simple_agent.py
"""

import json
import anthropic
import yfinance as yf
import pandas_ta as ta
from dotenv import load_dotenv
import os

load_dotenv()  # Loads ANTHROPIC_API_KEY from .env file


# ─────────────────────────────────────────────────────────────
# STEP 1 — Create the Anthropic client
# ─────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  # reads ANTHROPIC_API_KEY from env


# ─────────────────────────────────────────────────────────────
# STEP 2 — Define your tools (just plain Python functions)
# ─────────────────────────────────────────────────────────────

def get_stock_price(ticker: str) -> dict:
    """Fetch the live price for a stock ticker."""
    stock = yf.Ticker(ticker)
    price = stock.fast_info.last_price
    prev  = stock.fast_info.previous_close
    change_pct = ((price - prev) / prev) * 100

    return {
        "ticker": ticker.upper(),
        "price": round(price, 2),
        "change_pct": round(change_pct, 2),
    }


def get_technical_indicators(ticker: str) -> dict:
    """Compute RSI and SMA(20) for a stock."""
    df    = yf.Ticker(ticker).history(period="3mo")
    close = df["Close"]

    rsi   = ta.rsi(close, length=14).dropna().iloc[-1]
    sma20 = ta.sma(close, length=20).dropna().iloc[-1]

    return {
        "ticker": ticker.upper(),
        "rsi":    round(float(rsi), 2),
        "sma20":  round(float(sma20), 2),
        "price_vs_sma20": "above" if close.iloc[-1] > sma20 else "below",
    }


# ─────────────────────────────────────────────────────────────
# STEP 3 — A dispatcher: maps tool name → function
#           This is what YOUR code uses to execute tools.
#           Claude never calls tools directly — you do.
# ─────────────────────────────────────────────────────────────

TOOLS = {
    "get_stock_price":        get_stock_price,
    "get_technical_indicators": get_technical_indicators,
}


# ─────────────────────────────────────────────────────────────
# STEP 4 — Describe the tools to Claude in JSON schema format
#           This is how Claude knows WHAT tools exist and
#           WHAT arguments to pass to them.
# ─────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "get_stock_price",
        "description": "Get the live stock price and daily % change for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker e.g. AAPL"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": "Get RSI and SMA20 technical indicators for a stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker e.g. AAPL"}
            },
            "required": ["ticker"],
        },
    },
]


# ─────────────────────────────────────────────────────────────
# STEP 5 — The system prompt
#           Tells Claude its role and the format we want back.
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a stock analyst assistant.
When asked about a stock:
1. Always fetch the live price first.
2. Then fetch the technical indicators.
3. Write a short 3-5 sentence analyst report covering price, trend, and RSI signal."""


# ─────────────────────────────────────────────────────────────
# STEP 6 — The ReAct loop
#           This is the agent. Read it line by line.
# ─────────────────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    print(f"\n{'='*50}")
    print(f"USER: {user_message}")
    print(f"{'='*50}")

    # The messages list is the agent's memory.
    # Every tool call and result gets appended here,
    # so Claude always has the full context.
    messages = [
        {"role": "user", "content": user_message}
    ]

    # Loop until Claude says it's done
    while True:

        # ── Ask Claude what to do next ──────────────────────
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        print(f"\n[Agent] stop_reason = '{response.stop_reason}'")

        # ── Claude is done — return the final answer ────────
        if response.stop_reason == "end_turn":
            final_answer = next(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            print(f"\n[Agent] DONE.\n")
            print(final_answer)
            return final_answer

        # ── Claude wants to use a tool ──────────────────────
        if response.stop_reason == "tool_use":

            # Add Claude's response (which contains the tool request) to history
            messages.append({"role": "assistant", "content": response.content})

            # Claude may request multiple tools at once — loop through them all
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue  # skip text blocks, only care about tool calls

                print(f"\n[Tool] Calling '{block.name}' with {block.input}")

                # Execute the tool — YOUR code runs it, not Claude
                result = TOOLS[block.name](**block.input)

                print(f"[Tool] Result: {result}")

                # Package the result to send back to Claude
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,       # must match the request id
                    "content":     json.dumps(result),  # must be a string
                })

            # Feed all the results back to Claude and loop again
            messages.append({"role": "user", "content": tool_results})


# ─────────────────────────────────────────────────────────────
# Run it
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent("Analyze AAPL for me")