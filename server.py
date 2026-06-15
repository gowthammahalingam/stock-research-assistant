from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import yfinance as yf

mcp = FastMCP("Stock Research Assistant")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "financial_vector_db")

db = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
llm = OllamaLLM(model="llama3")


@mcp.tool()
def ask_company_report(question: str) -> str:

    docs = db.similarity_search(question, k=3)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer only from the annual reports.

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt)


@mcp.tool()
def get_stock_price(symbol: str) -> str:

    stock = yf.Ticker(symbol)

    price = stock.info.get("currentPrice")

    return f"{symbol} Current Price: {price}"


@mcp.tool()
def company_info(symbol: str) -> str:

    stock = yf.Ticker(symbol)

    info = stock.info

    return (
        f"Company: {info.get('longName')}\n"
        f"Sector: {info.get('sector')}\n"
        f"Market Cap: {info.get('marketCap')}\n"
        f"PE Ratio: {info.get('trailingPE')}"
    )


@mcp.tool()
def compare_companies(symbol1: str, symbol2: str) -> str:

    stock1 = yf.Ticker(symbol1)
    stock2 = yf.Ticker(symbol2)

    info1 = stock1.info
    info2 = stock2.info

    return (
        f"{symbol1}\n"
        f"Market Cap: {info1.get('marketCap')}\n"
        f"PE Ratio: {info1.get('trailingPE')}\n\n"
        f"{symbol2}\n"
        f"Market Cap: {info2.get('marketCap')}\n"
        f"PE Ratio: {info2.get('trailingPE')}"
    )


@mcp.tool()
def generate_investment_report(company: str) -> str:

    docs = db.similarity_search(company, k=5)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Analyze the company using the annual report.

Company:
{company}

Context:
{context}

Provide:

1. Business Summary
2. Growth Opportunities
3. Risks
4. Key Takeaways
"""

    return llm.invoke(prompt)


if __name__ == "__main__":
    mcp.run()