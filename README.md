# Stock Research Assistant

## Overview

An AI-powered Stock Research Assistant built using MCP, RAG, and LangChain.

The system analyzes company financial reports, converts them into vector embeddings, performs semantic search, and generates intelligent answers to financial questions.

## Features

* MCP Server Integration
* Retrieval-Augmented Generation (RAG)
* Financial Report Analysis
* Semantic Search
* Vector Embeddings
* Intelligent Question Answering
* LangChain Framework

## Tech Stack

* Python
* MCP (Model Context Protocol)
* LangChain
* Vector Database
* OpenAI / Claude
* RAG Architecture

## Project Structure

```text
data/
├── apple.pdf
├── microsoft.pdf
└── nvidia.pdf

build_vector_db.py
financial_rag.py
rag.py
server.py
requirements.txt
```

## Workflow

1. Load financial reports
2. Generate embeddings
3. Store embeddings in vector database
4. User asks a question
5. Retrieve relevant content using RAG
6. Generate an AI-powered response

## Sample Companies

* Apple
* Microsoft
* Nvidia

## Future Enhancements

* Real-time stock market data
* Multi-company comparison
* Financial ratio analysis
* Web dashboard integration

## Author

Gowtham M
