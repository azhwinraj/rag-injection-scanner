# rag-injection-scanner
A CLI tool and Python library that scans documents for embedded prompt injection payloads before they enter a RAG vector store. Three-layer detection (regex → heuristics → LLM judge), risk scoring, and JSON report output. Stops RAG poisoning at the ingestion gate.
