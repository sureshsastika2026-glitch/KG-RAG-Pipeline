# 🕸️ Tamil Nadu Schemes KG-RAG Chatbot

An interactive **Knowledge Graph Retrieval-Augmented Generation (KG-RAG)** chatbot built with **Python**, **LangChain**, **Neo4j Aura**, and **Streamlit**. The application scrapes official Tamil Nadu Government scheme pages, extracts structured entities and relationships using an LLM, stores them in a Neo4j Aura graph database, and answers natural language questions by generating and executing Cypher queries in real time.

---

## 🏗️ KG-RAG Pipeline Architecture

```text
Document Loader ──▶ Entity Extraction ──▶ Relationship Mapping ──▶ Graph Storage ──▶ Graph Query
(WebBaseLoader)    (LLMGraphTransformer)   (Allowed Schema)        (Neo4j Aura)     (GraphCypherQAChain)
