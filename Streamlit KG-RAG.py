import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, LLMGraphTransformer

load_dotenv()

st.set_page_config(page_title="TN Schemes KG-RAG Chatbot", page_icon="🕸️")
st.title("🕸️ Tamil Nadu Schemes KG-RAG Chatbot")

# Cache LLM, Neo4j connection, and QA chain across reruns
@st.cache_resource
def init_kg_chain():
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    graph = Neo4jGraph(
        url=os.getenv("neo4j+s://e76a1f2d.databases.neo4j.io"),
        username=os.getenv("e76a1f2d"),
        password=os.getenv("iEq7eiBlN0JX8jgukmpkeVKx-a5vTV4PUYjQMI2cMNA"),
    )
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )
    return llm, graph, chain

llm, graph, chain = init_kg_chain()

# ---------------------------------------------------------
# SIDEBAR: ON-DEMAND INGESTION (STAGES 1–4)
# ---------------------------------------------------------
with st.sidebar:
    st.header("Knowledge Graph Setup")
    url = st.text_input(
        "Schemes URL",
        value="https://www.tn.gov.in/scheme_list.php?dep_id=Mg==",
    )
    if st.button("Ingest / Refresh Graph"):
        with st.status("Running KG-RAG ingestion pipeline...", expanded=True) as status:
            st.write("1. Loading and chunking webpage...")
            raw_docs = WebBaseLoader(url).load()
            documents = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            ).split_documents(raw_docs)

            st.write(f"2. Extracting entities & relationships from {len(documents)} chunks...")
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=["Scheme", "Department", "Beneficiary", "Eligibility", "Benefit"],
                allowed_relationships=["OFFERED_BY", "TARGETS", "REQUIRES", "PROVIDES"],
            )
            graph_docs = transformer.convert_to_graph_documents(documents)

            st.write("3. Storing Knowledge Graph in Neo4j Aura...")
            graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
            graph.refresh_schema()
            status.update(label="Graph updated in Neo4j Aura!", state="complete")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# MAIN CHAT INTERFACE (STAGE 5: GRAPH QUERY)
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("cypher"):
            with st.expander("View Generated Cypher Query"):
                st.code(msg["cypher"], language="cypher")

# Handle new user input
if prompt := st.chat_input("Ask about Tamil Nadu Government schemes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Querying Neo4j Knowledge Graph..."):
            response = chain.invoke({"query": prompt})
            answer = response["result"]
            cypher_query = response["intermediate_steps"][0].get("query", "")

        st.markdown(answer)
        if cypher_query:
            with st.expander("View Generated Cypher Query"):
                st.code(cypher_query, language="cypher")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "cypher": cypher_query}
    )