import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# Load environment variables from .env
load_dotenv()

# Initialize LLM and Neo4j Aura connection
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
graph = Neo4jGraph(
    url=os.getenv("neo4j+s://e76a1f2d.databases.neo4j.io"),
    username=os.getenv("e76a1f2d"),
    password=os.getenv("iEq7eiBlN0JX8jgukmpkeVKx-a5vTV4PUYjQMI2cMNAexit"),
)

# ---------------------------------------------------------
# STAGE 1: DOCUMENT LOADER
# ---------------------------------------------------------
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
loader = WebBaseLoader(url)
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_docs)
print(f"Loaded and split into {len(documents)} chunks.")

# ---------------------------------------------------------
# STAGE 2 & 3: ENTITY EXTRACTION & RELATIONSHIP MAPPING
# ---------------------------------------------------------
# Constraining nodes/relationships keeps your government schemes graph clean
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Scheme", "Department", "Beneficiary", "Eligibility", "Benefit"],
    allowed_relationships=["OFFERED_BY", "TARGETS", "REQUIRES", "PROVIDES"],
)
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Extracted {len(graph_documents)} graph documents.")

# ---------------------------------------------------------
# STAGE 4: GRAPH STORAGE (Neo4j Aura)
# ---------------------------------------------------------
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True,
)
graph.refresh_schema()
print("Knowledge Graph stored in Neo4j Aura.")

# ---------------------------------------------------------
# STAGE 5: GRAPH QUERY (Chatbot Loop)
# ---------------------------------------------------------
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,               # Prints generated Cypher queries in terminal
    allow_dangerous_requests=True,
)

print("\nKG-RAG Chatbot ready! Type 'exit' to quit.")
while True:
    user_query = input("\nAsk about TN Schemes: ")
    if user_query.lower() == "exit":
        break
    response = chain.invoke({"query": user_query})
    print(f"Answer: {response['result']}")