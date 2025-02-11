import os
import json
import jq
from datetime import datetime
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

API_ENDPOINT = "https://developer.api.autodesk.com/aec/graphql"
ACCESS_TOKEN = os.getenv("APS_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError("APS_ACCESS_TOKEN environment variable is not set")

# Read the AEC Data Model GraphQL schema from a local file (TODO: fetch it from the API)

with open("schema/aecdm.json", "r") as file:
    schema = json.load(file)
schema_types = schema["data"]["__schema"]["types"]

def find_graphql_type(type_name: str) -> dict:
    return next((item for item in schema_types if item["name"] == type_name), None)

# Setup the tools for the assistant

@tool
def list_graphql_queries() -> list[tuple[str, str]]:
    """Returns all top-level GraphQL queries in Autodesk AEC Data Model API."""
    query_object = find_graphql_type("Query")
    if query_object:
        return query_object["fields"]
    return []

@tool
def get_graphql_type(type_name: str) -> dict:
    """Returns details about given GraphQL type in Autodesk AEC Data Model API."""
    return find_graphql_type(type_name)

@tool
def execute_graphql_query(query: str):
    """Executes the given GraphQL query in Autodesk AEC Data Model API, and returns the result."""
    transport = AIOHTTPTransport(url=API_ENDPOINT, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
    client = Client(transport=transport, fetch_schema_from_transport=True)
    return client.execute(gql(query))

@tool
def execute_jq_query(query: str, json: str):
    """Executes the given jq query on a stringified JSON."""
    return jq.compile(query).input_text(json).all()

# Setup the assistant

model = ChatOpenAI(model="gpt-4o")
tools = [list_graphql_queries, get_graphql_type, execute_graphql_query, execute_jq_query]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant answering questions about user's data in AEC Data Model, using its GraphQL API, and using jq to process the JSON responses."),
        ("placeholder", "{messages}"),
    ]
)
memory = MemorySaver()
agent = create_react_agent(model, tools, prompt=prompt, checkpointer=memory)
config = {"configurable": {"thread_id": "test-thread"}}

# Start the chat loop

log_filename = datetime.now().strftime("chat_%Y-%m-%d_%H-%M-%S.log")
with open(log_filename, "a") as log:
    while True:
        query = input("Enter your query (or press Enter to exit): ")
        if not query:
            break
        log.write(f"User: {query}\n\n")
        for step in agent.stream({"messages": [("human", query)]}, config, stream_mode="updates"):
            log.write(f"Assistant: {step}\n\n")
            if "agent" in step:
                print(step["agent"]["messages"][-1].content)
                print()
        log.flush()