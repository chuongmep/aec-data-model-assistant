import jq
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from aecdm import get_type_schema, execute_query

OPENAI_MODEL = "gpt-4o"

# Setup the tools for the assistant

@tool
def list_graphql_queries() -> list:
    """Returns all top-level GraphQL queries in Autodesk AEC Data Model API."""
    query_object = get_type_schema("Query")
    if query_object:
        return query_object["fields"]
    return []

@tool
def get_graphql_type(type_name: str) -> dict:
    """Returns details about given GraphQL type in Autodesk AEC Data Model API."""
    return get_type_schema(type_name)

@tool
def execute_graphql_query(graphql_query: str) -> dict:
    """Executes the given GraphQL query in Autodesk AEC Data Model API, and returns the result as a JSON."""
    return execute_query(graphql_query)

@tool
def execute_jq_query(jq_query: str, input_json: str):
    """Processes the given JSON input with the given jq query, and returns the result as a JSON."""
    return jq.compile(jq_query).input_text(input_json).all()

# Setup the assistant

model = ChatOpenAI(model=OPENAI_MODEL)
tools = [list_graphql_queries, get_graphql_type, execute_graphql_query, execute_jq_query]
system_prompt = " ".join([
    "You are a helpful assistant answering questions about user's data in AEC Data Model using its GraphQL API.",
    "Where possible, process JSON responses from the GraphQL API with jq queries to extract the relevant information.",
])
prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")])
memory = MemorySaver()
agent = create_react_agent(model, tools, prompt=prompt_template, checkpointer=memory)
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
                answer = step["agent"]["messages"][-1].content
                if answer:
                    print(f"{answer}\n\n")
        log.flush()