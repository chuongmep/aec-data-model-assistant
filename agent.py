import jq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from aecdm import get_type_schema, execute_query

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

def create_agent(model):
    tools = [list_graphql_queries, get_graphql_type, execute_graphql_query, execute_jq_query]
    system_prompt = " ".join([
        "You are a helpful assistant answering questions about data in AEC Data Model using its GraphQL API.",
        "When processing paginated responses, use the `cursor` field of the `Pagination` type to navigate through additional pages.",
        "When filtering responses using the `query` field, use **RSQL** syntax such as `'property.name.Element Name'==NameOfElement`.",
        "To find all the property IDs and names to filter by, use the `propertyDefinitions` field of the `ElementGroup` type.",
        "Process JSON responses from the GraphQL API with **jq** queries to extract the relevant information.",
    ])
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")])
    memory = MemorySaver()
    return create_react_agent(model, tools, prompt=prompt_template, checkpointer=memory)

def extract_response(step):
    results = []
    if "agent" in step:
        for message in step["agent"]["messages"]:
            if isinstance(message.content, str) and message.content:
                results.append(message.content)
            elif isinstance(message.content, list):
                for entry in message.content:
                    if isinstance(entry, str) and entry:
                        results.append(entry)
                    elif isinstance(entry, dict) and "text" in entry:
                        results.append(entry["text"])
    elif "tools" in step:
        pass
    return results