import jq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

INTROSPECTION_QUERY = """
{
  __schema {
    types {
      kind
      name
      description
      fields(includeDeprecated: false) {
        name
        description
        args {
          name
          description
          type {
            kind
            name
            ofType {
              kind
              name
            }
          }
        }
        type {
          kind
          name
          ofType {
            kind
            name
          }
        }
      }
    }
  }
}
"""

def create_graphql_agent(model, url, headers):
    """
    Creates an agent with custom tools for interacting with GraphQL API.
    Args:
        model (BaseChatModel): The model to be used by the agent.
        url (str): The GraphQL endpoint.
        headers (dict): The headers to be used for the GraphQL requests.
    Returns:
        An agent configured with tools for listing GraphQL queries, retrieving GraphQL type schemas,
        executing GraphQL queries, and processing JSON responses with jq queries.
    """

    def _query(query: str) -> dict:
        # Create a separate client because tools can be used in parallel
        transport = AIOHTTPTransport(url=url, headers=headers)
        client = Client(transport=transport, fetch_schema_from_transport=True)
        return client.execute(gql(query))

    # Get the introspection schema from the GraphQL API
    schema = _query(INTROSPECTION_QUERY) # TODO: maybe cache this?
    types = schema["__schema"]["types"]

    # Define tools for the agent
    @tool
    def list_graphql_queries() -> list:
        """Returns the name, description, and return type of all top-level GraphQL queries in Autodesk AEC Data Model API."""
        query_type = next((t for t in types if t["name"] == "Query"), None)
        return [{"name":field["name"], "description":field["description"], "type":field["type"]} for field in query_type["fields"]]

    @tool
    def get_graphql_type(type_name: str) -> dict | None:
        """Returns the introspection JSON schema for the given GraphQL type in Autodesk AEC Data Model API."""
        type = next((t for t in types if t["name"] == type_name), None)
        return type

    @tool
    def execute_graphql_query(graphql_query: str) -> dict:
        """Executes the given GraphQL query in Autodesk AEC Data Model API, and returns the result as a JSON."""
        return _query(graphql_query)

    @tool
    def execute_jq_query(jq_query: str, input_json: str):
        """Processes the given JSON input with the given jq query, and returns the result as a JSON."""
        return jq.compile(jq_query).input_text(input_json).all()

    # Create the agent
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