import jq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

ENDPOINT = "https://developer.api.autodesk.com/aec/graphql"

def create_agent(model, element_group_id, access_token):
    @tool
    async def execute_graphql_query(query: str) -> dict:
        """Executes the given GraphQL query in Autodesk AEC Data Model API, and returns the result as a JSON."""
        transport = AIOHTTPTransport(url=ENDPOINT, headers={"Authorization": f"Bearer {access_token}"})
        client = Client(transport=transport, fetch_schema_from_transport=True)
        return await client.execute_async(gql(query))

    @tool
    def execute_jq_query(query: str, input_json: str):
        """Processes the given JSON input with the given jq query, and returns the result as a JSON."""
        return jq.compile(query).input_text(input_json).all()

    # Create the agent
    tools = [execute_graphql_query, execute_jq_query]
    with open("AECDM.graphql", "r") as f:
        schema = f.read()
    system_prompt = f"""
        You are a helpful assistant answering questions about data in AEC Data Model using the GraphQL schema below:

        {schema.replace("{", "{{").replace("}", "}}")}

        When processing paginated responses, use the `cursor` field of the `Pagination` type to navigate through additional pages.
        When filtering responses using the `query` field, use **RSQL** syntax such as `'property.name.Element Name'==NameOfElement`.
        To find all the property IDs and names to filter by, use the `propertyDefinitions` field of the `ElementGroup` type.
        Process JSON responses from the GraphQL API with **jq** queries to extract the relevant information.
        Unless specified otherwise, the element group ID being discussed is `{element_group_id}`.
    """
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")])
    memory = MemorySaver()
    return create_react_agent(model, tools, prompt=prompt_template, checkpointer=memory)