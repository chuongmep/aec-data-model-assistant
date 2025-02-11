import os
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

API_ENDPOINT = "https://developer.api.autodesk.com/aec/graphql"
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
ACCESS_TOKEN = os.getenv("APS_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError("APS_ACCESS_TOKEN environment variable is not set")

def execute_query(query: str) -> dict:
    # Create a separate client because tools can be used in parallel
    transport = AIOHTTPTransport(url=API_ENDPOINT, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
    client = Client(transport=transport, fetch_schema_from_transport=True)
    return client.execute(gql(query))

schema = execute_query(INTROSPECTION_QUERY)
types = schema["__schema"]["types"]

def get_type_schema(type_name: str) -> dict:
    return next((t for t in types if t["name"] == type_name), None)