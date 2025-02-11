import os
import json
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

API_ENDPOINT = "https://developer.api.autodesk.com/aec/graphql"
ACCESS_TOKEN = os.getenv("APS_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError("APS_ACCESS_TOKEN environment variable is not set")

with open("schema/aecdm.json", "r") as file:
    schema = json.load(file)
    schema_types = schema["data"]["__schema"]["types"]

def get_type_schema(type_name: str) -> dict:
    return next((item for item in schema_types if item["name"] == type_name), None)

def execute_query(query: str) -> dict:
    transport = AIOHTTPTransport(url=API_ENDPOINT, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
    client = Client(transport=transport, fetch_schema_from_transport=True)
    return client.execute(gql(query))