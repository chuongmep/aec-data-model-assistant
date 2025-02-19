import os
import jq
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

_llm = ChatOpenAI(model="gpt-4o")
_endpoint = "https://developer.api.autodesk.com/aec/graphql"

def _load_content(relative_path: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), relative_path)) as f:
        return f.read()

class AECDataModelAgent:
    def __init__(self, element_group_id: str, access_token: str, cache_dir: str):
        @tool
        async def execute_graphql_query(query: str) -> dict:
            """Executes the given GraphQL query in Autodesk AEC Data Model API, and returns the result as a JSON."""
            transport = AIOHTTPTransport(url=_endpoint, headers={"Authorization": f"Bearer {access_token}"})
            client = Client(transport=transport, fetch_schema_from_transport=True)
            return await client.execute_async(gql(query))

        @tool
        def execute_jq_query(query: str, input_json: str):
            """Processes the given JSON input with the given jq query, and returns the result as a JSON."""
            return jq.compile(query).input_text(input_json).all()

        tools = [execute_graphql_query, execute_jq_query]
        system_prompts = [
            _load_content("SYSTEM_PROMPTS.md").replace("{", "{{").replace("}", "}}"),
            _load_content("AECDM.graphql").replace("{", "{{").replace("}", "}}"),
            f"Unless specified otherwise, the element group ID being discussed is `{element_group_id}`."
        ]
        prompt_template = ChatPromptTemplate.from_messages([("system", system_prompts), ("placeholder", "{messages}")])
        self._agent = create_react_agent(_llm, tools, prompt=prompt_template, checkpointer=MemorySaver())
        self._config = {"configurable": {"thread_id": element_group_id}}
        self._logs_path = os.path.join(cache_dir, "logs.txt")

    def _log(self, message: str):
        with open(self._logs_path, "a") as log:
            log.write(f"[{datetime.now().isoformat()}] {message}\n\n")

    async def prompt(self, prompt: str) -> list[str]:
        self._log(f"User: {prompt}")
        responses = []
        async for step in self._agent.astream({"messages": [("human", prompt)]}, self._config, stream_mode="updates"):
            self._log(f"Assistant: {step}")
            if "agent" in step:
                for message in step["agent"]["messages"]:
                    if isinstance(message.content, str) and message.content:
                        responses.append(message.content)
        return responses