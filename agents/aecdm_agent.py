import os
import json
import jq
import faiss
from datetime import datetime
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

with open(os.path.join(os.path.dirname(__file__), "SYSTEM_PROMPTS.md")) as f:
    SYSTEM_PROMPTS = f.read().replace("{", "{{").replace("}", "}}")
with open(os.path.join(os.path.dirname(__file__), "AECDM.graphql")) as f:
    AECDM_GRAPHQL = f.read().replace("{", "{{").replace("}", "}}")

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
INDEX_DIMENSIONS = 1536
AECDM_ENDPOINT = "https://developer.api.autodesk.com/aec/graphql"
MAX_RESPONSE_SIZE = (1 << 12)

class Agent:
    def __init__(self, llm: BaseChatModel, prompt_template: ChatPromptTemplate, tools: list[BaseTool], cache_urn_dir: str):
        self._agent = create_react_agent(llm, tools, prompt=prompt_template, checkpointer=MemorySaver())
        self._config = {"configurable": {"thread_id": os.path.basename(cache_urn_dir)}}
        self._logs_path = os.path.join(cache_urn_dir, "logs.txt")

    def _log(self, message: str):
        with open(self._logs_path, "a") as log:
            log.write(f"[{datetime.now().isoformat()}] {message}\n\n")

    async def prompt(self, prompt: str) -> list[str]:
        self._log(f"User: {prompt}")
        responses = []
        async for step in self._agent.astream({"messages": [("human", prompt)]}, config=self._config, stream_mode="updates"):
            if "agent" in step:
                for message in step["agent"]["messages"]:
                    self._log(message.pretty_repr())
                    if isinstance(message.content, str) and message.content:
                        responses.append(message.content)
            if "tools" in step:
                for message in step["tools"]["messages"]:
                    self._log(message.pretty_repr())
        return responses

async def _get_property_definitions(element_group_id: str, access_token: str, cache_dir: str) -> list[str]:
    props_cache_path = os.path.join(cache_dir, "props.json")
    if not os.path.exists(props_cache_path):
        transport = AIOHTTPTransport(url=AECDM_ENDPOINT, headers={"Authorization": f"Bearer {access_token}"})
        client = Client(transport=transport, fetch_schema_from_transport=True)
        query = gql("""
            query GetPropertyDefinitions($elementGroupId: ID!, $cursor:String) {
                elementGroupAtTip(elementGroupId:$elementGroupId) {
                    propertyDefinitions(pagination:{cursor:$cursor}) {
                        pagination {
                            cursor
                        }
                        results {
                            id
                            name
                            description
                            units {
                                id
                                name
                            }
                        }
                    }
                }
            }
        """)
        property_definitions = []
        response = await client.execute_async(query, variable_values={"elementGroupId": element_group_id})
        property_definitions.extend(response["elementGroupAtTip"]["propertyDefinitions"]["results"])
        while response["elementGroupAtTip"]["propertyDefinitions"]["pagination"]["cursor"]:
            cursor = response["elementGroupAtTip"]["propertyDefinitions"]["pagination"]["cursor"]
            response = await client.execute_async(query, variable_values={"elementGroupId": element_group_id, "cursor": cursor})
            property_definitions.extend(response["elementGroupAtTip"]["propertyDefinitions"]["results"])
        with open(props_cache_path, "w") as f:
            json.dump(property_definitions, f)
    with open(props_cache_path) as f:
        property_definitions = json.load(f)
    return property_definitions

async def _get_vector_store(element_group_id: str, access_token: str, cache_dir: str) -> VectorStore:
    index_cache_path = os.path.join(cache_dir, "faiss_index")
    if os.path.exists(index_cache_path):
        return FAISS.load_local(index_cache_path, _embeddings, allow_dangerous_deserialization=True)
    index = faiss.IndexFlatL2(INDEX_DIMENSIONS)
    vector_store = FAISS(
        embedding_function=_embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    property_definitions = await _get_property_definitions(element_group_id, access_token, cache_dir)
    documents = [
        Document(f"Property Name: {prop["name"]}\nID: {prop["id"]}\nDescription: {prop["description"]}\nUnits: {prop["units"]["name"] if prop["units"] and prop["units"]["name"] else ""}")
        for prop in property_definitions
    ]
    vector_store.add_documents(documents=documents)
    vector_store.save_local(index_cache_path)
    return vector_store

async def create_aecdm_agent(element_group_id: str, access_token: str, cache_dir: str):
    @tool
    async def execute_graphql_query(query: str) -> dict:
        """Executes the given GraphQL query in Autodesk AEC Data Model API, and returns the result as a JSON."""
        transport = AIOHTTPTransport(url=AECDM_ENDPOINT, headers={"Authorization": f"Bearer {access_token}"})
        client = Client(transport=transport, fetch_schema_from_transport=True)
        result = await client.execute_async(gql(query))
        # Limit the response size to avoid overwhelming the LLM
        if len(json.dumps(result)) > MAX_RESPONSE_SIZE:
            raise ValueError(f"Result is too large. Please refine your query.")
        return result

    @tool
    def execute_jq_query(query: str, input_json: str):
        """Processes the given JSON input with the given jq query, and returns the result as a JSON."""
        return jq.compile(query).input_text(input_json).all()

    vector_store = await _get_vector_store(element_group_id, access_token, cache_dir)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    retriever_tool = create_retriever_tool(retriever, "find_related_property_definitions", "Finds property definitions in the AEC Data Model API that are relevant to the input query.")

    llm = ChatOpenAI(model="gpt-4o")
    tools = [execute_graphql_query, execute_jq_query, retriever_tool]
    system_prompts = [
        SYSTEM_PROMPTS,
        AECDM_GRAPHQL,
        "Whenever referring to a list of elements, always include their external element ID.",
        f"Unless specified otherwise, the element group ID being discussed is `{element_group_id}`."
    ]
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompts), ("placeholder", "{messages}")])
    return Agent(llm, prompt_template, tools, cache_dir)