# aec-data-model-assistant

Experimental AI assistant for [Autodesk AEC Data Model API](https://aps.autodesk.com/autodesk-aec-data-model-api), built using [LangChain](https://www.langchain.com/) agents.

## How does it work?

The application implements a [LangGraph agent](https://python.langchain.com/docs/how_to/migrate_agent/) with a couple of custom tools/functions:

- Getting the list of all GraphQL queries available in the AEC Data Model API
- Getting the details about a specific GraphQL type in the AEC Data Model API
- Executing a specific GraphQL query against the AEC Data Model API
- Processing a JSON response from GraphQL API using [jq](https://jqlang.org/)

This way, we don't have to feed the complete (and very large) GraphQL schema to the context. The agent will pull in just the parts of the schema it needs based on the current task, generate a GraphQL query, execute it against the AEC Data Model API, and if needed, extract relevant information out of the JSON response using jq.

## Usage

The app runs in a loop as long as the user provides questions. Try some of these:

> list all hubs I have access to

> what projects are in "[some-hub-name]"?

> what element groups are in this project?

> list all elements in "[some-design-file]"

## Development

- Clone the repository
- Initialize and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
- Install Python dependencies: `pip3 install -r requirements.txt`
- Set the following environment variables:
    - `OPENAI_API_KEY` - your OpenAI API key
    - `APS_ACCESS_TOKEN` - your APS access token (with access to AEC Data Model API)
- Run the app: `python3 main.py`

> Note: the app creates a log file (for example, `chat_2025-02-11_15-57-37.log`) for every chat session, capturing the user input as well as all the tool calls and answers from the agent.