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

> list all my hubs and projects, names only

> what projects are in "some hub name"?

> what element groups are in this project?

> list all elements in "some element group"

## Development

- Clone the repository
- Initialize and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
- Install Python dependencies: `pip3 install -r requirements.txt`
- Set the following environment variables:
    - `APS_ACCESS_TOKEN` - your APS access token (with access to AEC Data Model API)
    - (optional) `OPENAI_API_KEY` - your OpenAI API key (if you want to use GPT-4o)
    - (optional) `ANTHROPIC_API_KEY` - your Anthropic API key (if you want to use Claude 3.5 Sonnet)
- To test the assistant using OpenAI GPT-4o, run `python3 test_gpt.py`
- To test the assistant using Anthropic Claude 3.5 Sonnet, run `python3 test_claude.py`

> Note: the app creates a log file (for example, `test_gpt_2025-02-11T15-57-37.log` or `test_claude_2025-02-12_13-29-40.log`) for every chat session, capturing the user input as well as all the tool calls and answers from the agent.