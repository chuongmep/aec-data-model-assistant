# AEC Data Model Assistant

Experimental chatbot for querying design data in [Autodesk Construction Cloud](https://construction.autodesk.com/) using custom [LangChain](https://www.langchain.com) agents and [Autodesk Platform Services](https://aps.autodesk.com) (specifically with [AEC Data Model API](https://aps.autodesk.com/autodesk-aec-data-model-api)).

![Thumbnail](thumbnail.png)

## How does it work?

The application implements a [LangGraph agent](https://python.langchain.com/docs/how_to/migrate_agent/) with a couple of custom tools/functions:

- Finding relevant property definitions for given input query (using [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/), cached locally)
- Executing a specific GraphQL query against the AEC Data Model API
- Processing a JSON response from GraphQL API using [jq](https://jqlang.org/)

The agent is also provided with a subset of the AEC Data Model GraphQL schema (see [agents/AECDM.graphql](./agents/AECDM.graphql)), and a simplified documentation of the AEC Data Model API concepts such as filtering or pagination (see [agents/SYSTEM_PROMPTS.md](./agents/SYSTEM_PROMPTS.md)).

## Usage

Login with your Autodesk credentials, select one of your design files in ACC, and try some of the prompts below:

> how many elements are there?

> what elements have volume larger than 3000?

> what are their external element IDs?

## Development

### Prerequisites

- [APS application](https://aps.autodesk.com/en/docs/oauth/v2/tutorials/create-app/) of the _Desktop, Mobile, Single-Page App_ type
- [OpenAI API key](https://platform.openai.com/docs/quickstart/create-and-export-an-api-key)
- [Python 3.x](https://www.python.org/downloads/)

### Setup

- Clone the repository
- Initialize and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
- Install Python dependencies: `pip install -r requirements.txt`
- Update [static/config.js](static/config.js) with your APS client ID and callback URL
- Set the following environment variables:
  - `OPENAI_API_KEY` - your OpenAI API key
- Run the dev server: `python server.py`
- Open http://localhost:8000 in the browser