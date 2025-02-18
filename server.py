import os
import uvicorn
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from agent import create_agent

model = ChatOpenAI(model="gpt-4o")
agents = {} # Cache agents by element group ID
app = FastAPI()

def check_access(request: Request):
    authorization = request.headers.get("authorization")
    if not authorization:
        raise HTTPException(status_code=401)
    return authorization.replace("Bearer ", "")

class PromptPayload(BaseModel):
    element_group_id: str
    prompt: str

@app.post("/chatbot/prompt")
async def chatbot_prompt(payload: PromptPayload, access_token: str = Depends(check_access)) -> dict:
    cache_key = payload.element_group_id
    cache_folder = f"__cache__/{cache_key}"
    os.makedirs(cache_folder, exist_ok=True)
    if cache_key not in agents:
        agents[cache_key] = create_agent(model, payload.element_group_id, access_token)
    agent = agents[cache_key]
    config = {"configurable": {"thread_id": cache_key}}
    responses = []
    with open(f"{cache_folder}/logs.txt", "a") as log:
        log.write(f"[{datetime.now().isoformat()}] User: {payload.prompt}\n\n")
        async for step in agent.astream({"messages": [("human", payload.prompt)]}, config, stream_mode="updates"):
            log.write(f"[{datetime.now().isoformat()}] Assistant: {step}\n\n")
            if "agent" in step:
                for message in step["agent"]["messages"]:
                    if isinstance(message.content, str) and message.content:
                        responses.append(message.content)
    return { "responses": responses }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)