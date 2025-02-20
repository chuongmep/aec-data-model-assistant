import os
import uvicorn
from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from agents import AECDataModelAgent

cache_dir = "__cache__"
app = FastAPI()
agents: Dict[str, AECDataModelAgent] = dict() # Cache agents by element group ID

def _check_access(request: Request):
    authorization = request.headers.get("authorization")
    if not authorization:
        raise HTTPException(status_code=401)
    return authorization.replace("Bearer ", "")

class PromptPayload(BaseModel):
    element_group_id: str
    prompt: str

@app.post("/chatbot/prompt")
async def chatbot_prompt(payload: PromptPayload, access_token: str = Depends(_check_access)) -> dict:
    id = payload.element_group_id
    cache_id_dir = os.path.join(cache_dir, id)
    os.makedirs(cache_id_dir, exist_ok=True)
    if id not in agents:
        agent = AECDataModelAgent()
        await agent.initialize(id, access_token, cache_id_dir)
        agents[id] = agent
    agent = agents[id]
    responses = await agent.prompt(payload.prompt)
    return { "responses": responses }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)