from datetime import datetime
from langchain_openai import ChatOpenAI
from agent import create_agent, extract_response

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(model)
config = {"configurable": {"thread_id": "test-thread"}}
log_filename = datetime.now().strftime("test_gpt_%Y-%m-%dT%H-%M-%S.log")
with open(log_filename, "a") as log:
    while True:
        query = input("Enter your query (or press Enter to exit): ")
        if not query:
            break
        log.write(f"User: {query}\n\n")
        for step in agent.stream({"messages": [("human", query)]}, config, stream_mode="updates"):
            log.write(f"Assistant: {step}\n\n")
            for message in extract_response(step):
                print(message, end="\n\n")
        log.flush()