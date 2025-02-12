from datetime import datetime
from langchain_anthropic import ChatAnthropic
from agent import create_agent, extract_response

model = ChatAnthropic(model="claude-3-5-sonnet-latest") # "claude-3-opus-latest" is *very* expensive!
agent = create_agent(model)
config = {"configurable": {"thread_id": "test-thread"}}
log_filename = datetime.now().strftime("test_claude_%Y-%m-%dT%H-%M-%S.log")
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