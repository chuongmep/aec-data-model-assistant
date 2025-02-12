from datetime import datetime
from langchain_anthropic import ChatAnthropic
from agent import create_agent

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
        print()
        for step in agent.stream({"messages": [("human", query)]}, config, stream_mode="updates"):
            log.write(f"Assistant: {step}\n\n")
            if "agent" in step:
                for message in step["agent"]["messages"]:
                    if isinstance(message.content, str) and message.content:
                        print(message.content, end="\n\n")
        log.flush()