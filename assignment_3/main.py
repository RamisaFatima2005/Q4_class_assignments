from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, WebSearchTool, FileSearchTool
import os
from dotenv import load_dotenv, find_dotenv
from openai.types.responses import ResponseTextDeltaEvent
import chainlit as cl

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=os.getenv["VECTOR_STORE_ID"],
        ),
    ],
)

result = Runner.run_sync(agent, "Which coffee shop should I go to, taking into account my preferences and the weather today in SF?")
print(result.final_output)