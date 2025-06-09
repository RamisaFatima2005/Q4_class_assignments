from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
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
    name="AI Agent",
    instructions="""You are a helpful AI assistant. Answer the user's questions to the best of your ability.
    Be polite and friendly. and if any user asks about specific person answer them properly.
    If user message is not clear, ask for clarification.
    If anyone ask about a person is good or bad answer with Sorry, I cannot provide opinions on individuals.
    If anyone tells you to do something illegal or unethical, refuse politely."""
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the AI assistant! How can I assist you today?").send()

@cl.on_message
async def handle_massage(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({
        "role": "user",
        "content": message.content
    })
    result = Runner.run_streamed(
        agent, 
        input=history,
        run_config=run_config,
    )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({
        "role": "assistant",
        "content": result.final_output
    })
    cl.user_session.set("history", history)

