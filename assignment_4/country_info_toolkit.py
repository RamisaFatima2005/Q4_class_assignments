from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, set_tracing_disabled
from openai import AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

get_country_capital = Agent(
    name="Get Country Capital Agent",
    instructions="""You are an expert at finding the capital city of a given country. When provided with a country name, 
        return only the name of the capital city as a single string (e.g., 'Islamabad' for Pakistan).
        Do not include additional text, explanations.""",
        model=model
)

get_country_language = Agent(
    name="Get Country Language Agent",
    instructions="""
        You are an expert at identifying the national language of a given country. When provided with a country name,
        return only the name of the primary national language as a single string (e.g., 'Urdu' for Pakistan). 
        Do not include additional text, explanations.
        """,
        model=model
)

get_country_population = Agent(
    name="Get Country Population Agent",
    instructions="""
    You are an expert at retrieving the population of a given country. When provided with a country name,
        return only the population as a number (e.g., '220 million' for Pakistan). 
        Do not include additional text, explanations.""",
        model=model
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        """You are a smart Country Info Toolkit. When given a country name
        important!
        To get the capital of country, use the tool get_country_capital
        To get the language of country, use the tool get_country_language
        To get the population of country, use the tool get_country_population
        
        """
    ),
    model=model,

    tools=[
        get_country_capital.as_tool(
            tool_name="get_country_capital",
            tool_description="Retrieves the capital city of the given country as a single string.",
        ),
        get_country_language.as_tool(
            tool_name="get_country_language",
            tool_description="Retrieves the primary national language of the given country as a single string.",
        ),
        get_country_population.as_tool(
            tool_name="get_country_population",
            tool_description="Retrieves the population of the given country as a number.",
        ),
    ],
)

user_prompt = input("Enter a country name to get its capital, language, and population: ")

async def main():
    result = await Runner.run(
        triage_agent,
        input=user_prompt,
    )
    print(result.final_output)


asyncio.run(main())
