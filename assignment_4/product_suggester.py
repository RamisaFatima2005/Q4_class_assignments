from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv, find_dotenv

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
    name="Product Suggester",
    instructions="You are a product suggester AI. Based on the user's input suggest them gift ideas.Don't give lengthy responses.",
)

occasion     = input("Enter Occasion... ")           
gender    = input("Enter Gender... ")             
relation = input("Enter relation... ")        
budget       = input("Tell me your Budget... ")  
age = input("Enter Age... ")         

input = (
    f"Occasion: {occasion}\n"
    f"Recipient: {gender}\n"
    f"Relationship: {relation}\n"
    f"Budget: {budget}\n"
    f"Age: {age}\n"
    "Give 3 gift suggestion to the user based on the information user give you about the above things."
)

result = Runner.run_sync(
    agent, 
    input=input,
    run_config=run_config,
)

print(result.final_output)
