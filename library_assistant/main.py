from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    ModelSettings,
    function_tool,
    input_guardrail,
    set_tracing_disabled,
    GuardrailFunctionOutput,
    TResponseInputItem,
)
from pydantic import BaseModel
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# User Context Model
class UserContext(BaseModel):
    name: str
    member_id: str

# Library Guardrail Agent Output
class LibraryAgentOutput(BaseModel):
    is_related_to_library: bool
    reasoning: str

# Book Database
book_db = {"1984": 3, "War And Peace": 0, "Harry Potter": 5, "Bang-e-Dara": 2, "Bal-e-Jibril":1, "Tahzib-ul-Akhlaq":4}

# Guardrail Agent to block non-library queries
guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="You are a strict guardrail agent. Only allow library-related questions (books, availability, books for borrow, timings of library, or listing all books). If user question is not related to library, respond with 'is_not_related_to_library: True' else respond with 'is_not_related_to_library: False' also provide reasoning.",
    model=model,
    output_type=LibraryAgentOutput,
)

# Input Guardrail Function
@input_guardrail
async def guardrail_check(
    ctx: RunContextWrapper[None], agent: Agent, input: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered= not result.final_output.is_related_to_library,
    )

# Function Tools
@function_tool
def search_book(title: str) -> str:
    """Search if a book exists in the library."""
    for book in book_db:
        if book.lower() == title.lower():
            return "Book found"
    return "Book not found"

@function_tool
def list_all_books() -> str:
    """List all books available in the library."""
    if not book_db:
        return "No books available in the library."
    return "Available books: " + ", ".join(f"{book} ({book_db[book]} copies)" for book in book_db)

def is_member(ctx: RunContextWrapper[UserContext], agent: Agent) -> bool:
    if ctx.context.member_id:
        return True
    return False

@function_tool(is_enabled=is_member)
def check_availability(title: str) -> str:
    """Check how many copies are available (only for members)."""
    for book in book_db:
        if book.lower() == title.lower():
            return f"Copies available: {book_db[book]}"
    return "Book not found"

@function_tool
def library_timings() -> str:
    """Get library timings."""
    return "Library is open from 9 AM to 6 PM, Monday to Thursday and 3 PM to 9 PM on Friday and Saturday."

# Dynamic Instructions
def personalize_instructions(ctx: RunContextWrapper[UserContext], agent: Agent[UserContext]) -> str:
    base_instruction = f"You are a helpful library assistant. Always greet the user by their name {ctx.context.name}. "
    if not ctx.context.member_id:
        base_instruction += "If the user asks about book availability, inform them that only registered members can check availability and suggest they provide a member ID or use other tools like search_book or list_all_books."
    else:
        base_instruction += "You can assist with checking book availability since the user is a registered member."
    return base_instruction

# Library Assistant Agent
library_agent = Agent(
    name="Library Assistant",
    instructions=personalize_instructions,
    model=model,
    tools=[search_book, check_availability, library_timings, list_all_books],
    input_guardrails=[guardrail_check],
    model_settings=ModelSettings(
        temperature=0
    ),
)

if __name__ == "__main__":
    name = input("Enter your name: ")
    member_id = input("Add your member id (if you do not have member id then leave this blank): ")
    user = UserContext(name=name, member_id=member_id)

    print("\nLibrary Assistant is ready! Ask me anything about the library. Type 'exit' to exit.\n")

    while True:
        q = input("Your query: ")
        if q.lower() == "exit":
            print("Bye! Session end.")
            break
        result = Runner.run_sync(library_agent, q, context=user)
        print(f"{result.final_output}\n")