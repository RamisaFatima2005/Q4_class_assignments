import os
from dotenv import load_dotenv
from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    RunContextWrapper,
    function_tool,
    input_guardrail,
    GuardrailFunctionOutput,
    set_tracing_disabled,
    enable_verbose_stdout_logging,
    set_tracing_disabled,
    output_guardrail
)
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

set_tracing_disabled(disabled=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")

# set_tracing_disabled(True)
# enable_verbose_stdout_logging()


provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# run_config = RunConfig(
#     model=model,
#     model_provider=provider,
#     tracing_disabled=True
# )

class Account(BaseModel):
    user_name: str
    user_id: int

class My_Output(BaseModel):
    name: str
    balance: str

class Guardrial_Output(BaseModel):
    is_bank_related: bool


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="""
    You are a bank query classifier.
    If the user query is about loans, bank accounts, credit cards, transactions, or other banking services,
    respond with {"is_bank_related": true}, otherwise {"is_bank_related": false}.
    """,
    output_type=Guardrial_Output,
    model=model
)


@input_guardrail
async def check_bank_related(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_agent,
        input,
        context=ctx.context
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered= not result.final_output.is_bank_related,
    )

class MessageOutput(BaseModel):
    response: str

class Guardrail_Output(BaseModel):
    is_bank_related: bool
    resoning : str

OutPutGuardrail = Agent(
    name="OutPut Guardrail",
    instructions="Check if the LLM response realted about bank.",
    model=model,
    output_type=Guardrail_Output
)

@output_guardrail
async def output_bank_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(OutPutGuardrail, output)
    
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_bank_related
    )

def check_user(ctx: RunContextWrapper[Account], agent=Agent) -> bool:
    if ctx.context.user_name == "Ramisa" and ctx.context.user_id == 1234:
        return True
    else:
        return False


@function_tool(is_enabled=check_user)
def check_balance(acc_number: str) -> str:
    """
    Check the balance of a bank account.
    """
    if acc_number != "123456789":
        return "Invalid account number."
    return "Your balance is $1000."

support_agent = Agent(
    name="Support Agent",
    instructions="You are a support agent. You help customers with their queries.",
    model=model
)

loan_inquiry_agent = Agent(
    name="Loan Inquiry Agent",
    instructions="You are a loan inquiry agent. You help customers with their loan inquiries.",
    model=model
)

bank_agent = Agent(
    name="Bank Agent",
    instructions="""
You are a bank agent. 
If the query is about loans → handoff to Loan Inquiry Agent.  
If the query is about opening an account, closing an account, debit/credit card issues, or general support → handoff to Support Agent directly without asking the user.
""",
    tools=[check_balance],
    handoffs=[support_agent, loan_inquiry_agent],
    input_guardrails=[check_bank_related],
    output_guardrails=[output_bank_guardrail],
    model=model
)

if __name__ == "__main__":
    try:
        user_context = Account(
            user_name=input("Enter your name: "),
            user_id=int(input("Enter your user ID: "))
        )

        user_input = input("Enter your query: ")

        result = Runner.run_sync(
            bank_agent,
            input=user_input,
            context=user_context,
            # run_config=run_config
        )
        print("\n🤖 Agent Response:", result.final_output)

    except InputGuardrailTripwireTriggered:
        print("⚠ Guardrail triggered — This query is not bank related.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")