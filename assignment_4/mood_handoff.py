# mood_handoff.py
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv, find_dotenv

# ─── ❶ Setup ─────────────────────────────
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# ─── ❷ Single Agent for Mood Detection ──
mood_detector = Agent(
    name="Mood Detector",
    instructions=(
        "You are a mood-detection AI. "
        "Given the user's message, reply with ONE WORD describing their mood "
        "(like: happy, excited, calm, sad, stressed, anxious, angry, bored, tired, depressed). "
        "Reply with the mood only — no extra text."
    ),
)

# ─── ❸ Mood-to-Activity Mapping ─────────
mood_suggestions = {
    "sad": "Take a moment to write down three things you're truly grateful for. It can help shift your focus toward the positive.",
    "stressed": "Try the 4-7-8 deep breathing technique: Inhale for 4 seconds, hold for 7, and exhale for 8. Repeat a few times to relax.",
    "anxious": "Do a quick grounding exercise: Look around and name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
    "angry": "Step away from the situation and take a brisk 5-minute walk. Moving your body can help release tension.",
    "tired": "Close your eyes, take a few deep breaths, and rest for 5 minutes. A short mental reset can boost your energy.",
    "depressed": "Listen to soft, calming instrumental music. Let it soothe your mind and bring you a sense of peace.",
    "bored": "Take 5 minutes to stretch gently or sketch something random — creativity and movement can refresh your mood.",
    "tired": " Close your eyes, breathe deeply, and rest for 5 minutes. A quick reset works wonders."
}

# ─── ❹ User Input + Agent Run ───────────
user_message = input("How are you feeling today? ➜ ")

mood_result = Runner.run_sync(
    mood_detector,
    input=user_message,
    run_config=run_config,
)

mood = mood_result.final_output.lower().strip()
print(f"\nMood detected → {mood}")

# ─── ❺ Show Suggestion if Mood is Matched ─
if mood in mood_suggestions:
    print("Suggested activity:", mood_suggestions[mood])
else:
    print("No activity suggestion needed. Have a great day! 😊")