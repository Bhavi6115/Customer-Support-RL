from fastapi import FastAPI
from models import Action, Observation
import uvicorn
import random
import os

app = FastAPI()

# --- 1. Define Customer Intents (The Core Logic) ---
INTENTS = {
    "refund": {
        "query": "I need a refund for my order #12345",
        "sequence": ["/refund", "/verify_purchase", "/process_refund"],
    },
    "technical": {
        "query": "The app keeps crashing on my phone",
        "sequence": ["/troubleshoot", "/ask_os_version", "/escalate_to_tech"],
    },
    "order_status": {
        "query": "Where is my order? I ordered 2 weeks ago",
        "sequence": ["/track_order", "/check_shipping", "/offer_compensation"],
    },
}

# --- 2. Environment State (In-Memory Storage) ---
# These global variables hold the state of the current episode.
# In a production environment, you might use a more robust session management system.
current_intent = "refund"
correct_sequence = []
current_step = 0
current_query = ""
conversation_history = []


# --- 3. Core OpenEnv / Gymnasium-Style API Endpoints ---

@app.post("/reset")
async def reset():
    """Reset the environment to a new initial state, randomly picking a new customer intent."""
    global current_intent, correct_sequence, current_step, current_query, conversation_history

    # Randomly select a new intent for each episode
    current_intent = random.choice(list(INTENTS.keys()))
    correct_sequence = INTENTS[current_intent]["sequence"]
    current_query = INTENTS[current_intent]["query"]
    current_step = 0
    conversation_history = [f"Conversation started with intent: {current_intent}"]

    obs = Observation(
        query=current_query,
        history="\n".join(conversation_history),
        stage="awaiting_action",
    )
    # Return in the standard OpenEnv format
    return {"observation": obs.model_dump()}


@app.post("/step")
async def step(action: Action):
    """Execute an action, update environment state, and return the next observation, reward, and termination flag."""
    global current_step, conversation_history

    action_value = action.action_value
    reward = 0.0
    terminated = False
    status = ""

    # --- Reward Logic: Check if the action is correct for the current step ---
    if current_step < len(correct_sequence) and action_value == correct_sequence[current_step]:
        # The agent took the correct next step!
        current_step += 1
        reward = 1.0  # Small positive reward for making progress
        if current_step == len(correct_sequence):
            # The agent has completed the entire sequence for this intent
            reward = 10.0  # Large positive reward for successfully resolving the issue
            terminated = True
            status = f"Success! The {current_intent} issue has been resolved."
        else:
            status = f"Correct. Next expected action: {correct_sequence[current_step]}"
    elif action_value == "/escalate":
        # The agent chose to give up
        reward = -5.0
        terminated = True
        status = f"Agent escalated the {current_intent} issue to a human agent."
    else:
        # The agent took an invalid or out-of-sequence action
        reward = -2.0
        status = f"Invalid action '{action_value}' for {current_intent} issue."

    # --- Update Conversation History ---
    conversation_history.append(f"Agent: {action_value}")
    conversation_history.append(f"System: {status}")

    obs = Observation(
        query=current_query,
        history="\n".join(conversation_history),
        stage="resolved" if terminated else "in_progress",
    )
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "terminated": terminated,
        "truncated": False,  # We're not using a step limit for this environment
        "info": {"intent": current_intent, "step": current_step},
    }


# --- 4. OpenEnv-Specific Endpoints (Enhance Your Submission) ---

@app.get("/state")
async def get_state():
    """Return the full internal state of the environment, as required by the OpenEnv spec."""
    return {
        "intent": current_intent,
        "current_step": current_step,
        "correct_sequence": correct_sequence,
        "conversation_history": conversation_history,
        "is_resolved": current_step == len(correct_sequence),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration (required for OpenEnv)."""
    return {"status": "healthy"}


# --- 5. Server Entry Point ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)