import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import random
import pickle
import numpy as np
from models import Action, Observation

# ---------- Load trained agent (if available) ----------
trained_q_table = None
actions = ["/refund", "/verify_purchase", "/process_refund", "/escalate", "/invalid"]

if os.path.exists("trained_q_table.pkl"):
    with open("trained_q_table.pkl", "rb") as f:
        trained_q_table = pickle.load(f)

def get_state_key(state):
    return f"{state['query'][:30]}_{state['stage']}_{state['history'][-50:]}"

def agent_choose_action(state):
    if trained_q_table is None:
        return "⚠️ No trained agent"
    key = get_state_key(state)
    if key in trained_q_table:
        best_action_idx = np.argmax(trained_q_table[key])
        return actions[best_action_idx]
    return actions[0]

# ---------- Environment state (same as before) ----------
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

current_intent = "refund"
correct_sequence = []
current_step = 0
current_query = ""
conversation_history = []

# ---------- FastAPI app ----------
app = FastAPI(title="Customer Support RL Environment")

# Enable CORS (optional, but helpful for cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- API Endpoints (required for OpenEnv) -----
@app.post("/reset")
async def reset():
    global current_intent, correct_sequence, current_step, current_query, conversation_history
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
    return {"observation": obs.model_dump()}

@app.post("/step")
async def step(action: Action):
    global current_step, conversation_history
    action_value = action.action_value
    reward = 0.0
    terminated = False
    status = ""
    if current_step < len(correct_sequence) and action_value == correct_sequence[current_step]:
        current_step += 1
        reward = 1.0
        if current_step == len(correct_sequence):
            reward = 10.0
            terminated = True
            status = f"✅ Success! The {current_intent} issue has been resolved."
        else:
            status = f"✔️ Correct. Next expected action: {correct_sequence[current_step]}"
    elif action_value == "/escalate":
        reward = -5.0
        terminated = True
        status = f"🔄 Agent escalated the {current_intent} issue to a human agent."
    else:
        reward = -2.0
        status = f"❌ Invalid action '{action_value}' for {current_intent} issue."
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
        "truncated": False,
        "info": {"intent": current_intent, "step": current_step},
    }

@app.get("/state")
async def get_state():
    return {
        "intent": current_intent,
        "current_step": current_step,
        "correct_sequence": correct_sequence,
        "conversation_history": conversation_history,
        "is_resolved": current_step == len(correct_sequence),
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ---------- Gradio Dashboard (calls the same API internally) ----------
def interact(action, auto_mode=False):
    """Send an action to the API and return pretty‑printed results."""
    with httpx.Client(base_url="http://localhost:7860", timeout=30.0) as client:
        if action == "Reset":
            resp = client.post("/reset", json={})
            state = resp.json()["observation"]
            return state["query"], "🔄 Reset complete. Ready.", "", ""
        else:
            resp = client.post("/step", json={"action_value": action})
            data = resp.json()
            obs = data["observation"]
            reward = data["reward"]
            terminated = data["terminated"]
            info = data.get("info", {})
            intent = info.get("intent", "unknown")
            # Pretty status line
            status = f"💰 Reward: {reward}  |  📍 Stage: {obs['stage']}  |  🎯 Intent: {intent}"
            next_action = ""
            if auto_mode and not terminated:
                state_dict = {
                    "query": obs["query"],
                    "stage": obs["stage"],
                    "history": obs["history"]
                }
                next_action = agent_choose_action(state_dict)
            return obs["query"], status, obs["history"], next_action

def reset_and_agent_run():
    """Run a full episode using the trained agent (or a fallback sequence)."""
    with httpx.Client(base_url="http://localhost:7860", timeout=30.0) as client:
        client.post("/reset", json={})
        total_reward = 0
        terminated = False
        step = 0
        history_log = []
        # Use trained agent if available; otherwise use a correct sequence for refund
        if trained_q_table is None:
            fallback_actions = ["/refund", "/verify_purchase", "/process_refund"]
        while not terminated and step < 10:
            if trained_q_table is not None:
                # Get current state from /state endpoint
                state_resp = client.get("/state")
                state_data = state_resp.json()
                obs_for_key = {
                    "query": state_data.get("current_query", ""),
                    "stage": "in_progress" if not state_data["is_resolved"] else "resolved",
                    "history": "\n".join(state_data["conversation_history"])
                }
                action = agent_choose_action(obs_for_key)
            else:
                if step < len(fallback_actions):
                    action = fallback_actions[step]
                else:
                    break
            step_resp = client.post("/step", json={"action_value": action})
            step_data = step_resp.json()
            reward = step_data["reward"]
            total_reward += reward
            terminated = step_data["terminated"]
            history_log.append(f"**Step {step+1}:** `{action}` → Reward {reward}")
            step += 1
        history_log.append(f"\n## 🎉 **Total Reward: {total_reward}**")
        return "\n".join(history_log)

# ---------- Build Gradio UI (pretty, clean, professional) ----------
with gr.Blocks(title="Customer Support RL Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Customer Support RL Environment")
    gr.Markdown("### Train an AI to resolve customer issues using Reinforcement Learning")
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Controls")
            action_dropdown = gr.Dropdown(
                choices=["/refund", "/verify_purchase", "/process_refund", "/escalate", "/invalid", "Reset"],
                label="Action",
                value="/refund"
            )
            with gr.Row():
                take_btn = gr.Button("🚀 Take Action", variant="primary")
                auto_checkbox = gr.Checkbox(label="🤖 Auto mode (agent suggests next)", value=False)
            auto_run_btn = gr.Button("⚡ Run Full Episode (trained agent)")
            gr.Markdown("---")
            gr.Markdown("### 📊 Status")
            output_query = gr.Textbox(label="💬 Customer Query", lines=2, interactive=False)
            output_reward = gr.Textbox(label="🏆 Current State", interactive=False)
            output_next_action = gr.Textbox(label="🔮 Agent Suggestion", lines=1, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### 📜 Conversation Log")
            output_history = gr.Textbox(label="", lines=12, interactive=False)
            auto_output = gr.Textbox(label="🤖 Agent Episode Result", lines=10, interactive=False)
    # Connect buttons
    take_btn.click(interact, inputs=[action_dropdown, auto_checkbox],
                  outputs=[output_query, output_reward, output_history, output_next_action])
    auto_run_btn.click(reset_and_agent_run, inputs=[], outputs=[auto_output])
    demo.load(lambda: interact("Reset", False),
              outputs=[output_query, output_reward, output_history, output_next_action])

# ---------- Mount Gradio at root (overwrites the root path) ----------
app = gr.mount_gradio_app(app, demo, path="/")

# ---------- Run the server ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
