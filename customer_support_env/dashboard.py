import gradio as gr
import httpx
import pickle
import numpy as np
import os

# ---------- Load Trained Agent ----------
trained_q_table = None
actions = ["/refund", "/verify_purchase", "/process_refund", "/escalate", "/invalid"]

if os.path.exists("trained_q_table.pkl"):
    with open("trained_q_table.pkl", "rb") as f:
        trained_q_table = pickle.load(f)
    print("Loaded trained Q-table. Agent mode available.")
else:
    print("No trained Q-table found. Run train_agent.py first for agent mode.")

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

# ---------- Synchronous Interaction Functions ----------
def interact(action, auto_mode=False):
    with httpx.Client(timeout=30.0) as client:
        if action == "Reset":
            resp = client.post("http://localhost:8000/reset", json={})
            state = resp.json()["observation"]
            return (
                state["query"],
                "Reset complete. Ready.",
                "",
                ""
            )
        else:
            resp = client.post("http://localhost:8000/step", json={"action_value": action})
            data = resp.json()
            obs = data["observation"]
            reward = data["reward"]
            terminated = data["terminated"]
            info = data.get("info", {})
            intent = info.get("intent", "unknown")
            
            status = f"Reward: {reward}  |  Stage: {obs['stage']}  |  Intent: {intent}"
            next_action = ""
            if auto_mode and not terminated:
                next_action = agent_choose_action(obs)
            return (
                obs["query"],
                status,
                obs["history"],
                next_action
            )

def reset_and_agent_run():
    with httpx.Client(timeout=30.0) as client:
        client.post("http://localhost:8000/reset", json={})
        total_reward = 0
        terminated = False
        step = 0
        history_log = []
        while not terminated and step < 10:
            resp_state = client.get("http://localhost:8000/state")
            state_data = resp_state.json()
            obs_for_key = {
                "query": state_data.get("current_query", ""),
                "stage": "in_progress" if not state_data["is_resolved"] else "resolved",
                "history": "\n".join(state_data["conversation_history"])
            }
            action = agent_choose_action(obs_for_key)
            step_resp = client.post("http://localhost:8000/step", json={"action_value": action})
            step_data = step_resp.json()
            reward = step_data["reward"]
            total_reward += reward
            terminated = step_data["terminated"]
            history_log.append(f"Step {step+1}: {action} → Reward {reward}")
            step += 1
        history_log.append(f"\nTotal Reward: {total_reward}")
        return "\n".join(history_log)

# ---------- Professional Light Theme CSS ----------
custom_css = """
body {
    background: #f5f7fa;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 2rem auto !important;
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    padding: 2rem !important;
}
h1 {
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.5rem;
}
h3 {
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    color: #334155 !important;
    margin-top: 1rem !important;
}
button {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
    color: #1e293b !important;
    transition: all 0.2s;
}
button:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
}
.gr-button-primary {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: white !important;
}
.gr-button-primary:hover {
    background: #1d4ed8 !important;
}
textarea, input, .gr-box {
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    font-family: monospace !important;
    font-size: 0.9rem !important;
}
label {
    font-weight: 500 !important;
    color: #334155 !important;
}
"""

# ---------- Build Dashboard ----------
with gr.Blocks(title="AI Customer Support Environment") as demo:
    gr.Markdown("# Customer Support RL Environment")
    gr.Markdown("Interactive environment for training and evaluating AI agents on customer service tasks.")
    
    with gr.Row():
        # Left panel: controls
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            action_dropdown = gr.Dropdown(
                choices=["/refund", "/verify_purchase", "/process_refund", "/escalate", "/invalid", "Reset"],
                label="Action",
                value="/refund"
            )
            with gr.Row():
                take_btn = gr.Button("Take Action", variant="primary")
                auto_checkbox = gr.Checkbox(label="Auto mode (agent suggests next)", value=False)
            auto_run_btn = gr.Button("Run Full Episode (trained agent)")
            
            gr.Markdown("---")
            gr.Markdown("### Status")
            output_query = gr.Textbox(label="Customer Query", lines=2, interactive=False)
            output_reward = gr.Textbox(label="Current State", interactive=False)
            output_next_action = gr.Textbox(label="Agent Suggestion", interactive=False)
        
        # Right panel: history and output
        with gr.Column(scale=1):
            gr.Markdown("### Conversation Log")
            output_history = gr.Textbox(label="", lines=12, interactive=False)
            auto_output = gr.Textbox(label="Agent Episode Result", lines=10, interactive=False)
    
    # Bind events
    take_btn.click(
        interact,
        inputs=[action_dropdown, auto_checkbox],
        outputs=[output_query, output_reward, output_history, output_next_action]
    )
    auto_run_btn.click(
        reset_and_agent_run,
        inputs=[],
        outputs=[auto_output]
    )
    
    # Initial reset
    demo.load(lambda: interact("Reset", False), outputs=[output_query, output_reward, output_history, output_next_action])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
