import gradio as gr
import httpx
import asyncio

# ---------- CONFIGURATION ----------
# Replace with your actual API Space URL (the one you submitted)
API_BASE_URL = "https://goolu123-customer-support-env.hf.space"

# ---------- Helper Functions ----------
async def interact(action, auto_mode=False):
    async with httpx.AsyncClient(timeout=30.0) as client:
        if action == "Reset":
            resp = await client.post(f"{API_BASE_URL}/reset", json={})
            data = resp.json()
            obs = data["observation"]
            return obs["query"], "🔄 Reset complete. Ready.", "", ""
        else:
            resp = await client.post(f"{API_BASE_URL}/step", json={"action_value": action})
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
                # For auto mode, you could load a trained agent here.
                # Since this is a separate dashboard, we'll skip for now.
                pass
            return obs["query"], status, obs["history"], next_action

async def reset_and_agent_run():
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(f"{API_BASE_URL}/reset", json={})
        total_reward = 0
        terminated = False
        step = 0
        history_log = []
        # Example: hardcoded correct sequence for "refund" intent
        actions = ["/refund", "/verify_purchase", "/process_refund"]
        while not terminated and step < len(actions):
            action = actions[step]
            step_resp = await client.post(f"{API_BASE_URL}/step", json={"action_value": action})
            step_data = step_resp.json()
            reward = step_data["reward"]
            total_reward += reward
            terminated = step_data["terminated"]
            history_log.append(f"**Step {step+1}:** `{action}` → Reward {reward}")
            step += 1
        history_log.append(f"\n## 🎉 **Total Reward: {total_reward}**")
        return "\n".join(history_log)

# ---------- Orange / Dark Theme CSS ----------
orange_css = """
:root {
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --primary: #f97316;
    --primary-light: #fb923c;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
}
body {
    background: var(--bg-dark);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 2rem auto !important;
    background: var(--card-bg);
    border-radius: 24px;
    box-shadow: 0 20px 35px -10px rgba(0,0,0,0.5);
    padding: 2rem !important;
    border: 1px solid #334155;
}
h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--primary), var(--primary-light));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    margin-bottom: 0.25rem;
}
h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    border-left: 4px solid var(--primary);
    padding-left: 12px;
    margin-top: 1rem !important;
}
button {
    background: linear-gradient(135deg, var(--primary), #ea580c) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(249,115,22,0.4);
}
.gr-button-primary {
    background: linear-gradient(135deg, var(--primary), #ea580c) !important;
}
textarea, input, .gr-box {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}
label {
    font-weight: 500 !important;
    color: var(--text-muted) !important;
}
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #1e293b;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: var(--primary);
    border-radius: 10px;
}
"""

# ---------- Build Gradio UI ----------
with gr.Blocks(title="Customer Support RL Environment", theme=gr.themes.Soft(), css=orange_css) as demo:
    gr.Markdown("# 🍊 Customer Support RL Environment")
    gr.Markdown("### Train an AI to resolve customer issues using Reinforcement Learning")
    
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
    demo.load(lambda: interact("Reset", False), outputs=[output_query, output_reward, output_history, output_next_action])

# ---------- Launch ----------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
