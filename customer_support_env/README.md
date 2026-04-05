# 🤖 Customer Support RL Environment

**Meta PyTorch OpenEnv Hackathon 2026 – Round 1 Submission**

A production‑ready Reinforcement Learning environment where an AI agent learns to resolve customer service issues (refunds, technical support, order status). Built with FastAPI, compliant with the OpenEnv specification, and fully containerized with Docker.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/facebookresearch/openenv)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-blue)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44-orange)](https://gradio.app/)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Environment Spec](#environment-spec)
- [Installation](#installation)
- [Run Locally](#run-locally)
- [Testing (Grader)](#testing-grader)
- [Train an Agent](#train-an-agent)
- [Dashboard](#dashboard)
- [Docker & Deployment](#docker--deployment)
- [Project Structure](#project-structure)
- [Hackathon Submission](#hackathon-submission)

---

## 🎯 Overview

The environment simulates a **customer service chatbot**. The agent must choose the correct sequence of actions to resolve a randomly selected customer intent:

| Intent | Correct Sequence |
|--------|------------------|
| Refund | `/refund` → `/verify_purchase` → `/process_refund` |
| Technical | `/troubleshoot` → `/ask_os_version` → `/escalate_to_tech` |
| Order Status | `/track_order` → `/check_shipping` → `/offer_compensation` |

At each reset, one intent is chosen at random. The agent learns to generalise across tasks.

---

## 🧠 Environment Spec

### Observation (what the agent sees)
```json

  "query": "I need a refund for my order #12345",
  "history": "Conversation started.\nAgent: /refund\nSystem: Correct. Next step: /verify_purchase",
  "stage": "in_progress"

Action (what the agent can do)
Any string command, but the environment recognises:

Correct‑sequence actions (listed above)

/escalate (gives up)

/invalid (any other – used for testing)

Reward Logic
Scenario	Reward
Correct step in sequence	+1
Full resolution (last correct action)	+10
Invalid action	-2
Escalation (/escalate)	-5
API Endpoints (OpenEnv compliant)
POST /reset → initial observation

POST /step → takes {"action_value": "..."} → returns (observation, reward, terminated, truncated, info)

GET /state → full internal state (intent, current step, history, etc.)

GET /health → {"status": "healthy"}

⚙️ Installation
bash
git clone https://github.com/YOUR_USERNAME/customer-support-env.git
cd customer-support-env
pip install -r requirements.txt
🚀 Run Locally
Start the server (one terminal):

bash
python api.py
Test with the client (another terminal):

bash
python client.py
Expected output:

text
Customer Query: I need a refund for my order #12345

Action: /refund -> Reward: 1.0
  Stage: in_progress
Action: /verify_purchase -> Reward: 1.0
  Stage: in_progress
Action: /process_refund -> Reward: 10.0
  Stage: resolved

Total reward: 12.0
✅ Testing (Grader)
Run automated programmatic checks (required for Round 1):

bash
python grader.py
All tests should pass with green checkmarks. The grader verifies:

Health endpoint

Reset structure

Step endpoint

State endpoint

Reward logic (correct, invalid, escalate)

Termination flag

Multiple‑intent randomness

🧪 Train an Agent
Train a Q‑learning agent to solve the environment:

bash
python train_agent.py
After training, the Q‑table is saved as trained_q_table.pkl. The dashboard can then use it for autonomous demonstration.

🖥️ Dashboard
Launch the Gradio web UI for manual/auto interaction:

bash
python dashboard.py
Open http://localhost:7861 in your browser. Features:

Manual action dropdown

Auto mode (agent suggests next action using trained Q‑table)

“Run Full Episode” with trained agent

Live conversation log and reward display

🐳 Docker & Deployment
Build Docker image (local)
bash
docker build -t customer-support-env .
docker run -p 8000:8000 customer-support-env
OpenEnv‑compliant build & push (used by GitHub Actions)
bash
pip install openenv-core
python -m openenv.cli build
python -m openenv.cli validate --verbose
python -m openenv.cli push   # deploys to Hugging Face Spaces
GitHub Actions CI/CD
The repository includes .github/workflows/deploy.yml that automatically builds and pushes to Hugging Face Spaces on every push to main.
Secret required: HF_TOKEN (Hugging Face write token).

📂 Project Structure
text
customer-support-env/
├── .github/workflows/
│   └── deploy.yml          # GitHub Actions CI/CD
├── server/
│   └── app.py              # OpenEnv entry point (imports api.py)
├── api.py                  # Main FastAPI server
├── models.py               # Pydantic models (Action, Observation)
├── client.py               # Test client
├── grader.py               # Programmatic tests (required)
├── dashboard.py            # Gradio UI
├── train_agent.py          # Q‑learning training script
├── trained_q_table.pkl     # Trained Q‑table (optional)
├── requirements.txt        # Dependencies
├── Dockerfile              # Container definition
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Project metadata & dependencies
├── README.md               # This file
└── .gitignore
🏆 Hackathon Submission
This environment is submitted to Round 1 of the Meta PyTorch OpenEnv Hackathon 2026.

GitHub repository: https://github.com/YOUR_USERNAME/customer-support-env

Live Hugging Face Space: https://huggingface.co/spaces/YOUR_USERNAME/customer-support-env

All requirements fulfilled:

✅ Mini‑RL environment with defined tasks, grader, and reward logic.

✅ Programmatic checks (grader.py).

✅ OpenEnv API (/reset, /step, /state).

✅ Docker containerization.

✅ Open source on GitHub.

✅ Deployed to Hugging Face Spaces.

👤 Author
Bhavika vasule– GitHub
Hackathon: Meta PyTorch OpenEnv Hackathon 2026 by Scaler School of Technology
Date: April 2026

License: MIT

text

Replace `YOUR_USERNAME` with your actual GitHub username. After your workflow runs successfully, replace the Hugging Face Space URL placeholder with the real one. Good luck! 🚀