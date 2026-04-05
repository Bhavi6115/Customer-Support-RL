import numpy as np
import httpx
import asyncio
import pickle
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        # Create a simple string key from the observation
        return f"{state['query'][:30]}_{state['stage']}_{state['history'][-50:]}"

    def get_action(self, state_key):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.actions))
        return np.argmax(self.q_table[state_key])

    def update(self, state_key, action, reward, next_state_key):
        best_next = np.max(self.q_table[next_state_key])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error

async def train_agent(episodes=200):
    actions = ["/refund", "/verify_purchase", "/process_refund", "/escalate", "/invalid"]
    agent = QLearningAgent(actions)
    rewards_per_episode = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for episode in range(episodes):
            # Reset environment
            resp = await client.post("http://localhost:8000/reset", json={})
            state = resp.json()["observation"]
            state_key = agent.get_state_key(state)
            total_reward = 0
            step = 0
            terminated = False

            while not terminated and step < 10:
                action_idx = agent.get_action(state_key)
                action = actions[action_idx]

                step_resp = await client.post("http://localhost:8000/step", json={"action_value": action})
                step_data = step_resp.json()
                next_state = step_data["observation"]
                reward = step_data["reward"]
                terminated = step_data["terminated"]

                next_state_key = agent.get_state_key(next_state)
                agent.update(state_key, action_idx, reward, next_state_key)

                total_reward += reward
                state_key = next_state_key
                step += 1

            rewards_per_episode.append(total_reward)
            if episode % 10 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}")

    # Convert defaultdict to regular dict for pickling
    q_table_dict = dict(agent.q_table)
    with open("trained_q_table.pkl", "wb") as f:
        pickle.dump(q_table_dict, f)
    print("Training complete! Q-table saved.")
    return q_table_dict, rewards_per_episode

if __name__ == "__main__":
    asyncio.run(train_agent(episodes=200))