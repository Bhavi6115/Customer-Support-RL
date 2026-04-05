import asyncio
import httpx
from models import Action, Observation

class SimpleClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def reset(self):
        response = await self.client.post(f"{self.base_url}/reset", json={})
        data = response.json()
        return Observation(**data["observation"])
    
    async def step(self, action: Action):
        # Use model_dump() instead of dict()
        response = await self.client.post(f"{self.base_url}/step", json=action.model_dump())
        data = response.json()
        obs = Observation(**data["observation"])
        return obs, data["reward"], data["terminated"], data["truncated"], data["info"]
    
    async def close(self):
        await self.client.aclose()

async def main():
    client = SimpleClient()
    obs = await client.reset()
    print(f"Customer Query: {obs.query}\n")
    
    actions = ["/refund", "/verify_purchase", "/process_refund"]
    total_reward = 0
    
    for action_value in actions:
        action = Action(action_value=action_value)
        obs, reward, terminated, truncated, info = await client.step(action)
        total_reward += reward
        print(f"Action: {action_value} -> Reward: {reward}")
        print(f"  Stage: {obs.stage}")
        if terminated:
            break
    
    print(f"\nTotal reward: {total_reward}")
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())