from pydantic import BaseModel

class Action(BaseModel):
    action_value: str   # e.g., "/refund", "/verify_purchase"

class Observation(BaseModel):
    query: str          # customer's issue
    history: str        # conversation log
    stage: str          # "in_progress" or "resolved"