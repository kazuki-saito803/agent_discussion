from fastapi import FastAPI
from pydantic import BaseModel
from models import agent
import uvicorn

app = FastAPI()

class Item(BaseModel):
    theme: str
    turn: int

@app.on_event("startup")
def load_agents():
    global agents
    agents = {
        "educator": agent("educator"),
        "engineer": agent("engineer"),
        "ethics_committee": agent("ethics_committee"),
        "gurdian": agent("gurdian"),
        "student": agent("student")
    }

@app.post("/discussion/")
def discussion(item: Item):
    agent_instances = [
        agents["educator"],
        agents["engineer"],
        agents["ethics_committee"],
        agents["gurdian"],
        agents["student"]
    ]

    conversation_history = {turn: [] for turn in range(item.turn)}

    for turn in range(item.turn):
        for idx, instance in enumerate(agent_instances):
            if idx == 0:
                prompt = f"{item.theme}についてあなたの意見を聞かせてください。"
            else:
                last_comment = conversation_history[turn][-1]
                prompt = f"ある方は「{last_comment}」と言っています。{item.theme}についてあなたの意見を聞かせてください。"

            response = instance.predict(prompt)
            conversation_history[turn].append(f"{instance.role}: {response}")

    return conversation_history


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
