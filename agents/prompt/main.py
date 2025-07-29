from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import models
import templates


app = FastAPI()

class Item(BaseModel):
    theme: str
    turn: int

@app.post("/discussion/")
def discussion(item: Item):
    template = {
        "educator": templates.educator,
        "engineer": templates.engineer,
        "ethics_committee": templates.ethics_committee,
        "gurdian": templates.gurdian,
        "student": templates.student
    }
    agents = ["educator", "engineer", "ethics_committee", "gurdian", "student"]
    conversation_history = {turn: [] for turn in range(item.turn)}    
    for turn in range(item.turn):
        for j, agent in enumerate(agents):
            if j == 0:
                tmp = models.predict(item.theme, template[agent], True)
                conversation_history[turn].append(f"{agents[j]}: {tmp}")
            else:
                last_comment = conversation_history[turn][-1]
                tmp = models.predict(item.theme, template[agent], False, last_comment=last_comment)
                conversation_history[turn].append(f"{agents[j]}: {tmp}")
    return conversation_history

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)