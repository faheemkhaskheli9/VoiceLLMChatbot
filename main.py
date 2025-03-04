from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pdf_qa_bot_chain import PDFChatbot
from chatbot_internet import InternetChatBot

app = FastAPI()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

internet_bot = InternetChatBot()

qa_dataset = {
    "THE MUSLIM FAMILY LAWS ORDINANCE, 1961": "data/administratoreecaf3b490e2d43d2e3b50c0c068b5d7.pdf"
}

bot = PDFChatbot()
bot.load_data(qa_dataset["THE MUSLIM FAMILY LAWS ORDINANCE, 1961"])

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    # Here you would implement your logic to generate an answer
    # For demonstration, we will return a static answer
    result = bot.ask_question(question.question)
    return Answer(answer=result)

@app.post("/chat", response_model=Answer)
async def chat_with_bot(question: Question):
    # Here you would implement your logic to generate an answer
    # For demonstration, we will return a static answer
    result = internet_bot.ask(question.question)
    return Answer(answer=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
