## POST API for to response the user  
# from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from cl_functions import main_
# import nltk

app = FastAPI()

class QA(BaseModel):
    question: str


@app.post('/api/predict')
def predict(request: QA):
    """
        requestBody contain the question
        in a string fromat and it return the response
    """
    msg = main_(request.question.lower())
    print(msg)
    return {'data': msg}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)