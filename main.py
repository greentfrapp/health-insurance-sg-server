from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
import nest_asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from llama_index.core.base.llms.types import ChatMessage

import llamaqa
from llamaqa.agents.paperqa.base import PaperQAAgent


load_dotenv()
nest_asyncio.apply()


app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
origins = [
    "http://localhost:5173",
    "https://health-insurance-sg.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryPayload(BaseModel):
    query: str
    history: List[ChatMessage] = []


@app.get("/status")
def get_status():
    return {"version": llamaqa.__version__}


# Helper function for logging
def stream_thoughts_helper(
    agent: PaperQAAgent,
    query: str,
    history: List[ChatMessage] = [],
):
    agent.memory.set(history)
    stream = agent.stream_thoughts(query)
    for chunk in stream:
        print(f"\033[38;5;228m{chunk}\033[0m")
        yield(chunk)


@app.post("/stream_query")
def post_stream_query(payload: QueryPayload):
    agent = PaperQAAgent.from_config()
    return StreamingResponse(
        stream_thoughts_helper(agent, payload.query, payload.history),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    agent = PaperQAAgent.from_config()
    def test_stream_thoughts(query: str):
        response = stream_thoughts_helper(agent, query)
        for _ in response:
            pass
    response = test_stream_thoughts("summarize aia gold")
    # response = test_stream_thoughts("what was my last question")
    # response = test_stream_thoughts("how about for aia")
    # response = test_stream_thoughts("summarize all of that in a table format")
    # response = test_stream_thoughts("Point form instead")
    # response = test_stream_thoughts("Tell me about prosthetic coverage for ntuc income. Give your answer in point form")
    # response = test_stream_thoughts("do the same for aia")
    agent.pprint_memory()
