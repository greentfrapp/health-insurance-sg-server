from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
import logging
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
    current_policy: Optional[str] = None
    query: str
    history: List[ChatMessage] = []


@app.get("/status")
def get_status():
    return {"version": llamaqa.__version__}


# Helper function for logging
async def stream_thoughts_helper(
    agent: PaperQAAgent,
    query: str,
    history: List[ChatMessage] = [],
    current_document: Optional[str] = None,
    step_by_step = False,
):
    agent.memory.set(history)
    stream = agent.stream_thoughts(query, current_document, step_by_step)
    async for chunk in stream:
        print(f"\033[38;5;228m{chunk}\033[0m")
        yield(chunk)
    print(f"\033[38;5;69mTotal cost: {agent.cost_logger.total_cost}\033[0m")


@app.post("/stream_query")
async def post_stream_query(payload: QueryPayload):
    agent = PaperQAAgent.from_config()
    return StreamingResponse(
        stream_thoughts_helper(agent, payload.query, payload.history, payload.current_policy),
        media_type="text/event-stream",
    )


async def main():
    logging.basicConfig()

    agent = PaperQAAgent.from_config()
    agent.cost_logger.logger.setLevel(logging.INFO)
    async def test_stream_thoughts(query: str, step_by_step=False):
        response = stream_thoughts_helper(agent, query, [], "AIA HealthShield Gold", step_by_step)
        async for _ in response:
            pass
    query = "lasik coverage for this policy"
    while True:
        # response = test_stream_thoughts("lasik coverage for aia gold")
        response = await test_stream_thoughts(query, step_by_step=True)
        # response = test_stream_thoughts("what was my last question")
        # response = test_stream_thoughts("how about for aia")
        # response = test_stream_thoughts("summarize all of that in a table format")
        # response = test_stream_thoughts("Point form instead")
        # response = test_stream_thoughts("Tell me about prosthetic coverage for ntuc income. Give your answer in point form")
        # response = test_stream_thoughts("do the same for aia")
        query = input("Reply: ")
        if query == "q": break
    agent.pprint_memory()
    print("Total cost: ", agent.cost_logger.total_cost)


if __name__ == "__main__":
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
