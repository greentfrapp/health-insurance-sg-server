import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import nest_asyncio

from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.litellm import LiteLLM
import os

from llamaqa.llms.embedding_model import LiteLLMEmbeddingModel
from llamaqa.llms.litellm_model import LiteLLMModel
from llamaqa.store.supabase_store import SupabaseStore
from llamaqa.tools.paperqa_tools import PaperQAToolSpec, tell_llm_about_failure_in_extract_reasoning_step
from llamaqa.utils.inner_context import InnerContext
from llamaqa.utils.prompts import PAPERQA_SYSTEM_PROMPT
from llamaqa.utils.react_agent import ReActOutputParser

from llamaqa.agents.paperqa import PaperQAAgent
from llamaqa.utils.stream import stream_thoughts
from llamaqa.utils.api import format_response


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


# Initialize models
embedding_model = LiteLLMEmbeddingModel(
    name="gemini/text-embedding-004"
)
summary_llm_model = LiteLLMModel(
    name="gemini/gemini-1.5-flash-002"
)
store = SupabaseStore(
    supabase_url=os.environ["SUPABASE_URL"],
    supabase_key=os.environ["SUPABASE_SERVICE_KEY"],
)
context = InnerContext()
toolspec = PaperQAToolSpec(
    store=store,
    context=context,
    embedding_model=embedding_model,
    summary_llm_model=summary_llm_model,
)
do_not_have_access_phrases = [
    "need more information",
    "do not have access",
    "need access",
]
agent = None


@app.get("/init")
def get_init_agent():
    llm = LiteLLM("gemini/gemini-1.5-flash-002")
    # llm = OpenAI("gpt-4o-mini")
    # llm = OpenAI("gpt-3.5-turbo-instruct")
    
    global agent
    agent = PaperQAAgent.from_tools(
        toolspec.to_tool_list(),
        llm=llm,
        verbose=True,
        # max_iterations=20,
        handle_reasoning_failure_fn=tell_llm_about_failure_in_extract_reasoning_step,
        output_parser=ReActOutputParser(),
    )
    agent.update_prompts({
        "agent_worker:system_prompt": PromptTemplate(PAPERQA_SYSTEM_PROMPT)
    })


@app.get("/status")
def get_status():
    global agent
    if agent is None:
        get_init_agent()
    return {
        "history_length": len(agent.memory.chat_store.to_dict()["store"]["chat_history"]),
    }


@app.post("/query")
def post_query(payload: QueryPayload):
    global agent

    response = agent.chat(f"{payload.query}\nRemember to call gather_evidence if the user is asking about insurance, especially if you are citing anything.")
    
    memory_dict = agent.memory.chat_store.to_dict()["store"]["chat_history"]
    memory_dict = [ChatMessage(**i) for i in memory_dict]
    memory_dict[-2].content = payload.query
    agent.memory.set(memory_dict)

    # Format response
    return format_response(payload.query, str(response), context)


def stream_query_helper(query: str):
    global agent
    stream = stream_thoughts(agent, context, query)
    for chunk in stream:
        print([chunk])
        yield(chunk)


@app.post("/stream_query")
def post_stream_query(payload: QueryPayload):
    return StreamingResponse(stream_query_helper(payload.query), media_type="text/event-stream")


if __name__ == "__main__":
    status = get_status()
    print(status)
    post_query(QueryPayload(query="Hello"))
    status = get_status()
    print(status)
