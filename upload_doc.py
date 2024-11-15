import asyncio
import json
import os

from dotenv import load_dotenv

from llamaqa.llms import LiteLLMEmbeddingModel, LiteLLMModel
from llamaqa.reader.reader import Reader
from llamaqa.reader.parsing_settings import ParsingSettings
from llamaqa.store.supabase_store import SupabaseStore


with open("./policies.json", "r") as file:
    POLICIES = json.load(file)


load_dotenv()


async def main():
    parse_config = ParsingSettings()
    embedding_model = LiteLLMEmbeddingModel(name="gemini/text-embedding-004")
    llm_model = LiteLLMModel(name="gemini/gemini-1.5-flash-002")
    reader = Reader(
        parse_config=parse_config,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )

    store = SupabaseStore(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SERVICE_KEY"],
    )

    for policy in POLICIES:
        print(f"Uploading {policy['title']}...")
        doc = await reader.read_doc(
            **policy,
            summarize_chunks=True,
        )
        await store.upload(doc)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
