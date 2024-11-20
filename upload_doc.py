import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from llamaqa.llms import LiteLLMEmbeddingModel, LiteLLMModel
from llamaqa.reader.parsing_settings import ParsingSettings
from llamaqa.reader.reader import Reader
from llamaqa.store.supabase_store import SupabaseStore
from llamaqa.utils.logger import CostLogger

with open("./essential_policies_plus.json", "r") as file:
    POLICIES = json.load(file)


load_dotenv()


logging.basicConfig()
cost_logger = CostLogger()
cost_logger.logger.setLevel(logging.INFO)


async def main():
    parse_config = ParsingSettings()
    parse_config.disable_doc_valid_check = True
    embedding_model = LiteLLMEmbeddingModel(
        name="gemini/text-embedding-004", cost_logger=cost_logger
    )
    llm_model = LiteLLMModel(
        name="gemini/gemini-1.5-flash-002", cost_logger=cost_logger
    )
    reader = Reader(
        parse_config=parse_config,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )

    store = SupabaseStore(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SERVICE_KEY"],
    )
    # existing_dockeys = await store.get_existing_dockeys()

    for i, policy in enumerate(POLICIES):
        if i < 23:
            continue
        policy["path"] = "Insurance Policies/" + policy["path"]
        cost_logger.start_split()
        print(f"Uploading #{i}/{len(POLICIES)} {policy['title']}...")
        # doc = await reader.read_doc(
        #     **policy,
        #     summarize_chunks=False,
        # )

        # if doc.dockey in existing_dockeys:
        #     print(f"{policy['title']} already uploaded")
        #     continue

        doc = await reader.read_doc(
            **policy,
            summarize_chunks=True,
        )
        cost_logger.get_split()
        await store.upload(doc, ignore_duplicate_doc=True)

    print(f"Total cost: {cost_logger.total_cost}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
