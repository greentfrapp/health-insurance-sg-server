import json
from typing import AsyncGenerator, Optional

import nest_asyncio

from llamaqa.agents.paperqa.base import PaperQAAgent
from llamaqa.llms.litellm_model import LiteLLMModel
from llamaqa.llms.llm_model import LLMModel
from llamaqa.llms.llm_result import llm_parse_json
from llamaqa.utils.logger import CostLogger

nest_asyncio.apply()

BASIC_MODEL_EVAL_SYSTEM_PROMPT = """
You are a large language model designed to evaluate responses according to certain stipulated criteria.

You will be given a response string and a condition.

The condition will stipulate a criteria and optionally provide a grading scheme, such as "on a scale of 1 to 10 with 1 being strongly disagree and 10 being strongly agree".
If no grading scheme is provided, simply grade with "PASS" or "FAIL".

Return your response as a JSON recording the grade, for instance:

{{
    "grade": "PASS"
}}
"""


CONVERSATIONAL_EVAL_SYSTEM_PROMPT = """
You are a large language model designed to evaluate responses according to certain stipulated criteria.

You will be given a condition. You will then engage in a conversation and grade your conversation partner according to the provided condition.

The condition will stipulate a criteria and optionally provide a grading scheme, such as "on a scale of 1 to 10 with 1 being strongly disagree and 10 being strongly agree".
If no grading scheme is provided, simply grade with "PASS" or "FAIL".

At any point, end the conversation by saying "END CONVERSATION" followed by a JSON recording the grade, for instance:

END CONVERSATION
{{
    "grade": "PASS"
}}

Condition:
{condition}
"""


async def system_fn(response: str, history=None, agent=None):
    agent = agent or PaperQAAgent.from_config()
    agent.memory.set(history or [])
    stream = agent.stream_thoughts(response, current_document=None, step_by_step=False)
    final_response = []
    async for chunk in stream:
        if chunk.startswith("Final Response: "):
            final_response = json.loads(chunk[len("Final Response: ") :])
    return final_response[-1]["content"], agent.memory.chat_store.store["chat_history"]


class ModelGrader:
    llm: LLMModel
    cost_logger: CostLogger

    def __init__(self, llm: Optional[LLMModel] = None):
        self.cost_logger = CostLogger("modelgrader-cost")
        self.llm = llm or LiteLLMModel(
            name="gemini/gemini-1.5-flash-002", cost_logger=self.cost_logger
        )

    async def basic_model_eval(self, response: str, condition: str):
        result = await self.llm.run_prompt(
            f"Response: {response}\n\nCondition: {condition}",
            {},
            system_prompt=BASIC_MODEL_EVAL_SYSTEM_PROMPT,
        )
        return llm_parse_json(result.text).get("grade")

    async def conversational_eval(
        self,
        condition: str,
        system_fn: AsyncGenerator,
        agent: PaperQAAgent = None,
        verbose=False,
    ):
        messages = [
            {
                "role": "system",
                "content": CONVERSATIONAL_EVAL_SYSTEM_PROMPT.format(
                    condition=condition
                ),
            },
            {"role": "user", "content": "Hello"},
        ]
        grader_response = ""
        gradee_history = []
        rounds = 0
        max_rounds = 5

        llm_result = await self.llm.achat(messages)
        grader_response = llm_result.text
        if verbose:
            print(f"Grader: {grader_response}")

        while "END CONVERSATION" not in grader_response:
            gradee_response, gradee_history = await system_fn(
                grader_response, gradee_history, agent
            )
            if verbose:
                print(f"Gradee: {gradee_response}")
            messages += [
                {
                    "role": "assistant",
                    "content": grader_response,
                },
                {
                    "role": "user",
                    "content": gradee_response,
                },
            ]
            rounds += 1
            if rounds > max_rounds:
                break
            llm_result = await self.llm.achat(messages)
            grader_response = llm_result.text
            if verbose:
                print(f"Grader: {grader_response}")
        messages += [
            {
                "role": "assistant",
                "content": grader_response,
            },
        ]

        return llm_parse_json(grader_response).get("grade"), messages


async def main():
    grader = ModelGrader()

    response = 'Based on the provided AIA HealthShield Gold Max policy documents, LASIK surgery is likely not covered.  The policy explicitly excludes coverage for "correcting refractive errors," such as short-sightedness, which is the condition LASIK surgery addresses (0a442226Health2024 pages 32-32 quote1).  Several sections of the policy booklet detail various aspects of coverage, including hospitalisation, surgical procedures, and other benefits, but none mention LASIK (0a442226Health2024 pages 25-26, 0a442226Health2024 pages 3-5, 0a442226Health2024 pages 32-33, 0a442226Health2024 pages 7-8).  Therefore, while not explicitly stated as excluded in every section, the strong implication from the exclusion of refractive error correction suggests LASIK would not be covered.'
    condition = "Check that the response is negative because LASIK is not covered"
    grade = await grader.basic_model_eval(
        response,
        condition,
    )
    print(grade)

    grade, messages = await grader.conversational_eval(
        condition="Ask about lasik coverage under the AIA HealthShield Gold Max policy and check that the response is negative. Do not explicitly ask about AIA HealthShield first. Wait to be asked which policy you are interested in.",
        system_fn=system_fn,
    )
    print(grade)
    print(messages)


if __name__ == "__main__":
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # loop.run_until_complete(basic_model_eval(response, condition))

    loop.run_until_complete(main())
