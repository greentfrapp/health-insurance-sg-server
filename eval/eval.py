import json
import logging
import re
from typing import List, Optional

import nest_asyncio
from dotenv import load_dotenv
from llamaqa.agents.paperqa.base import PaperQAAgent

from .model_grader import ModelGrader, system_fn

load_dotenv()
nest_asyncio.apply()


_STY_PASS_COLOR = "\033[38;5;46m"
_STY_FAIL_COLOR = "\033[38;5;196m"
_STY_SCORE_COLOR = "\033[38;5;226m"
_STY_RESET = "\033[0m"


async def test_stream_thoughts(
    agent: PaperQAAgent, document: str, query: str, history=None, step_by_step=False
):
    history = history or []
    agent.memory.set(history)
    stream = agent.stream_thoughts(query, document, step_by_step)
    buffer = ""
    async for chunk in stream:
        buffer += chunk
    history = agent.memory.chat_store.store["chat_history"]

    final_response = buffer
    if "Final Response: " in buffer:
        final_response = json.loads(buffer.split("Final Response: ")[1])[-1]["content"]
    return final_response, history


async def eval(test_file: str, test_labels: Optional[List[str]] = None, verbose=False):
    with open("eval/tests/" + test_file, "r") as file:
        tests = json.load(file)

    logging.basicConfig()

    eval_logger = logging.getLogger("eval_logger")
    eval_logger.setLevel(logging.INFO)

    agent = PaperQAAgent.from_config(verbose=verbose)
    agent.cost_logger.logger.setLevel(logging.INFO)

    grader = ModelGrader()
    grader.cost_logger.logger.setLevel(logging.INFO)

    for i, test in enumerate(tests, start=1):
        label = test.get("label")
        if test_labels and label not in test_labels:
            eval_logger.info(
                f"Skipping eval #{i}/{len(tests)}{' - ' + label if label else ''}"
            )
            continue

        eval_logger.info(
            f"Running eval #{i}/{len(tests)}{' - ' + label if label else ''}"
        )

        messages = test.get("messages", [])
        open_test = test.get("test", "")

        if messages:
            history = []
            document = test.get("setup", {}).get("current_document")
            for message in messages:
                content = message.get("content", "")
                assert_response_includes = message.get("assert_response_includes", [])
                open_ended_eval = message.get("open_ended_eval", "")
                response, history = await test_stream_thoughts(
                    agent, document, content, history
                )
                if assert_response_includes:
                    for phrase in assert_response_includes:
                        match = re.search(rf"[^a-zA-Z\d]{phrase}[^a-zA-Z\d]", response)
                        if match:
                            eval_logger.info(
                                f"{_STY_PASS_COLOR}Passed test: Response includes {phrase}{_STY_RESET}"
                            )
                        else:
                            eval_logger.warn(
                                f"{_STY_FAIL_COLOR}Failed test: Response does not include {phrase}{_STY_RESET}"
                            )
                if open_ended_eval:
                    grade = await grader.basic_model_eval(response, open_ended_eval)
                    if grade == "PASS":
                        eval_logger.info(
                            f"{_STY_PASS_COLOR}Passed test: {open_ended_eval}{_STY_RESET}"
                        )
                    elif grade == "FAIL":
                        eval_logger.info(
                            f"{_STY_FAIL_COLOR}Failed test: {open_ended_eval}{_STY_RESET}"
                        )
                    else:
                        eval_logger.info(
                            f"{_STY_SCORE_COLOR}Grade={grade}: {open_ended_eval}{_STY_RESET}"
                        )
        elif open_test:
            grade, messages = await grader.conversational_eval(
                open_test,
                system_fn,
                agent,
                verbose=verbose,
            )
            if grade == "PASS":
                eval_logger.info(
                    f"{_STY_PASS_COLOR}Passed test: {open_test}{_STY_RESET}"
                )
            elif grade == "FAIL":
                eval_logger.info(
                    f"{_STY_FAIL_COLOR}Failed test: {open_test}{_STY_RESET}"
                )
            else:
                eval_logger.info(
                    f"{_STY_SCORE_COLOR}Grade={grade}: {open_test}{_STY_RESET}"
                )

    print("Total cost: ", agent.cost_logger.total_cost)


def main():
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("test", type=str)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(eval(args.test, args.labels, verbose=args.verbose))


if __name__ == "__main__":
    main()
