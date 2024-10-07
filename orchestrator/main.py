# planner.py

import asyncio
import pprint

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from orchestrator import app
from orchestrator.types.plan_execute_state import State

n_models_planner = 1
n_models_executor = 2


async def run_workflow(
    input_message: str, n_models_planner: int, n_models_executor: int
):
    initial_state = State(
        messages=[HumanMessage(content=input_message)],
        system_task=input_message,
        n_models_planner=n_models_planner,
        n_models_executor=n_models_executor,
    )

    print('Initial state:', pprint.pp(initial_state))

    async for output in app.app.astream(initial_state):
        # print(output)
        if isinstance(output, dict) and any(value == END for value in output.values()):
            print('Task completed.')
            break


async def run_as_main():
    await run_workflow(
        "Invent rules for a board game with a scoring system on a 1-step scale. Play that game until you reach a score of 5. Respond with 'FINISH' once a player has reached score 5. Do not ever mention the word 'FINISH' unless a player has reached 5 points.",
        n_models_planner=n_models_planner,
        n_models_executor=n_models_executor,
    )


def main():
    asyncio.run(run_as_main())


if __name__ == '__main__':
    main()
