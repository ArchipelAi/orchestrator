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
        print(output)
        if isinstance(output, dict) and any(value == END for value in output.values()):
            print('Task completed.')
            break

        # try:
        #     if isinstance(output, tuple) and len(output) == 2:
        #         state, feature_vector = output
        #         state['feature_vectors'] = state.get('feature_vectors', []) + [
        #             feature_vector
        #         ]
        #     else:
        #         state = output

        #     # print("Final state for iteration:", json.dumps(state, cls=CustomJSONEncoder, indent=2))

        #     if 'messages' in state and len(state.messages) > 1:
        #         print(
        #             f"Step {state.get('current_step', 'N/A')}: {state['messages'][-1].content}"
        #         )

        #         # print(f"Current n_models_planner: {state.get('n_models_planner', 'N/A')}, "
        #         f"n_models_executor: {state.get('n_models_executor', 'N/A')}"

        #     if 'feature_vectors' in state and state['feature_vectors']:
        #         print(f"Latest feature vector: {state['feature_vectors'][-1]}")

        # except Exception:
        #     # print(f"Error processing output: {e}")
        #     # print(f"Problematic output: {output}")
        #     pass
        #     # print(traceback.format_exc())


# async def run_as_main():
#     await run_workflow(
#         'Order a vegetarian pizza in Munich and have it delivered to Arcisstrasse 21, 80331 Munich. A human will pay the delivery person upon arrival.',  # Invent rules for a board game with a scoring system on a 1-step scale and play that game until you reach score 5. Respond with 'FINISH' once a player has reached score 5. Do not ever mention the word 'FINISH' unless a player has reached 5 points.
#         n_models_planner=1,
#         n_models_executor=2,
#     )


async def run_as_main():
    await run_workflow(
        "Invent rules for a board game with a scoring system on a 1-step scale and play that game until you reach score 5. Respond with 'FINISH' once a player has reached score 5. Do not ever mention the word 'FINISH' unless a player has reached 5 points.",
        n_models_planner=n_models_planner,
        n_models_executor=n_models_executor,
    )


def main():
    asyncio.run(run_as_main())


if __name__ == '__main__':
    main()
