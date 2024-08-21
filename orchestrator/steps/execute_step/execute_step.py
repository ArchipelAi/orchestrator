import asyncio
import dataclasses
import pprint
from typing import List, TypedDict

from langchain.schema import HumanMessage
from pydantic import ValidationError

from orchestrator.agents.executor_agent import ExecutorAgent
from orchestrator.types.execute_result import ExecuteResult
from orchestrator.types.plan_execute_state import BestResponseBackup, State
from orchestrator.utils import oracle_repo as orep

model = 'gpt-4o-mini'  # define model type


class Output(TypedDict):
    model_type: str
    agent_type: str
    message: str


# TODO: change state to string, only necessary things are added
async def execute_step(
    state: State,
):
    current_task = state.plan[state.current_step]  # calls the sequence of steps

    n_models_executor = state.n_models_executor

    executor = ExecutorAgent(
        output_schema=ExecuteResult,
        model=model,
    )

    outputs: List[Output] = []

    try:
        for _ in range(n_models_executor):
            result = await executor.ainvoke(
                output_type=ExecuteResult,
                agent_scratchpad=', '.join(state.solutions_history),
                n_models=1,
                task=current_task,
            )

            outputs.append(
                {
                    'model_type': f'{model}',
                    'agent_type': 'llm',
                    'message': result.solution,
                }
            )

        feature_vector, best_response_body = orep.generate_feature_vector(
            outputs,
            state.problem_type,
            state.problem_scope,
            state.current_step,
            state.n_models_executor,
            state.best_response_backup,
            state.projected_time_to_finish,
        )

        updated_state = dataclasses.replace(state)

        updated_state.feature_vectors.append(feature_vector)
        updated_state.solutions_history.append(best_response_body)
        updated_state.messages = updated_state.messages + [
            HumanMessage(content=best_response_body)
        ]
        updated_state.best_response_backup = updated_state.best_response_backup + [
            BestResponseBackup(message=best_response_body)
        ]
        updated_state.current_step = updated_state.current_step + 1

        return updated_state
    except (Exception, ValidationError) as error:
        raise Exception(error) from error


async def run_as_main():
    input_message = 'How many players are allowed to play football simultaneously?'
    initial_state = State(
        messages=[HumanMessage(content=input_message)],
        system_task=input_message,
        plan=['How many players are allowed to play football simultaneously?'],
    )
    updated_state = await execute_step(state=initial_state)
    pprint.pp(updated_state)


def main():
    asyncio.run(run_as_main())


if __name__ == '__main__':
    main()
