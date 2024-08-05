import asyncio
import dataclasses
import pprint

from langchain.schema import HumanMessage

from orchestrator.agents.planner import oracle_repo as orep
from orchestrator.models.solution_agent import SolutionAgent
from orchestrator.types.plan_execute_state import BestResponseBackup, State

model = 'gpt-4o-mini'  # define model type


# TODO: change state to string, only necessary things are added
async def execute_step(
    state: State,
):
    current_task = state.plan[state.current_step]  # calls the sequence of steps

    print(f'Current task: {current_task}')

    n_models = state.n_models_executor
    response_outputs = []

    for _ in range(n_models):
        solution_agent = SolutionAgent(n_models=1, model=model)
        solution = await solution_agent.ainvoke(
            {
                'agent_scratchpad': ', '.join(state.solutions_history),
                'n_models': n_models,
                'task_step': current_task,
            }
        )

        output = {
            'model_type': f'{model}',
            'agent_type': 'llm',
            'message': str(solution),
        }
        response_outputs.append(output)

    feature_vector, best_response_body = orep.generate_feature_vector(
        response_outputs,
        state.problem_type,
        state.problem_scope,
        state.current_step,
        n_models,
        state.best_response_backup,
        state.projected_time_to_finish,
    )

    state.feature_vectors.append(feature_vector)
    state.solutions_history.append(best_response_body)

    updated_state = dataclasses.replace(state)

    updated_state.messages = updated_state.messages + [
        HumanMessage(content=best_response_body)
    ]
    updated_state.best_response_backup = updated_state.best_response_backup + [
        BestResponseBackup(message=best_response_body)
    ]
    updated_state.current_step = updated_state.current_step + 1

    return updated_state


async def run_as_main():
    input_message = (
        'From which country is the trainer of the Champions League winner 2014?'
    )
    initial_state = State(
        messages=[HumanMessage(content=input_message)], system_task=input_message
    )
    updated_state = await execute_step(state=initial_state)
    pprint.pp(updated_state)


def main():
    asyncio.run(run_as_main())


if __name__ == '__main__':
    main()
