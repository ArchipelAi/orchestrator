import dataclasses
import os
import sys
from typing import List, TypedDict

from langchain.schema import HumanMessage

from orchestrator.agents.code_assistant import CodeAssistant
from orchestrator.agents.executor_agent import ExecutorAgent
from orchestrator.agents.orchestrator_agent import Orchestrator
from orchestrator.types.execute_result import ExecuteResult
from orchestrator.types.plan_execute_state import BestResponseBackup, State
from orchestrator.utils import oracle_repo as orep
from orchestrator.utils.helper_functions import extract_list, save_state

model = 'gpt-4o-mini'  # define model type


class Output(TypedDict):
    model_type: str
    agent_type: str
    message: str


# TODO: change state to string, only necessary things are added
async def execute_step(
    state: State,
):
    # Check if the plan is empty
    if not state.plan:
        raise ValueError('The plan is empty. Cannot execute steps.')
    if state.current_step < len(state.plan):
        current_task = state.plan[
            state.current_step
        ]  # calls the current step, according to the sequence of steps
    else:
        current_task = None
        print(
            'The current step index is greater than the length of the plan. System is calculating next steps...'
        )
        orchestrator = Orchestrator()
        system_message = f'system_task: {state.system_task}; solutions_history: {(state.solutions_history)}'
        response = orchestrator.process_request(system_message=system_message)
        response = extract_list(response)
        for step in response:
            state.plan.append(step)
        state.current_step -= 1  # Reset the model to include the last step added

    # store current State status
    # extract cwd and save the state object to a file
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(current_directory, 'state_log.txt')
    save_state(state, filepath=filepath)

    n_models_executor = state.n_models_executor

    outputs: List[Output] = []

    if current_task and 'code_needed = true' in current_task.lower():
        print('Code is needed. Launching code agent.')
        code_assistant = CodeAssistant()

        try:
            code_response = await code_assistant.execute_message(
                {
                    'role': 'user',
                    'content': f"""
                        You are a coding assistant. 
                        Execute Python code to implement the following job towards solving "system_task: {state.system_task}":
                        Job: {current_task}
                        To implement the code and solve "system_task:", use the devised "solutions history: {state.solutions_history}" as reference. 
                        Your output is the final result of your code.
                        If you find that your code results complete "system_task: {state.system_task}", append "FINISH" to your output message.
                        """,
                    'tools': [{'type': 'code_interpreter'}],
                }
            )
            print('Code Assistant Response:', code_response)

            outputs.append(
                {
                    'model_type': f'{model}',
                    'agent_type': 'llm',  # coding_llm
                    'message': code_response,
                }
            )

            if 'finish' in outputs[0]['message'].lower():
                print(
                    f'The task has been completed. The final output is: {code_response}'
                )
                sys.exit(0)

        except Exception as error:
            raise Exception(f'Error during code execution: {error}') from error
    else:
        executor = ExecutorAgent(
            output_schema=ExecuteResult,
            model=model,
        )
        try:
            for _ in range(n_models_executor):
                result = await executor.ainvoke(
                    output_type=ExecuteResult,
                    agent_scratchpad=', '.join(state.solutions_history),
                    n_models=1,
                    task=state.plan[state.current_step],
                )

            outputs.append(
                {
                    'model_type': f'{model}',
                    'agent_type': 'llm',
                    'message': result.solution,
                }
            )
        except Exception as error:
            raise Exception(f'Error during agent invocation: {error}') from error

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
