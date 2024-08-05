import asyncio
import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, Graph
from pydantic import BaseModel, Field

from orchestrator.agents.planner import oracle_repo as orep
from orchestrator.models.planner_agent import SequencePlanner
from orchestrator.models.solution_agent import SolutionAgent
from orchestrator.types.plan import Plan


class State(TypedDict):
    messages: Annotated[List[HumanMessage], operator.add]
    plan: List[str]
    current_step: int
    n_models_planner: int
    n_models_executor: int
    problem_type: str
    problem_scope: str
    best_response_backup: List[Dict[str, Any]]
    projected_time_to_finish: int
    response_outputs: List[Dict[str, Any]]
    response_outputs_backup: List[List[Dict[str, Any]]]
    num_steps_proposed: int


class Plan(BaseModel):
    message: List[str] = Field(description='List of plan steps')

    class Config:
        title = 'Plan'
        description = 'A concise plan structure with steps to accomplish a task.'


async def planner(state: State) -> State:
    n_models = state['n_models_planner']
    response_outputs = []

    for i in range(n_models):
        planner = SequencePlanner(
            n_models=1,
            system_task=state['messages'][-1].content,
            output_schema=Plan,
            model='gpt-3.5-turbo',
        )
        response = await planner.agent_runnable.ainvoke(
            {
                'agent_scratchpad': '',
                'n_models': 1,
                'system_task': state['messages'][-1].content,
            }
        )
        plan: Plan = Plan.model_validate(response)

        output = {
            'model_type': 'gpt-3.5-turbo',
            'agent_type': 'planner',
            'message': plan.message,
        }
        response_outputs.append(output)
        print(output)

    feature_vector, best_response_body = orep.generate_feature_vector(
        response_outputs,
        state['problem_type'],
        state['problem_scope'],
        state['current_step'],
        n_models,
        state['best_response_backup'],
        state['projected_time_to_finish'],
    )

    (
        task_step,
        num_steps_proposed,
        new_step,
        response_outputs_backup,
        new_response_outputs,
    ) = orep.update_request_parameters(
        best_response_body, state['current_step'], response_outputs
    )

    return {
        **state,
        'plan': best_response_body,
        'current_step': new_step,
        'projected_time_to_finish': state['projected_time_to_finish']
        + len(best_response_body),
        'response_outputs': new_response_outputs,
        'response_outputs_backup': response_outputs_backup,
        'num_steps_proposed': num_steps_proposed,
    }


async def executor(state: State) -> State:
    if state['current_step'] >= state['num_steps_proposed']:
        return END

    current_task = state['plan'][state['current_step'] - 1]
    print(f'Current task: {current_task}')
    n_models = state['n_models_executor']
    response_outputs = []

    for i in range(n_models):
        solution_agent = SolutionAgent(n_models=1, model='gpt-4o-mini')
        solution = await solution_agent.ainvoke(
            {'agent_scratchpad': '', 'n_models': 1, 'task_step': current_task}
        )

        output = {
            'model_type': 'gpt-4o-mini',
            'agent_type': 'executor',
            'message': str(solution),
        }
        response_outputs.append(output)
        print(output)

    feature_vector, best_response_body = orep.generate_feature_vector(
        response_outputs,
        state['problem_type'],
        state['problem_scope'],
        state['current_step'],
        n_models,
        state['best_response_backup'],
        state['projected_time_to_finish'],
    )

    return {
        **state,
        'messages': state['messages'] + [HumanMessage(content=best_response_body)],
        'best_response_backup': state['best_response_backup']
        + [{'message': best_response_body}],
    }


workflow = Graph()
workflow.add_node('planner', planner)
workflow.add_node('executor', executor)
workflow.add_edge('planner', 'executor')
workflow.add_edge('executor', 'planner')
workflow.set_entry_point('planner')
app = workflow.compile()


async def run_workflow(
    input_message: str,
    initial_n_models_planner: int = 1,
    initial_n_models_executor: int = 2,
):
    initial_state = {
        'messages': [HumanMessage(content=input_message)],
        'plan': [],
        'current_step': 0,
        'n_models_planner': initial_n_models_planner,
        'n_models_executor': initial_n_models_executor,
        'problem_type': 'deterministic',
        'problem_scope': 'open',
        'best_response_backup': [],
        'projected_time_to_finish': 0,
        'response_outputs': [],
        'response_outputs_backup': [],
        'num_steps_proposed': 0,
    }

    async for output in app.astream(initial_state):
        if 'messages' in output and len(output['messages']) > 1:
            print(f"Step {output['current_step']}: {output['messages'][-1].content}")
        print(
            f"Current n_models_planner: {output['n_models_planner']}, n_models_executor: {output['n_models_executor']}"
        )


async def run_as_main():
    await run_workflow(
        'Order a vegetarian pizza in Munich and have it delivered to Arcisstraße 21, Munich. A human will pay the delivery person upon arrival.',
        initial_n_models_planner=1,
        initial_n_models_executor=2,
    )


def main():
    asyncio.run(run_as_main())


if __name__ == '__main__':
    main()
