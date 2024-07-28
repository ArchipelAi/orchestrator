# executor.py

from orchestrator.models.solution_agent import SolutionAgent
from orchestrator.agents.planner import oracle_repo as orep
from langchain_core.messages import HumanMessage
from langgraph.graph import END

async def executor(state):
    if state['current_step'] >= state['num_steps_proposed']:
        return END

    current_task = state['plan'][state['current_step'] - 1]
    print(f"Current task: {current_task}")
    n_models = state['n_models_executor']
    response_outputs = []

    for i in range(n_models):
        solution_agent = SolutionAgent(n_models=1, model=f'gpt-4o-mini')
        solution = await solution_agent.ainvoke({
            'agent_scratchpad': '',
            'n_models': 1,
            'task_step': current_task
        })
        
        output = {
            'model_type': f'gpt-4o-mini',
            'agent_type': f'executor',
            'message': str(solution)
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
        state['projected_time_to_finish']
    )

    updated_state = {
        **state,
        "messages": state["messages"] + [HumanMessage(content=best_response_body)],
        "best_response_backup": state['best_response_backup'] + [{'message': best_response_body}],
        "current_step": state['current_step'] + 1
    }

    return updated_state, feature_vector