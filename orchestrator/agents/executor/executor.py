# executor.py

from orchestrator.models.solution_agent import SolutionAgent
from orchestrator.agents.planner import oracle_repo as orep
from langchain_core.messages import HumanMessage
from langgraph.graph import END
from pydantic import BaseModel, Field
from typing import Union, Dict, Any

##debug code
import json

def debug_print(obj, name):
    def default_serializer(o):
        if isinstance(o, BaseModel):
            return f"<Pydantic Model: {o.__class__.__name__}>"
        return f"<non-serializable: {type(o).__name__}>"

    try:
        print(f"Debug {name}:", json.dumps(obj, default=default_serializer, indent=2))
    except Exception as e:
        print(f"Error serializing {name}:", str(e))
        if isinstance(obj, dict):
            for k, v in obj.items():
                print(f"  {k}:", type(v).__name__)
                if isinstance(v, BaseModel):
                    print(f"    Pydantic Model found in {name}.{k}: {v.__class__.__name__}")
        elif isinstance(obj, BaseModel):
            print(f"  Object is a Pydantic Model: {obj.__class__.__name__}")
            for field_name, field_value in obj:
                print(f"    {field_name}: {type(field_value).__name__}")

##end debug code

async def executor(state: Union[tuple, Dict[str, Any], str]) -> Union[tuple, Dict[str, Any], str]:

    if isinstance(state, str) and state == END:
        return END

    if isinstance(state, tuple):
        state = state[0]

    if state['current_step'] >= state['num_steps_proposed']:
        return END
    current_task = state['plan'][state['current_step']] #calls the sequence of steps
    #debug_print(current_task, "current_task")
    print(f"Current task: {current_task}")
    n_models = state['n_models_executor']
    response_outputs = []
    for i in range(n_models):
        solution_agent = SolutionAgent(n_models=1, model=f'gpt-3.5-turbo')
        solution = await solution_agent.ainvoke({
            'agent_scratchpad': '',
            'n_models': 1,
            'task_step': current_task
        })
        #debug_print(solution, f"solution_{i}")
        output = {
            'model_type': f'gpt-3.5-turbo',
            'agent_type': f'llm',
            'message': str(solution)
        }
        response_outputs.append(output)
        #debug_print(output, f"output_{i}")
    #debug_print(response_outputs, "response_outputs")
    feature_vector, best_response_body = orep.generate_feature_vector(
        response_outputs,
        state['problem_type'],
        state['problem_scope'],
        state['current_step'],
        n_models,
        state['best_response_backup'],
        state['projected_time_to_finish']
    )
    debug_print(feature_vector, "feature_vector")
    #debug_print(best_response_body, "best_response_body")
    updated_state = {
        **state,
        "messages": state["messages"] + [HumanMessage(content=best_response_body)],
        "best_response_backup": state['best_response_backup'] + [{'message': best_response_body}],
        "current_step": state['current_step'] + 1
    }
    debug_print(updated_state, "updated_state")
   
    return updated_state, feature_vector

__all__ = ['executor']