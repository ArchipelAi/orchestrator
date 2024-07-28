# planner.py

import asyncio
from dataclasses import replace
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import ValidationError
from orchestrator.models.sequence_planner import SequencePlanner
from orchestrator.types.plan import Plan
from orchestrator.types.plan_execute_state import PlanEntry, PlanExecuteState
from langgraph.graph import Graph
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage
from executor import executor

planner_prompt = ChatPromptTemplate.from_messages([
    ('system', """For the given objective, come up with a simple step by step plan. 
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
    Do not add any superfluous steps. 
    The result of the final step should be the final answer. 
    Make sure that each step has all the information needed - do not skip steps."""),
    ('human', '{objective}'),
])

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

async def planner(state: State) -> State:
    planner = SequencePlanner(
        n_models=1, 
        system_task=state['messages'][-1].content, 
        output_schema=Plan, 
        model='gpt-3.5-turbo'
    )
    response = await planner.agent_runnable.ainvoke({
        'agent_scratchpad': '',
        'n_models': 1,
        'system_task': state['messages'][-1].content
    })
    
    try:
        plan: Plan = Plan.validate(response)
        plan_entry_array = [PlanEntry(step=step, sub_steps=None) for step in plan.message]
        
        return {
            **state,
            "plan": plan_entry_array,
            "num_steps_proposed": len(plan_entry_array)
        }
    except ValidationError as ve:
        raise Exception(ve) from ve

workflow = Graph()
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "planner")
workflow.set_entry_point("planner")
app = workflow.compile()

async def run_workflow(input_message: str, initial_n_models_planner: int = 1, initial_n_models_executor: int = 2):
    initial_state = {
        "messages": [HumanMessage(content=input_message)],
        "plan": [],
        "current_step": 0,
        "n_models_planner": initial_n_models_planner,
        "n_models_executor": initial_n_models_executor,
        "problem_type": "deterministic",
        "problem_scope": "open",
        "best_response_backup": [],
        "projected_time_to_finish": 0,
        "response_outputs": [],
        "response_outputs_backup": [],
        "num_steps_proposed": 0
    }
    
    async for output in app.astream(initial_state):
        if 'messages' in output and len(output['messages']) > 1:
            print(f"Step {output['current_step']}: {output['messages'][-1].content}")
        print(f"Current n_models_planner: {output['n_models_planner']}, n_models_executor: {output['n_models_executor']}")

async def run_as_main():
    await run_workflow("Order a vegetarian pizza in Munich and have it delivered to Arcisstraße 21, Munich. A human will pay the delivery person upon arrival.", 
                       initial_n_models_planner=1, initial_n_models_executor=2)

def main():
    asyncio.run(run_as_main())

if __name__ == "__main__":
    main()