# planner.py

import asyncio
import json
from dataclasses import replace
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError, BaseModel, Field
from orchestrator.models.sequence_planner import SequencePlanner
from orchestrator.models.solution_agent import SolutionAgent
from orchestrator.types.plan import Plan
from orchestrator.types.plan_execute_state import PlanEntry, PlanExecuteState
from langgraph.graph import Graph
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import HumanMessage
from orchestrator.agents.executor.executor import executor
from langgraph.graph import END

import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

n_models_planner = 1
n_models_executor = 2

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, HumanMessage):
            return {"__human_message__": True, "content": obj.content}
        if isinstance(obj, BaseModel):
            return obj.model_dump()  # Use model_dump() instead of dict()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)

def custom_json_decoder(dct):
    if "__human_message__" in dct:
        return HumanMessage(content=dct["content"])
    return dct

planner_prompt = ChatPromptTemplate.from_messages([
    ('system', """For the given objective, come up with a simple step by step plan. 
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
    Do not add any superfluous steps. 
    The result of the final step should be the final answer. 
    Make sure that each step has all the information needed - do not skip steps."""),
    ('human', '{objective}'),
])

class State(BaseModel):
    messages: List[HumanMessage] = Field(default_factory=list)
    system_task: str
    plan: List[str] = Field(default_factory=list)
    current_step: int = 0
    n_models_planner: int
    n_models_executor: int
    problem_type: str
    problem_scope: str
    best_response_backup: List[dict] = Field(default_factory=list)
    projected_time_to_finish: int = 0
    response_outputs: List[dict] = Field(default_factory=list)
    response_outputs_backup: List[List[dict]] = Field(default_factory=list)
    num_steps_proposed: int = 0
    feature_vectors: List = Field(default_factory=list)

async def planner(state: State) -> State:
    planner = SequencePlanner(
        n_models=1, 
        system_task=dict(state)['system_task'],
        output_schema=Plan, 
        model='gpt-3.5-turbo',
        agent_scratchpad=', '.join(state['solutions_history'])
    )
    #print(">>> STATE >>> :", state, " >>> ", type(state))
    response = await planner.agent_runnable.ainvoke({
        'agent_scratchpad': '',
        'n_models': 1,
        'system_task': dict(state)['system_task']

    })
    
    try:
        plan: Plan = Plan.parse_obj(response)
        state['plan'] = list(plan.message)
        state['num_steps_proposed'] = len(plan.message)
        updated_state = state

        return updated_state
    except ValidationError as ve:
        raise Exception(ve) from ve

workflow = Graph()
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "executor")
workflow.set_entry_point("planner")
app = workflow.compile()

async def run_workflow(input_message: str, n_models_planner: int = 1, n_models_executor: int = 2):
    initial_state = {
        "messages": [HumanMessage(content=input_message)],
        "system_task": input_message,
        "plan": [],
        "current_step": 0,
        "n_models_planner": n_models_planner,
        "n_models_executor": n_models_executor,
        "problem_type": "deterministic",
        "problem_scope": "open",
        "best_response_backup": [],
        "projected_time_to_finish": 0,
        "response_outputs": [],
        "response_outputs_backup": [],
        "num_steps_proposed": 0,
        "feature_vectors": [],
        "solutions_history": []
    }
    
    print("Initial state:", json.dumps(initial_state, cls=CustomJSONEncoder, indent=2))

    async for output in app.astream(initial_state):
        if isinstance(output, dict) and any(value == END for value in output.values()):
            print("Task completed.")
            break

        try:
            if isinstance(output, tuple) and len(output) == 2:
                state, feature_vector = output
                state['feature_vectors'] = state.get('feature_vectors', []) + [feature_vector]
            else:
                state = output
  
            #print("Final state for iteration:", json.dumps(state, cls=CustomJSONEncoder, indent=2))
            
            if 'messages' in state and len(state['messages']) > 1:
                print(f"Step {state.get('current_step', 'N/A')}: {state['messages'][-1].content}")
            
            #print(f"Current n_models_planner: {state.get('n_models_planner', 'N/A')}, "
                f"n_models_executor: {state.get('n_models_executor', 'N/A')}"
            
            if 'feature_vectors' in state and state['feature_vectors']:
                print(f"Latest feature vector: {state['feature_vectors'][-1]}")
        
        except Exception as e:
            #print(f"Error processing output: {e}")
            #print(f"Problematic output: {output}")
            import traceback
            #print(traceback.format_exc())

async def run_as_main():
    await run_workflow("Order a vegetarian pizza in Munich and have it delivered to Arcisstrasse 21, 80331 Munich. A human will pay the delivery person upon arrival.", #Invent rules for a board game with a scoring system on a 1-step scale and play that game until you reach score 5. Respond with 'FINISH' once a player has reached score 5. Do not ever mention the word 'FINISH' unless a player has reached 5 points.
                       n_models_planner=1, n_models_executor=2)

def main():
    asyncio.run(run_as_main())

if __name__ == "__main__":
    main()