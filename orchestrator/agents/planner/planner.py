#This is the lang graph implementation of the setup in agents.py

import asyncio
from dataclasses import replace

from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import ValidationError

from orchestrator.models.models import planner_model
from orchestrator.types.plan import Plan
from orchestrator.types.plan_execute_state import PlanEntry, PlanExecuteState

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ('human', '{objective}'),
    ]
)


model = planner_model.with_structured_output(schema=Plan, include_raw=False)

planner_chain = planner_prompt | model


async def plan_step(state: PlanExecuteState):
    response = await planner_chain.ainvoke({'objective': state.input})
    try:
        plan = Plan.validate(response)
        plan_entry_array = [PlanEntry(step=step, sub_steps=None) for step in plan.steps]
        return replace(state, plan=plan_entry_array)
    except ValidationError as ve:
        raise Exception(ve) from ve


async def run_as_main():
    test_state = PlanExecuteState(
        input='What country whon the European Championship in soccer in 2014?',
        plan=None,
        past_steps=None,
        response=None,
    )
    test_return = await plan_step(test_state)
    print(test_return)


def main():
    asyncio.run(run_as_main())


if __name__ == '__main__':
    main()
