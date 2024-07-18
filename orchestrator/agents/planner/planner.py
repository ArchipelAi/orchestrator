import asyncio
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field

from orchestrator.models.models import plannerModel


class PlanEntry(BaseModel):
    step: str
    sub_steps: Optional['PlanEntry']


class PastSteps(BaseModel):
    step: str
    solution: str


class PlanExecuteState(BaseModel):
    input: str
    plan: List[PlanEntry]
    past_steps: List[PastSteps]
    response: Optional[str]


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description='different steps to follow, should be in sorted order'
    )


class PlanFunction(BaseModel):
    """This tool is used to plan the steps to follow."""

    name: str = 'plan'
    description: str = 'This tool is used to plan the steps to follow'
    parameters: Plan


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

model = plannerModel.with_structured_output(PlanFunction)

planner_chain = planner_prompt | model


# async def plan_step(state: PlanExecuteState):
#     response = await planner_chain.ainvoke({'objective': state.input})
#     plan = Plan(**response)  # type: ignore
#     mapped_plan = [PlanEntry(step=step, sub_steps=None) for step in plan.steps]
#     return {'plan': mapped_plan}
async def plan_step(state: str):
    response = await planner_chain.ainvoke({'objective': state})
    plan = Plan(**response)  # type: ignore
    mapped_plan = [PlanEntry(step=step, sub_steps=None) for step in plan.steps]
    return {'plan': mapped_plan}


async def main():
    test_return = await plan_step(
        'What country whon the European Championship in soccer in 2014?'
    )
    print(test_return)


if __name__ == '__main__':
    asyncio.run(main())
