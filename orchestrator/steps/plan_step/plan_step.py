import asyncio
import dataclasses
import pprint

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from orchestrator.agents.planner_agent import PlannerAgent
from orchestrator.types.plan import Plan
from orchestrator.types.plan_execute_state import State

model = 'gpt-3.5-turbo'


async def plan_step(state: State) -> State:
    planner = PlannerAgent(
        output_schema=Plan,
        model=model,
    )
    try:
        plan = await planner.ainvoke(
            output_type=Plan,
            agent_scratchpad=', '.join(state.solutions_history),
            n_models=1,
            task=state.system_task,
        )

        updated_state = dataclasses.replace(state)
        updated_state.plan = list(plan.message)
        updated_state.num_steps_proposed = len(plan.message)

        return updated_state
    except (Exception, ValidationError) as error:
        raise Exception(error) from error


async def run_as_main():
    input_message = (
        'From which country is the trainer of the Champions League winner 2014?'
    )
    initial_state = State(
        messages=[HumanMessage(content=input_message)], system_task=input_message
    )
    updated_state = await plan_step(state=initial_state)
    pprint.pp(updated_state)


def main():
    asyncio.run(run_as_main())


if __name__ == '__main__':
    main()
