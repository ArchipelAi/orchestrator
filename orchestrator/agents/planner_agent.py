from typing import Type

from orchestrator.agents.base_agent import BaseAgent, T


class PlannerAgent(BaseAgent):
    prompt_template = """
    You are one agent in a multi-agent model of {n_models} agents. 
    It is your job to provide constructive responses your team of agents can work with to solve the task at hand.
    Provide a first breakdown of tasks in bulletpoints '-' to plan how you would solve the challenge. Your answer should be geared towards solving the task at hand.  
    The current task is {task}. Provide .json output as a list of bulletpoints. Do not add anything extra.

    Here is the history of previous solutions: {agent_scratchpad}

    Use this history to inform your plan, ensuring you don't repeat steps that have already been completed.
    """

    input_variables = ['agent_scratchpad', 'n_models', 'task']

    def __init__(
        self,
        output_schema: Type[T],
        model: str,
        temperature: int = 0,
        top_p: float = 0.1,
    ):
        super().__init__(
            output_schema,
            model,
            self.prompt_template,
            self.input_variables,
            temperature,
            top_p,
        )
