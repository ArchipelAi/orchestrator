##debug code
from typing import Type

from orchestrator.models.base_agent import BaseAgent, T


class ExecutorAgent(BaseAgent):
    prompt_template = """
    Find the solution to the last system output: {task_step}.
    Provide a clear solution that has no list of steps. But is a clear instruction.
    Your answer should be geared towards solving the task at hand, provide a "solution:" object containing the designated answer.
    If you cannot provide a clear solution, clearly demarcate the .json object as "steps:" and list each with bulletpoints.
    If you cannot provide a solution, also provide a reason why you need substeps.

    Either answer with a solution or with steps.

    Here is the history of previous solutions: {agent_scratchpad}

    Use this history to inform your plan, ensuring you don't repeat steps that have already been completed.
    """

    input_variables = ['agent_scratchpad', 'n_models', 'task_step']

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
