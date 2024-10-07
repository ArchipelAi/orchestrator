##debug code
from typing import List, Optional, Type

from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

from orchestrator.agents.base_agent import BaseAgent, T


class ExecutorAgent(BaseAgent):
    prompt_template = """
    You are an assistant that can solve tasks by thinking through them step-by-step.
    You can also use tools if necessary.

    Find the solution to the last system output: {task}.
    Provide a clear solution that has no list of steps. But is a clear instruction.
    Your answer should be geared towards solving the task at hand.

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
        tools: Optional[List[Tool]] = None,
        use_tools: bool = False,
    ):
        self.use_tools = use_tools
        super().__init__(
            output_schema,
            model,
            self.prompt_template,
            self.input_variables,
            temperature,
            top_p,
        )

        if self.use_tools:
            self.python_tool = PythonREPL()

    def execute_code(self, code):
        # Uses the PythonREPLTool to execute code
        if not self.use_tools or not hasattr(self, 'python_tool'):
            raise ValueError('Tools are not enabled or python_tool is not initialized.')

        output = self.python_tool.run(code)
        return output
