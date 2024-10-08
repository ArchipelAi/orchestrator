##debug code
from typing import Type

from orchestrator.agents.base_agent import BaseAgent, T


class CodingAgent(BaseAgent):
    prompt_template = """
            You are a coding assistant. Provide code to implement the following step in Python:

            Step: {step}

            Provide the code only. Ensure the code is properly formatted so it can be directly executed in a python shell.
            """
    input_variables = ['step']

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

        self.chain = self.prompt | self.model

    async def generate_code(self, step):
        code_response = await self.chain.ainvoke({'step': step})

        print(code_response)
        # Extract code between ```python and ```
        # code = code_response.split('```python')[1].split('```')[0].strip()
        return code_response

    def set_chain(self, chain):
        self.chain = chain
