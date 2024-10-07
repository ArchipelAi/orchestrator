##debug code
from typing import Type

from orchestrator.agents.base_agent import BaseAgent, T


class CodingAgent(BaseAgent):
    prompt_template = """
            You are a code assistant agent. Generate the necessary Python code to execute the following step:

            Step: {step}

            Provide only the code, without any explanations.

            Code:
            ```python
            # Your code here
            ```
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
        code = code_response.split('```python')[1].split('```')[0].strip()
        return code

    def set_chain(self, chain):
        self.chain = chain
