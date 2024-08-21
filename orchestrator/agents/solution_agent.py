##debug code
from typing import Any, Dict, List, Optional, Union

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Solution(BaseModel):
    solution: Union[Dict[str, Any]]  # = Field(None, description='solution details')
    steps: Optional[List[str]] = (
        None  # = Field(None, description='steps to complete the task')
    )
    message: Optional[List[str]] = (
        None  # Field(None, description='general message or response')
    )

    class Config:
        arbitrary_types_allowed = True


class SolutionAgent:
    # Define Solution Agent Context
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
        n_models: int,
        model: str,
        temperature: int = 0,
        top_p: float = 0.1,
        agent_scratchpad: str = '',
    ):
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.n_models = n_models
        self.agent_scratchpad = agent_scratchpad
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=self.input_variables
        )
        self.output_parser = PydanticOutputParser(pydantic_object=Solution)

    def get_model(self):
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            model_kwargs={'top_p': self.top_p},
        )

    async def ainvoke(self, inputs):
        # debug_print(inputs, "SolutionAgent inputs")
        model = self.get_model()
        formatted_prompt = self.prompt.format(**inputs)
        messages = [
            SystemMessage(
                content="""You are one helpful agent in a multi-agent model of {n_models} agents.
            It is your job to provide constructive responses so your team of agents can work with to solve the task at hand."""
            ),
            HumanMessage(content=formatted_prompt),
        ]
        # debug_print(messages, "SolutionAgent messages")
        response = await model.agenerate([messages])
        # debug_print(response, "SolutionAgent raw response")

        response = response.dict()

        try:
            result = self.output_parser.parse(response['generations'][0][0]['text'])
            # print(result)
            return result
        except Exception:
            print('Exception parsing response:', response)
