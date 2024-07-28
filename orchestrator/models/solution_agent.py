from typing import Any, Dict, List, Optional

from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel
from orchestrator.types.base_model_config import BaseModelConfig
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser


class Solution(BaseModel):
    solution: Optional[Dict[str, Any]] = Field(None, description='solution details')
    steps: Optional[List[str]] = Field(None, description='steps to complete the task')
    message: Optional[List[str]] = Field(None, description='general message or response')

    class Config:
        arbitrary_types_allowed = True


class SolutionAgent:
    # Define Solution Agent Context
    prompt_template = """
    Find the solution to the last system output: {task_step}. Your answer should be geared towards solving the task at hand, provide a "solution:" object containing the designated answer. 
    If you find sub_steps are needed to solve the matter, clearly demarcate the .json object as "steps:" and list each with bulletpoints. 

    Here is some useful information: {agent_scratchpad}
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
        return ChatOpenAI(model=self.model_name, temperature=self.temperature, model_kwargs={"top_p": self.top_p})

    async def ainvoke(self, inputs):
        model = self.get_model()
        formatted_prompt = self.prompt.format(**inputs)
        messages = [
            SystemMessage(content="""You are one helpful agent in a multi-agent model of {n_models} agents.
            It is your job to provide constructive responses so your team of agents can work with to solve the task at hand."""),
            HumanMessage(content=formatted_prompt)
        ]
        response = await model.agenerate([messages])
        print(response)
        return self.output_parser.parse(response.generations[0][0].text)
