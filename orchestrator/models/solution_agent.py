from typing import List

from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field

from orchestrator.types.base_model_config import BaseModelConfig


class Solution(BaseModelConfig):
    """Solution to task"""

    message: List[str] = Field(description='solution to task')


class SolutionAgent:
    # Define Solution Agent Context
    prompt_template = """
    You are one agent in a multi-agent model of {n_models} agents.
    It is your job to provide constructive responses your team of agents can work with to solve the task at hand.
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
        self.llm = ChatOpenAI(model=model, temperature=temperature, top_p=top_p)  # type: ignore
        self.n_models = n_models
        self.agent_scratchpad = agent_scratchpad
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=self.input_variables
        )
        self.model = self.llm.with_structured_output(schema=Solution, include_raw=False)
        self.agent_runnable = self.prompt | self.model
