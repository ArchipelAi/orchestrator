from typing import Type

from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import json
from orchestrator.types.plan import Plan
import os

class SequencePlanner:
    # Define Sequence Planner Context
    prompt_template = """
    You are one agent in a multi-agent model of {n_models} agents. 
    It is your job to provide constructive responses your team of agents can work with to solve the task at hand.
    Provide a first breakdown of tasks in bulletpoints '-' to plan how you would solve the challenge. Your answer should be geared towards solving the task at hand.  
    The current task is {system_task}. Provide .json output as a list of bulletpoints. Do not add anything extra.

    Here is the history of previous solutions: {agent_scratchpad}

    Use this history to inform your plan, ensuring you don't repeat steps that have already been completed.
    """

    input_variables = ['agent_scratchpad', 'n_models', 'system_task']

    # initialize Sequence Planner
    def __init__(
        self,
        n_models: int,
        system_task: str,
        output_schema: Type[Plan],
        model: str,
        temperature: int = 0,
        top_p: float = 0.1,
        agent_scratchpad: str = '',
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature, model_kwargs={"top_p": top_p})  # type: ignore
        self.n_models = n_models
        self.agent_scratchpad = agent_scratchpad
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=self.input_variables
        )
        self.model = self.llm.with_structured_output(
            schema=output_schema, include_raw=False
        )
        self.agent_runnable = self.prompt | self.model

        #debug
        # print("Model config:", json.dumps(self.model, indent=2, default=str))
        # print("Prompt template:", self.prompt.template)
        # print("Output schema:", output_schema.schema())

    async def agent_runnable_debug(self, *args, **kwargs):
        #print("agent_runnable input:", json.dumps(kwargs, indent=2, default=str))
        result = await self.agent_runnable.ainvoke(*args, **kwargs)
        #print("agent_runnable output:", json.dumps(result, indent=2, default=str))
        return result