from typing import Type

from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from orchestrator.types.plan import Plan


class SequencePlanner:
    # Define Sequence Planner Context
    prompt_template = """
    You are one agent in a multi-agent model of {n_models} agents. 
    It is your job to provide constructive responses your team of agents can work with to solve the task at hand.
    Provide a first breakdown of tasks in bulletpoints '-' to plan how you would solve the challenge. Your answer should be geared towards solving the task at hand.  
    The current task is {system_task}. Provide .json output as a list of bulletpoints. Do not add anything extra.

    Here is some useful information: {agent_scratchpad}
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
        self.llm = ChatOpenAI(model=model, temperature=temperature, top_p=top_p)  # type: ignore
        self.n_models = n_models
        self.agent_scratchpad = agent_scratchpad
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=self.input_variables
        )
        self.model = self.llm.with_structured_output(
            schema=output_schema, include_raw=False
        )
        self.agent_runnable = self.prompt | self.model

        # def plan(self, n_models, system_task, agent_scratchpad=''):
        # inputs = {
        #     'chat_history': [],  # Initial chat history
        #     'intermediate_steps': [],  # Initialize with an empty list
        #     'agent_scratchpad': agent_scratchpad,  # Initial value for agent_scratchpad
        #     'n_models': n_models,  # Number of models in the multi-agent system
        #     'system_task': system_task,
        # }
        # try:
        #     # Invoke the agent with the inputs
        #     response = self.agent_runnable.invoke(inputs)
        #     # print(response)
        #     # Extract the JSON content from the return_values attribute of the AgentFinish object
        #     response_json_str = response.return_values['output']

        #     # Clean the JSON string by removing the code block markers
        #     cleaned_json_str = response_json_str.strip('```json\n').strip('\n```')

        #     # Parse the cleaned JSON string
        #     response_dict = json.loads(cleaned_json_str)
        #     # Extract the list of tasks dynamically
        #     tasks_breakdown = None
        #     for value in response_dict.values():
        #         if isinstance(value, list) and all(
        #             isinstance(item, str) for item in value
        #         ):
        #             tasks_breakdown = value
        #             break
        # except Exception as e:
        #     print(f'Error during plan execution: {e}')
        #     return None

        # return tasks_breakdown
