####
### DEFINE ENVIRONMENT
####

import os

import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

os.environ['OPENAI_API_KEY'] = config['openai']['api_key']
# os.environ['TAVILY_API_KEY'] = config['tavily']['api_key']


# Initialize number of models
n_models = 2

# Initialize system_task
system_task = 'Order a vegetarian pizza in Munich and have it delivered to Arcisstraße 21, Munich. A human will pay the delivery person upon arrival.'

# initialize tools
tools = []

####
### SETUP FUNCTIONS
####


# INITALIZE DUMMY FUNCTION
def dummy_function(input):
    return 'This is a dummy function output'  # Dummy Tool to initialize simple agents


from langchain.tools import Tool

dummy_tool = Tool(
    name='dummy_tool',
    func=dummy_function,
    description='A dummy tool that does nothing and returns a placeholder value',
)

tools.append(dummy_tool)

# # INITIALIZE WEB SEARCH FUNCTION
# from langchain.utilities.tavily_search import TavilySearchAPIWrapper
# from langchain.tools.tavily_search import TavilySearchResults

# search = TavilySearchAPIWrapper()
# tavily_tool = TavilySearchResults(api_wrapper=search)

# tools.append(tavily_tool)


# EXECUTE TOOL (WIP)
def execute(task_step):
    """Find a solution to the given problem"""
    model: BaseOpenAI = ChatOpenAI(temperature=0)
    res = model.predict(f"""Find the solution to the last system output: {task_step}. Your answer should be geared towards solving the task at hand, provide a "solution:" object containing the designated answer. 
    If you find sub_steps are needed to solve the matter, clearly demarcate the .json object as "steps:" and list each with bulletpoints.""")
    return res.strip()


# INITIALIZE VARIABLE NAME TRACKER
class VariableNameTracker:
    def __init__(self):
        self.name_to_obj = {}

    def register(self, name, obj):
        self.name_to_obj[name] = obj

    def get_name(self, obj):
        for name, registered_obj in self.name_to_obj.items():
            if registered_obj is obj:
                return name
        return None


def update_request_parameters(best_response_body, num_step, response_outputs):
    """
    Update the request parameters for the next step.

    Parameters:
    best_response_body (str): The body of the best response message.
    num_step (int): The current step number.
    response_outputs (list): The list of response outputs.

    Returns:
    tuple: Updated task_step, num_steps_proposed, num_step, response_outputs_backup, response_outputs
    """
    task_step = best_response_body[num_step]
    num_steps_proposed = len(best_response_body)
    num_step += 1
    response_outputs_backup.append(response_outputs)
    response_outputs = []

    return (
        task_step,
        num_steps_proposed,
        num_step,
        response_outputs_backup,
        response_outputs,
    )


####
### DEFINE AGENTS
####

# INITIALIZE SEQUENCE PLANNER AGENT

import json
from typing import Any, Dict, List, Optional

from langchain.agents import create_openai_functions_agent
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field


class Solution(BaseModel):
    solution: Optional[Dict[str, Any]] = Field(None, description='solution details')
    steps: Optional[List[str]] = Field(None, description='steps to complete the task')
    message: Optional[List[str]] = Field(
        None, description='general message or response'
    )

    class Config:
        arbitrary_types_allowed = True


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
    def __init__(self, llm, tools, n_models, system_task):
        self.llm = llm
        self.tools = tools
        self.n_models = n_models
        self.task_step = task_step
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=self.input_variables
        )
        self.prompt.format(
            n_models=n_models, system_task=system_task, agent_scratchpad=''
        )
        self.agent_runnable = create_openai_functions_agent(llm, tools, self.prompt)

    def plan(self, n_models, system_task, agent_scratchpad=''):
        inputs = {
            'chat_history': [],  # Initial chat history
            'intermediate_steps': [],  # Initialize with an empty list
            'agent_scratchpad': agent_scratchpad,  # Initial value for agent_scratchpad
            'n_models': n_models,  # Number of models in the multi-agent system
            'system_task': system_task,
        }
        try:
            # Invoke the agent with the inputs
            response = self.agent_runnable.invoke(inputs)
            # print(response)
            # Extract the JSON content from the return_values attribute of the AgentFinish object
            response_json_str = response.return_values['output']

            # Clean the JSON string by removing the code block markers
            cleaned_json_str = response_json_str.strip('```json\n').strip('\n```')

            # Parse the cleaned JSON string
            response_dict = json.loads(cleaned_json_str)
            # Extract the list of tasks dynamically
            tasks_breakdown = None
            for value in response_dict.values():
                if isinstance(value, list) and all(
                    isinstance(item, str) for item in value
                ):
                    tasks_breakdown = value
                    break
        except Exception as e:
            print(f'Error during plan execution: {e}')
            return None

        return tasks_breakdown


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
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            model_kwargs={'top_p': self.top_p},
        )

    async def ainvoke(self, inputs):
        model = self.get_model()
        formatted_prompt = self.prompt.format(**inputs)
        messages = [
            SystemMessage(
                content="""You are one helpful agent in a multi-agent model of {n_models} agents.
            It is your job to provide constructive responses so your team of agents can work with to solve the task at hand."""
            ),
            HumanMessage(content=formatted_prompt),
        ]
        response = await model.agenerate([messages])
        print(response)
        return self.output_parser.parse(response.generations[0][0].text)


####
### DEFINE BENCHMARKING TESTS
####

import oracle_repo as orep


def generate_feature_vector(
    response_outputs,
    problem_type,
    problem_scope,
    num_step,
    n_models,
    best_response_backup,
    projected_time_to_finish,
):
    # Initialize fitness scores list
    fitness_scores = []

    # Calculate fitness scores for each response
    for response in response_outputs:
        ag_rationality, model_type_label, agent_type_label = orep.evaluate_fitness(
            response['model_type'], response['agent_type'], problem_type
        )
        fitness_scores.append(ag_rationality + len(response['message']))

    # Calculate consensus score
    consensus_score = orep.calculate_consensus_score(response_outputs)

    # Calculate action maximization scores for each response
    act_max_scores = []
    for fitness in fitness_scores:
        ag_rationality = fitness
        act_max_score = orep.calculate_act_max_score(ag_rationality, consensus_score)
        act_max_scores.append(act_max_score)

    # Determine the best response
    best_response_index = act_max_scores.index(max(act_max_scores))
    best_response = response_outputs[best_response_index]
    outputs = {
        'index': num_step,
        'model_type': best_response['model_type'],
        'agent_type': best_response['agent_type'],
        'act_max_score': max(act_max_scores),
        'message': best_response['message'],
    }

    best_response_backup.append(outputs)
    best_response_body = best_response['message']

    # Evaluate error rate
    error_rate = orep.evaluate_error_rate(
        best_response_body, best_response_backup, responses=None, num_step=num_step
    )
    if num_step == 0:
        projected_time_to_finish += len(best_response_body)
    else:
        pass
    # Evaluate environment complexity
    problem_scope_label, problem_type_label, ratio_steps_left = (
        orep.evaluate_environment_complexity(
            problem_scope,
            problem_type,
            num_step,
            n_models,
            best_response_body,
            projected_time_to_finish,
        )
    )

    # Generate feature vector
    feature_vector = (
        max(act_max_scores),
        ag_rationality,
        model_type_label,
        agent_type_label,
        problem_scope_label,
        problem_type_label,
        num_step,
        n_models,
        projected_time_to_finish,
        ratio_steps_left,
        error_rate,
    )

    return feature_vector, best_response_body


####
### RUNNING SCRIPT (to be converted into langgraph format and ported to planner.py)
####

# Phase 1: Planning

# INITIALIZE DATABASES
response_outputs_backup = []
training_data = []

# GENERATE A RESPONSE OBJECT
response_outputs = []

# STORE SEQUENCE OF BEST_RESPONSES AS BACKUP
best_response_backup = []

# INITIALIZE TIME
num_step = 0
adaptation_rate = 0
projected_time_to_finish = 0

problem_type = 'deterministic'
problem_scope = 'open'

# Initialize the tracker
tracker = VariableNameTracker()

# Initialize the language model
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, top_p=0.1)
# Register the variable
tracker.register('llm', llm)

for i in range(n_models):
    # Initialize the SequencePlanner agent
    sequence_planner = SequencePlanner(
        llm, tools, n_models, system_task
    )  # can be modified to change llm type with alternating i

    tasks_breakdown = sequence_planner.plan(n_models, system_task)
    if tasks_breakdown:
        model_type = llm.model_name
        agent_type = tracker.get_name(llm)
        output = {
            'model_type': model_type,
            'agent_type': agent_type,
            'message': tasks_breakdown,
        }
        response_outputs.append(output)
    else:
        print(f'Error in generating breakdown for provided task: {system_task}')

# RUN BENCHMARKS ON OUTPUT

feature_vector, best_response_body = generate_feature_vector(
    response_outputs,
    problem_type,
    problem_scope,
    num_step,
    n_models,
    best_response_backup,
    projected_time_to_finish,
)

training_data.append(feature_vector)

# UPDATE REQUEST PARAMETERS
task_step, num_steps_proposed, num_step, response_outputs_backup, response_outputs = (
    update_request_parameters(best_response_body, num_step, response_outputs)
)


# Phase 2: Execution

for i in range(n_models):
    # Initialize Task Execution Agent
    solution_agent = SolutionAgent(
        llm, tools, n_models, task_step
    )  # can be modified to change llm type with alternating i
    task_execution = solution_agent.plan(n_models, task_step)
    if task_execution:
        model_type = llm.model_name
        agent_type = tracker.get_name(llm)
        output = {
            'model_type': model_type,
            'agent_type': agent_type,
            'message': task_execution,
        }
        response_outputs.append(output)
    else:
        print(f'Error in generating breakdown for provided task: {task_step}')
