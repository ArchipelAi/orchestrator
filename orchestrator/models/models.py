import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

#added this as key was not loaded 
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

OPENAI_MODEL = 'gpt-4o-mini'

planner_model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
executor_model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
