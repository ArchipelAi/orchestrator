from langchain_openai import ChatOpenAI

OPENAI_MODEL = 'gpt-4o-mini'

planner_model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
executor_model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
