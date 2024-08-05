from langchain_openai import ChatOpenAI

OPENAI_MODEL = ''

planner_model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
executor_model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
