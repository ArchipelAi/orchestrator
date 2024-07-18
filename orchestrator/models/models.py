from langchain_openai import ChatOpenAI

OPENAI_MODEL = 'gpt-4o'

ChatOpenAI

plannerModel = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
