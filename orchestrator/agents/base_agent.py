from typing import List, Type, TypeVar

from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from orchestrator.types.base_model_config import BaseModelConfig

T = TypeVar('T', bound=BaseModelConfig)


class BaseAgent:
    def __init__(
        self,
        output_schema: Type[T],
        model: str,
        prompt_template: str,
        input_variables: List[str],
        temperature: int = 0,
        top_p: float = 0.1,
    ):
        self.llm = ChatOpenAI(
            model=model, temperature=temperature, model_kwargs={'top_p': top_p}
        )
        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=input_variables
        )
        self.model = self.llm.with_structured_output(
            schema=output_schema, include_raw=False
        )
        self.agent_runnable = self.prompt | self.model

    async def ainvoke(
        self,
        output_type: type[T],
        agent_scratchpad: str,
        n_models: int,
        task: str,
    ):
        try:
            response = await self.agent_runnable.ainvoke(
                {
                    'agent_scratchpad': agent_scratchpad,
                    'n_models': n_models,
                    'task': task,
                }
            )
            plan: T = output_type.parse_obj(response)

            return plan
        except (Exception, ValidationError) as error:
            if type(error) is Exception:
                print('Exception parsing response:', error)
            else:
                print('Validation error', error)
            raise Exception(error) from error
