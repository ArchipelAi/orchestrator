from pydantic.v1 import Field

from orchestrator.types.base_model_config import BaseModelConfig


class ExecuteResult(BaseModelConfig):
    """Solution to provided task."""

    solution: str = Field(description='Solution to task provided.')
