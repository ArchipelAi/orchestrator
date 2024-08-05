from typing import List

from pydantic.v1 import Field

from orchestrator.types.base_model_config import BaseModelConfig


class Plan(BaseModelConfig):
    """Plan to follow in future"""

    message: List[str] = Field(
        description='different steps to follow, should be in sorted order'
    )
