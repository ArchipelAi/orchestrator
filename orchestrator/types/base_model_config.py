# base model that can be inherited by other models
from pydantic.v1 import BaseModel


class BaseModelConfig(BaseModel):
    class Config:
        extra = 'forbid'
        anystr_strip_whitespace = True
        use_enum_values = True
