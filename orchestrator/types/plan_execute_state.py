from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PlanEntry:
    step: str
    sub_steps: Optional['PlanEntry']


@dataclass
class PastSteps:
    step: str
    solution: str


@dataclass
class PlanExecuteState:
    input: str
    plan: Optional[List[PlanEntry]]
    past_steps: Optional[List[PastSteps]]
    response: Optional[str]
