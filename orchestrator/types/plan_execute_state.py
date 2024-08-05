from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage


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


@dataclass
class BestResponseBackup:
    message: Any


@dataclass
class State:
    messages: List[HumanMessage]
    system_task: str
    plan: List[str] = field(default_factory=list)
    current_step: int = 0
    n_models_planner: int = 1
    n_models_executor: int = 1
    # TODO: add enums for problem type and problem scope
    problem_type: str = 'deterministic'
    problem_scope: str = 'open'
    best_response_backup: List[BestResponseBackup] = field(default_factory=list)
    projected_time_to_finish: int = 0
    response_outputs: List[str] = field(default_factory=list)
    response_outputs_backup: List[List[str]] = field(default_factory=list)
    num_steps_proposed: int = 0
    # TODO: add better type for feature vectors in orep function
    feature_vectors: List[Any] = field(default_factory=list)
    solutions_history: List[str] = field(default_factory=list)
