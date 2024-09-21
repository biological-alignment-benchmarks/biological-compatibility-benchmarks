from typing import Mapping, Type
from aintelope.agents.abstract_agent import Agent
from aintelope.agents.example_agent import ExampleAgent

from aintelope.agents.q_agent import QAgent
from aintelope.agents.simple_agents import (
    IterativeWeightOptimizationAgent,
    OneStepPerfectPredictionAgent,
    RandomWalkAgent,
)

AGENT_REGISTRY: Mapping[str, Type[Agent]] = {}


def register_agent_class(agent_id: str, agent_class: Type[Agent]):
    if agent_id in AGENT_REGISTRY:
        raise ValueError(f"{agent_id} is already registered")
    AGENT_REGISTRY[agent_id] = agent_class


def get_agent_class(agent_id: str) -> Type[Agent]:
    if agent_id not in AGENT_REGISTRY:
        raise ValueError(f"{agent_id} is not found in agent registry")
    return AGENT_REGISTRY[agent_id]


# add agent class to registry
register_agent_class("random_walk_agent", RandomWalkAgent)
register_agent_class("one_step_perfect_prediction_agent", OneStepPerfectPredictionAgent)
register_agent_class(
    "iterative_weight_optimization_agent", IterativeWeightOptimizationAgent
)
register_agent_class("q_agent", QAgent)
register_agent_class("example_agent", ExampleAgent)
