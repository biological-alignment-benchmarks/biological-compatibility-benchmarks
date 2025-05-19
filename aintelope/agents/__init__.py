# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

from typing import Mapping, Type
from aintelope.agents.abstract_agent import Agent
from aintelope.agents.example_agent import ExampleAgent

from aintelope.agents.random_agent import RandomAgent
from aintelope.agents.handwritten_rules_agent import HandwrittenRulesAgent

from aintelope.agents.q_agent import QAgent

# SB3 Discrete action space models
from aintelope.agents.ppo_agent import PPOAgent
from aintelope.agents.dqn_agent import DQNAgent
from aintelope.agents.a2c_agent import A2CAgent

from aintelope.agents.llm_agent import LLMAgent

from aintelope.agents.simple_agents import (
    # IterativeWeightOptimizationAgent,
    # OneStepPerfectPredictionAgent,
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
# register_agent_class("one_step_perfect_prediction_agent", OneStepPerfectPredictionAgent)
# register_agent_class(
#    "iterative_weight_optimization_agent", IterativeWeightOptimizationAgent
# )

register_agent_class("q_agent", QAgent)
register_agent_class("example_agent", ExampleAgent)
register_agent_class("random_agent", RandomAgent)
register_agent_class("handwritten_rules_agent", HandwrittenRulesAgent)

register_agent_class("sb3_ppo_agent", PPOAgent)
register_agent_class("sb3_dqn_agent", DQNAgent)
register_agent_class("sb3_a2c_agent", A2CAgent)

register_agent_class("llm_agent", LLMAgent)
