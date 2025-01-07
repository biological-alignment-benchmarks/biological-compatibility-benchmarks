# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

"""
Collection of simple (rule based) agents, e.g. a completely random agent
"""

import logging
import random

import numpy as np

from aintelope.agents.q_agent import QAgent

logger = logging.getLogger("aintelope.agents.simple_agents")

# numerical constants
EPS = 0.0001
INF = 9999999999


class RandomWalkAgent(QAgent):
    def get_action(self, *args, **kwargs) -> int:
        return self.trainer.action_space.sample()


# TODO: these agents are currently not used

# class OneStepPerfectPredictionAgent(QAgent):
#    def get_action(self, epsilon: float, device: str) -> int:
#        """Using the given network, decide what action to carry out using an
#        epsilon-greedy policy.

#        Args:
#            epsilon: value to determine likelihood of taking a random action
#            device: current device

#        Returns:
#            action
#        """
#        if self.done:
#            return None
#        elif np.random.random() < epsilon:
#            # GYM_INTERACTION
#            action = self.trainer.action_space.sample()
#        else:
#            # TODO
#            action = self.trainer.action_space.sample()
#        return action


# class IterativeWeightOptimizationAgent(QAgent):
#    def reset(self) -> None:
#        """Resets the environment and updates the state."""
#        self.done = False
#        self.action_weights = np.repeat([1.0], self.action_space(self.id).n)
#        # GYM_INTERACTION
#        self.state = self.env.reset()
#        if isinstance(self.state, tuple):
#            self.state = self.state[0]

#    def get_action(self, epsilon: float, device: str) -> int:
#        MIN_WEIGHT = 0.05
#        learning_rate = 0.01
#        learning_randomness = 0.00

#        if np.random.random() < epsilon:
#            # GYM_INTERACTION
#            action = self.action_space.sample()
#            return action

#        recent_memories = self.replay_buffer.fetch_recent_memories(2)

#        logger.info("info", recent_memories)

#        reward = self.replay_buffer.get_reward_from_memory(recent_memories[0])
#        previous_reward = self.replay_buffer.get_reward_from_memory(recent_memories[1])
#        last_action = self.replay_buffer.get_action_from_memory(recent_memories[0])

#        # avoid big weight change on the first valid step
#        if last_action is not None and previous_reward > EPS:
#            last_action_reward_delta = reward - previous_reward
#            last_action_weight = self.action_weights[last_action]
#            logger.info(
#                "dreward",
#                last_action_reward_delta,
#                last_action,
#            )
#            last_action_weight += last_action_reward_delta * learning_rate
#            last_action_weight = max(MIN_WEIGHT, last_action_weight)
#            self.action_weights[last_action] = last_action_weight
#            logger.info("action_weights", self.action_weights)

#            weight_sum = np.sum(self.action_weights)
#            self.action_weights /= weight_sum

#        def cdf(ds):
#            res = {}
#            x = 0
#            for k, v in ds:
#                x += v
#                res[k] = x
#            for k in res:
#                res[k] /= x
#            return res

#        def choose(cdf):
#            assert cdf
#            x = random.uniform(0, 1 - EPS)
#            k = None
#            for k, v in cdf.items():
#                if x >= v:
#                    return k
#            return k

#        action_weights_cdf = cdf(enumerate(self.action_weights))
#        logger.info(
#            "cdf",
#            ", ".join([f"{iaction}: {w}" for iaction, w in action_weights_cdf.items()]),
#        )

#        logger.info(action_weights_cdf)
#        action = choose(action_weights_cdf)
#        if random.uniform(0, 1) < learning_randomness:
#            action = self.action_space.sample()
#        logger.info("chose action", action)
#        return action
