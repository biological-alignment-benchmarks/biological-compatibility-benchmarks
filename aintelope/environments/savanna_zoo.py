import logging
from typing import Dict, Optional

from pettingzoo import AECEnv, ParallelEnv

from aintelope.environments import register_env_class

from aintelope.environments.savanna import (
    SavannaEnv,
    Action,
    Step,
)

logger = logging.getLogger("aintelope.environments.savanna_zoo")


class SavannaZooParallelEnv(SavannaEnv, ParallelEnv):
    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}
        SavannaEnv.__init__(self, env_params)
        ParallelEnv.__init__(self)

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def step(self, actions: Dict[str, Action], *args, **kwargs) -> Step:
        (observations, rewards, dones, truncateds, infos) = SavannaEnv.step(
            self, actions, *args, **kwargs
        )

        # NB! clone since it will be modified below (dones is reference to self.dones)
        dones = dict(dones)
        for agent in list(
            self.agents
        ):  # clone list since it will be modified during iteration
            if self.dones[agent]:
                self.agents.remove(agent)
                del self.dones[agent]

        return (observations, rewards, dones, truncateds, infos)


class SavannaZooSequentialEnv(SavannaEnv, AECEnv):
    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}
        SavannaEnv.__init__(self, env_params)
        AECEnv.__init__(self)

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    @property
    def terminations(self):
        return self.dones

    @property
    def truncations(self):
        return {agent: False for agent, done in self.dones.items()}

    @property
    def agent_selection(self):
        return self._next_agent

    def reset(self, *args, **kwargs):
        self._next_agent = self.possible_agents[0]
        self._next_agent_index = 0
        self._all_agents_done = False
        return SavannaEnv.reset(self, *args, **kwargs)

    def step(self, action: Action, *args, **kwargs) -> None:
        self.step_single_agent(
            action, *args, **kwargs
        )  # NB! no return here, else Zoo tests will fail

    def step_single_agent(self, action: Action, *args, **kwargs):
        # a dead agent must call .step(None) once more after becoming dead.
        # Only after that call will this dead agent be removed from various
        # dictionaries and from .agent_iter loop.
        if self.terminations[self._next_agent] or self.truncations[self._next_agent]:
            if action is not None:
                raise ValueError("When an agent is dead, the only valid action is None")

            # Dead agents should stay in the agent_iter for one more loop,
            # but should get None as action.
            # Dead agents need to be removed from agents list only upon next step
            # function on this dead agent.
            del self.dones[self._next_agent]
            del self.infos[self._next_agent]
            # del self._cumulative_rewards[self._next_agent]
            self.agents.remove(self._next_agent)

            # other agents do not collect reward from current agent's "dead step" and
            # rewards from previous step need to be cleared
            reward = 0.0
            self.rewards = {agent: reward for agent in self.agents}

            self._move_to_next_agent()
            return

        for agent in self.agents:
            # the agent should be visible in .rewards after it dies
            # (until its "dead step"), but during next agent's step it
            # should get zero reward
            if self.dones[agent]:
                self.rewards[agent] = 0.0

        # this needs to be so according to Zoo unit test. See
        # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/test/api_test.py
        self._cumulative_rewards[self._next_agent] = 0.0

        # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope provide
        # slightly modified Zoo API. Normal Zoo sequential API step()
        # method does not return values.
        result = SavannaEnv.step(self, {self._next_agent: action}, *args, **kwargs)
        (
            observations,
            scores,
            terminateds,
            truncateds,
            infos,
        ) = result

        # NB! the agent_selection will change after call to _move_to_next_agent()
        # so we need to save the agent_id which just took the step
        step_agent = self._next_agent
        self._move_to_next_agent()
        return (
            observations[step_agent],
            scores[step_agent],
            terminateds[step_agent],
            truncateds[step_agent],
            infos[step_agent],
        )

    def _move_to_next_agent(
        self,
    ):
        """
        https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments
        """

        continue_search_for_non_done_agent = True
        search_loops_count = 0

        while continue_search_for_non_done_agent:
            self._next_agent_index = (self._next_agent_index + 1) % len(
                self.possible_agents
            )  # loop over agents repeatedly
            agent = self.possible_agents[self._next_agent_index]
            done = agent not in self.agents
            continue_search_for_non_done_agent = done

            search_loops_count += 1
            if continue_search_for_non_done_agent and search_loops_count == len(
                self.possible_agents
            ):  # all agents are done
                self._next_agent_index = -1
                self._next_agent = None
                self._all_agents_done = True
                return

        # / while search_for_non_done_agent:

        self._next_agent = agent

    # / def _move_to_next_agent(self):


register_env_class("savanna-zoo-sequential-v2", SavannaZooSequentialEnv)
register_env_class("savanna-zoo-parallel-v2", SavannaZooParallelEnv)
