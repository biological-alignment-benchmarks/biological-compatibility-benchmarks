from typing import Optional, Dict
import logging
import functools

from pettingzoo import AECEnv, ParallelEnv

from aintelope.environments.savanna import (
    SavannaEnv,
    RenderSettings,
    RenderState,
    HumanRenderState,
    move_agent,
    reward_agent,
    PositionFloat,
    Action,
)

logger = logging.getLogger("aintelope.environments.savanna_zoo")


class SavannaZooParallelEnv(SavannaEnv, ParallelEnv):
    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}
        SavannaEnv.__init__(self, env_params)
        ParallelEnv.__init__(self)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]


class SavannaZooSequentialEnv(SavannaEnv, AECEnv):
    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}
        SavannaEnv.__init__(self, env_params)
        AECEnv.__init__(self)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
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

    @property
    def _cumulative_rewards(self):
        return self.rewards

    def reset(self, *args, **kwargs):
        self._next_agent = self.possible_agents[0]
        self._next_agent_index = 0
        self._all_agents_done = False
        SavannaEnv.reset(self, *args, **kwargs)

    def step(self, action: Action, *args, **kwargs):
        # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope provide slightly modified Zoo API. Normal Zoo sequential API step() method does not return values.
        result = SavannaEnv.step(self, {self.agent_selection: action}, *args, **kwargs)
        (
            observations,
            scores,
            terminateds,
            truncateds,
            infos,
        ) = result
        step_agent = (
            self.agent_selection
        )  # NB! the agent_selection will change after call to _move_to_next_agent()
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
    ):  # https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments
        continue_search_for_non_done_agent = True
        search_loops_count = 0

        while continue_search_for_non_done_agent:
            self._next_agent_index = (self._next_agent_index + 1) % len(
                self.possible_agents
            )  # loop over agents repeatedly     # https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments
            agent = self.possible_agents[self._next_agent_index]

            done = self.terminations[agent] or self.truncations[agent]
            continue_search_for_non_done_agent = done

            search_loops_count += 1
            if continue_search_for_non_done_agent and search_loops_count == len(
                self.possible_agents
            ):  # all agents are done     # https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments
                self._next_agent_index = -1
                self._next_agent = None
                self._all_agents_done = True
                return

        # / while search_for_non_done_agent:

        self._next_agent = agent

    # / def _move_to_next_agent(self):
