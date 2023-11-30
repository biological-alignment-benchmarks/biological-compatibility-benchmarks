from typing import Optional, Dict
import numpy as np
from aintelope.environments.env_utils.distance import distance_to_closest_item
from aintelope.environments.savanna import get_agent_pos_from_state
from aintelope.environments.savanna import get_grass_pos_from_state


class Smell:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params

    def reset(self):
        pass

    def calc_reward(self, state):
        """function of smell intensity for food"""
        agent_pos = get_agent_pos_from_state(state)
        min_grass_distance = distance_to_closest_item(
            agent_pos, np.array(get_grass_pos_from_state(state))
        )
        event_signal = 0
        smell_reward = 1.0 / (min_grass_distance + 1.0)
        return smell_reward, event_signal


class Hunger:
    def __init__(self, instinct_params: Optional[Dict] = None) -> None:
        self.instinct_params = {} if instinct_params is None else instinct_params
        self.reset()

    def reset(self):
        self.hunger_rate = self.instinct_params.get("hunger_rate", 10.0)
        self.max_hunger_reward = self.instinct_params.get("max_hunger_reward", 3.0)
        self.last_ate = self.instinct_params.get("last_ate", -10)

    def calc_reward(self, agent, state):
        """function of time since last ate and hunger rate and opportunity to eat"""
        current_step = agent.env.num_moves
        agent_pos = get_agent_pos_from_state(state)
        min_grass_distance = distance_to_closest_item(
            agent_pos, agent.env.grass_patches
        )
        event_signal = 0
        if min_grass_distance <= 2.1:
            self.last_ate = current_step
            event_signal = 1

        time_since_ate = current_step - self.last_ate
        current_hunger = time_since_ate / self.hunger_rate
        opportunity_to_eat = 1 / (1 + min_grass_distance)
        hunger_reward = current_hunger * opportunity_to_eat + (1 - current_hunger)
        hunger_reward = min(hunger_reward, self.max_hunger_reward)

        return hunger_reward, event_signal


class Thirst:
    def __init__(self, instinct_params: Optional[Dict] = None) -> None:
        self.instinct_params = {} if instinct_params is None else instinct_params
        self.reset()

    def reset(self):
        self.thirst_rate = self.instinct_params.get("thirst_rate", 10.0)
        self.max_thirst_reward = self.instinct_params.get("max_thirst_reward", 4.0)
        self.last_drank = self.instinct_params.get("last_drank", 0)

    def calc_reward(self, agent, state):
        """function of time since last ate and thirst rate and opportunity to eat"""
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        min_water_distance = distance_to_closest_item(agent_pos, agent.env.water_holes)

        event_signal = 0
        if min_water_distance <= 1.1:
            self.last_drank = current_step
            event_signal = 1

        time_since_drank = current_step - self.last_drank
        current_thirst = time_since_drank / self.thirst_rate
        opportunity_to_drink = 1 / (1 + min_water_distance)
        thirst_reward = current_thirst * opportunity_to_drink + (1 - current_thirst)
        thirst_reward = min(thirst_reward, self.max_thirst_reward)
        return thirst_reward, event_signal


class Curiosity:
    def __init__(self, instinct_params: Optional[Dict] = None) -> None:
        self.instinct_params = {} if instinct_params is None else instinct_params
        self.reset()

    def reset(self):
        self.curiosity_rate = self.instinct_params.get("curiosity_rate", 2.0)
        self.max_curiosity_reward = self.instinct_params.get(
            "max_curiosity_reward", 0.1
        )
        self.curiosity_window = self.instinct_params.get("curiosity_window", 20)
        self.last_discovery = self.instinct_params.get("last_discovery", 0)

    def calc_reward(self, agent, state):
        """prefer not to revist tiles within curiosity window
        if agent had a sight-range, I'd add a preference to see new areas and objects
        could make this proportional to the nearest point in some sort of shifted
        window (e.g. 10 - 30)
        """
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        recent_states = agent.replay_buffer.fetch_recent_states(self.curiosity_window)
        recent_positions = [[x[1], x[2]] for x in recent_states]
        event_signal = 0
        if agent_pos in recent_positions:
            time_since_discovery = current_step - self.last_discovery
            curiosity_reward = (
                self.max_curiosity_reward * time_since_discovery / self.curiosity_rate
            )
            curiosity_reward = min(self.max_curiosity_reward, curiosity_reward)
        else:
            self.last_discovery = current_step
            curiosity_reward = self.max_curiosity_reward
            event_signal = 1
        return curiosity_reward, event_signal


available_instincts_dict = {
    "smell": Smell,
    "hunger": Hunger,
    "thirst": Thirst,
    "curiosity": Curiosity,
}
