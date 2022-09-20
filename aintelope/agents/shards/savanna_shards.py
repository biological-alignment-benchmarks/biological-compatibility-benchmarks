import typing as typ

from aintelope.environments.env_utils.distance import distance_to_closest_item


class Hunger:

    def __init__(self, shard_params={}) -> None:
        self.shard_params = shard_params
        self.hunger_rate = None
        self.max_hunger_reward = None
        self.last_ate = None

    def reset(self):
        self.hunger_rate = self.shard_params.get('hunger_rate', 10)
        self.max_hunger_reward = self.shard_params.get(
            'max_hunger_reward', 3.0)
        self.last_ate = self.shard_params.get('last_ate', -10)

    def calc_reward(self, agent, state):
        '''function of time since last ate and hunger rate and opportunity to eat'''
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        min_grass_distance = distance_to_closest_item(
            agent_pos, agent.env.grass_patches)
        event_signal = 0
        if min_grass_distance <= 2.1:
            self.last_ate = current_step
            event_signal = 1

        time_since_ate = current_step - self.last_ate
        current_hunger = time_since_ate / self.hunger_rate
        opportunity_to_eat = 1 / (1 + min_grass_distance)
        hunger_reward = (current_hunger *
                         opportunity_to_eat + (1 - current_hunger))
        hunger_reward = min(hunger_reward, self.max_hunger_reward)
        
        return hunger_reward, event_signal


class Thirst:

    def __init__(self, shard_params={}) -> None:
        self.shard_params = shard_params
        self.thirst_rate = None
        self.max_thirst_reward = None
        self.last_drank = None

    def reset(self):
        self.thirst_rate = self.shard_params.get('thirst_rate', 10)
        self.max_thirst_reward = self.shard_params.get(
            'max_thirst_reward', 4.0)
        self.last_drank = self.shard_params.get('last_drank', 0)

    def calc_reward(self, agent, state):
        '''function of time since last ate and thirst rate and opportunity to eat'''
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        min_water_distance = distance_to_closest_item(
            agent_pos, agent.env.water_holes)

        event_signal = 0
        if min_water_distance <= 1.1:
            self.last_drank = current_step
            event_signal = 1

        time_since_drank = current_step - self.last_drank
        current_thirst = time_since_drank / self.thirst_rate
        opportunity_to_drink = 1 / (1 + min_water_distance)
        thirst_reward = (current_thirst *
                         opportunity_to_drink + (1 - current_thirst))
        thirst_reward = min(thirst_reward, self.max_thirst_reward)
        return thirst_reward, event_signal


class Curiosity:

    def __init__(self, shard_params={}) -> None:
        self.shard_params = shard_params
        self.curiosity_rate = None
        self.max_curiosity_reward = None
        self.last_discovery = None

    def reset(self):
        self.curiosity_rate = self.shard_params.get('curiosity_rate', 2)
        self.max_curiosity_reward = self.shard_params.get(
            'max_curiosity_reward', 0.1)
        self.curiosity_window = self.shard_params.get('curiosity_window', 20)
        self.last_discovery = self.shard_params.get('last_discovery', 0)

    def calc_reward(self, agent, state):
        '''prefer not to revist tiles within curiosity window
        if agent had a sight-range, I'd add a preference to see new areas and objects
        could make this proportional to the nearest point in some sort of shifted window (e.g. 10 - 30)
        '''
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        recent_states = agent.replay_buffer.fetch_recent_states(
            self.curiosity_window)
        recent_positions = [[x[1], x[2]] for x in recent_states]
        event_signal = 0
        if agent_pos in recent_positions:
            time_since_discovery = current_step - self.last_discovery
            curiosity_reward = (self.max_curiosity_reward *
                                time_since_discovery / self.curiosity_rate)
            curiosity_reward = min(self.max_curiosity_reward, curiosity_reward)
        else:
            self.last_discovery = current_step
            curiosity_reward = self.max_curiosity_reward
            event_signal = 1
        return curiosity_reward, event_signal


available_shards_dict = {
    'hunger': Hunger,
    'thirst': Thirst,
    'curiosity': Curiosity
}
