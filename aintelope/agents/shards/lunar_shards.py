import typing as typ

from aintelope.environments.env_utils.distance import distance_to_closest_item

'''
Nathan: I'm not too sure about this. If we create instincts for a 'lander' as if it were
a creature in and of itself, rather than a vehicle being piloted by a creature,
that limits our interchanagability. I think in the future we'd want a more general interpretation
where these 'shards' are explicitly learned rather than hardcoded, and the underlying shards 
that we hardcode would be sufficiently generic that they'd work for the savanna roles (lion, antelope) as well as this.
But if we want the learned-shards to be explicit (which would be nice) we'd need some different
learning system for that. Not sure how to accomplish this, more thought and discussion needed.
'''

class Safety:

    def __init__(self, shard_params={}) -> None:
        self.shard_params = shard_params
        self.hunger_rate
        self.max_hunger_reward
        self.last_ate

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

        if min_grass_distance < 1.0:
            self.last_ate = current_step

        time_since_ate = current_step - self.last_ate
        current_hunger = time_since_ate / self.hunger_rate
        opportunity_to_eat = 1 / (1 + min_grass_distance)
        hunger_reward = (current_hunger *
                         opportunity_to_eat + (1 - current_hunger))
        hunger_reward = min(hunger_reward, self.max_hunger_reward)
        return hunger_reward


class Fuel:

    def __init__(self, shard_params={}) -> None:
        self.shard_params = shard_params
        self.thirst_rate
        self.max_thirst_reward
        self.last_drank

    def reset(self):
        self.thirst_rate = self.shard_params.get('thirst_rate', 10)
        self.max_thirst_reward = self.shard_params.get(
            'max_thirst_reward', 4.0)
        self.last_drank = self.shard_params.get('last_drank', 0)

    def calc_reward(self, agent, state):
        '''function of time since last ate and thirst rate and opportunity to eat'''
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        min_grass_distance = distance_to_closest_item(
            agent_pos, agent.env.water_holes)

        if min_grass_distance < 1.0:
            self.last_drank = current_step

        time_since_drank = current_step - self.last_drank
        current_thirst = time_since_drank / self.thirst_rate
        opportunity_to_drink = 1 / (1 + min_grass_distance)
        thirst_reward = (current_thirst *
                         opportunity_to_drink + (1 - current_thirst))
        thirst_reward = min(thirst_reward, self.max_thirst_reward)
        return thirst_reward


class Mission:

    def __init__(self, shard_params={}) -> None:
        self.shard_params = shard_params
        self.thirst_rate
        self.max_thirst_reward
        self.last_drank

    def reset(self):
        self.thirst_rate = self.shard_params.get('thirst_rate', 10)
        self.max_thirst_reward = self.shard_params.get(
            'max_thirst_reward', 4.0)
        self.last_drank = self.shard_params.get('last_drank', 0)

    def calc_reward(self, agent, state):
        '''function of time since last ate and thirst rate and opportunity to eat'''
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        min_grass_distance = distance_to_closest_item(
            agent_pos, agent.env.water_holes)

        if min_grass_distance < 1.0:
            self.last_drank = current_step

        time_since_drank = current_step - self.last_drank
        current_thirst = time_since_drank / self.thirst_rate
        opportunity_to_drink = 1 / (1 + min_grass_distance)
        thirst_reward = (current_thirst *
                         opportunity_to_drink + (1 - current_thirst))
        thirst_reward = min(thirst_reward, self.max_thirst_reward)
        return thirst_reward


class Curiosity:

    def __init__(self, shard_params={}) -> None:
        self.shard_params = shard_params
        self.curiosity_rate
        self.max_curiosity_reward
        self.last_discovery

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
        if agent_pos in recent_positions:
            time_since_discovery = current_step - self.last_discovery
            curiosity_reward = (self.max_curiosity_reward *
                                time_since_discovery / self.curiosity_rate)
            curiosity_reward = min(self.max_curiosity_reward, curiosity_reward)
        else:
            self.last_discovery = current_step
            curiosity_reward = self.max_curiosity_reward
        return curiosity_reward


available_shards_dict = {
    'hunger': Hunger,
    'thirst': Thirst,
    'curiosity': Curiosity
}
