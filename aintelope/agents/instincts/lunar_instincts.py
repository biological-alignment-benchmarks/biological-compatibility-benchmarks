import typing as typ

from aintelope.environments.env_utils.distance import distance_to_closest_item

"""
Nathan: I'm not too sure about this. If we create instincts for a 'lander' as if it were
a creature in and of itself, rather than a vehicle being piloted by a creature,
that limits our interchanagability. I think in the future we'd want a more general interpretation
where these 'instincts' are explicitly learned rather than hardcoded, and the underlying instincts 
that we hardcode would be sufficiently generic that they'd work for the savanna roles (lion, antelope) as well as this.
But if we want the learned-instincts to be explicit (which would be nice) we'd need some different
learning system for that. Not sure how to accomplish this, more thought and discussion needed.
"""


class Safety:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.previous_speed = None
        self.last_vertical_orientation = None
        self.reluctance_for_speed_at_obstacles = None
        self.reluctance_for_acceleration_at_obstacles = None
        self.desired_obstacle_clearance = None
        self.max_safety_reward = None

    def reset(self):
        self.speed = self.instinct_params.get("speed", 10)
        self.last_vertical_orientation = 0
        self.desired_obstacle_clearance = self.instinct_params.get(
            "desired_obstacle_clearance", 10
        )
        self.reluctance_for_speed_at_obstacles = self.instinct_params.get(
            "reluctance_for_speed_at_obstacles", 10
        )
        self.reluctance_for_acceleration_at_obstacles = self.instinct_params.get(
            "reluctance_for_acceleration_at_obstacles", 10
        )
        self.max_safety_reward = self.instinct_params.get("max_safety_reward", 10)

    def calc_reward(self, agent, state):
        """function of:
        desire to remain close to upright,
        not too close to obstacles (desired obstacle clearance),
        not heading towards obstacles,
        not accelerating towards obstacles,

        """
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        min_obstacle_distance = distance_to_closest_item(
            agent_pos, agent.env.obstacle_boundaries
        )

        safety_reward = 1  # TODO: implement
        safety_reward = min(safety_reward, self.max_safety_reward)
        return safety_reward


class Fuel:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.starting_fuel = None
        self.current_fuel = None
        self.initial_goal_distance = None
        self.goal_distance = None
        self.fuel_location_in_state = None

    def reset(self):
        self.starting_fuel = self.instinct_params.get("starting_fuel", 10)
        self.current_fuel = self.starting_fuel
        self.initial_goal_distance = self.instinct_params.get(
            "initial_goal_distance", 100
        )
        self.max_fuel_reward = self.instinct_params.get("max_fuel_reward", 4.0)
        self.fuel_location_in_state = self.instinct_params.get(
            "fuel_location_in_state", 0
        )

    def calc_reward(self, agent, state):
        """function of starting fuel, current_fuel, goal_distance"""
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        current_goal_distance = distance_to_specific_item(agent_pos, agent.env.goal)

        current_fuel = state[self.fuel_location_in_state]

        fuel_reward = current_fuel / self.max_fuel_reward
        # something about fuel consumption rate calculated from distance travelled so far and fuel consumed so far,
        # remaining distance to goal,
        # and whether we're on track to get to goal still with current remaining fuel
        fuel_reward = min(fuel_reward, self.max_fuel_reward)
        return fuel_reward


class Mission:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.initial_goal_distance = None

    def reset(self):
        self.initial_goal_distance = self.instinct_params.get(
            "initial_goal_distance", 100
        )

    def calc_reward(self, agent, state):
        """function of remaining distance to goal as percent of initial distance, plus big bonus for actually safely at goal"""
        agent_pos = [state[1], state[2]]
        current_distance_to_goal = current_goal_distance = distance_to_specific_item(
            agent_pos, agent.env.goal
        )
        mission_reward = 10 * (current_distance_to_goal / self.initial_goal_distance)
        if current_distance_to_goal == 0:
            mission_reward += 100
        return mission_reward


class Curiosity:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.curiosity_rate
        self.max_curiosity_reward
        self.last_discovery

    def reset(self):
        self.curiosity_rate = self.instinct_params.get("curiosity_rate", 2)
        self.max_curiosity_reward = self.instinct_params.get(
            "max_curiosity_reward", 0.1
        )
        self.curiosity_window = self.instinct_params.get("curiosity_window", 20)
        self.last_discovery = self.instinct_params.get("last_discovery", 0)

    def calc_reward(self, agent, state):
        """a mild desire to experience new states not recently entered into.
        maybe add a random factor to this so it comes and goes?
        """
        current_step = agent.env.num_moves
        agent_pos = [state[1], state[2]]
        recent_states = agent.replay_buffer.fetch_recent_states(self.curiosity_window)
        recent_positions = [[x[1], x[2]] for x in recent_states]
        if agent_pos in recent_positions:
            time_since_discovery = current_step - self.last_discovery
            curiosity_reward = (
                self.max_curiosity_reward * time_since_discovery / self.curiosity_rate
            )
            curiosity_reward = min(self.max_curiosity_reward, curiosity_reward)
        else:
            self.last_discovery = current_step
            curiosity_reward = self.max_curiosity_reward
        return curiosity_reward


available_instincts_dict = {
    "safety": Safety,
    "fuel": Fuel,
    "mission": Mission,
    "curiosity": Curiosity,
}
