# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

from typing import Dict, Optional

import math
import numpy as np

from aintelope.environments.savanna_safetygrid import (
    AGENT_CHR1,
    AGENT_CHR2,
    ALL_AGENTS_LAYER,
    FOOD_CHR,
    DRINK_CHR,
    GOLD_CHR,
    SILVER_CHR,
    DANGER_TILE_CHR,
    PREDATOR_NPC_CHR,
    WALL_CHR,
    INFO_AGENT_OBSERVATION_COORDINATES,
    INFO_AGENT_OBSERVATION_LAYERS_ORDER,
    INFO_AGENT_INTEROCEPTION_ORDER,
    INFO_AGENT_INTEROCEPTION_VECTOR,
    INTEROCEPTION_FOOD,
    INTEROCEPTION_DRINK,
)


def format_float(value):
    if isinstance(value, str):  # for some reason np.isscalar() returns True for strings
        return value
    elif isinstance(value, list):
        return [format_float(x) for x in value]
    elif np.isscalar(value):
        if abs(value) < 1e-10:  # TODO: tune/config
            value = 0
        return "{0:G}".format(value)  # TODO: tune/config
    else:
        return str(value)


class Food:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.weight = 1.0  # TODO
        self.prev_interoception = None
        self.max_len_coordinates = 0
        self.deficiency_threshold = -1
        self.oversatiation_hysteresis = 1  # stop eating at this level since eating one more time will raise the satiation to oversatiation threshold
        self.oversatiation_threshold = 2

    def reset(self):
        pass

    def calc_reward(
        self, agent, state, info, agent_coordinate=(0, 0), predicting=False
    ):
        coordinates = info[INFO_AGENT_OBSERVATION_COORDINATES][FOOD_CHR]

        interoception_index = info[INFO_AGENT_INTEROCEPTION_ORDER].index(
            INTEROCEPTION_FOOD
        )
        interoception_vector = info[INFO_AGENT_INTEROCEPTION_VECTOR]
        interoception = interoception_vector[interoception_index]

        rewards = 0
        for coordinate in coordinates:
            # TODO: refactor distance calculation to a method
            xd = coordinate[0] - agent_coordinate[0]
            yd = coordinate[1] - agent_coordinate[1]
            distance = math.sqrt(xd * xd + yd * yd)

            # we use sum of inverse distances because it is better to be close to one or few foods than in the middle of many
            if interoception > self.oversatiation_hysteresis and distance == 0:
                # start moving away from food when hysteresis level has reached
                reward = -1 / (0.1 + 0.9)

            elif (
                interoception > 0
                and interoception <= self.oversatiation_hysteresis
                and distance == 0
            ):
                # NB! do not provide much extra reward for standing on food if the satiation is > 0 and less than hysteresis value, but do not penalise it either yet
                reward = 1 / (0.1 + 0.9)

            elif interoception <= 0:
                reward = 1 / (0.1 + distance)

            else:  # if the interoception is above zero then value being near food less
                reward = 0.1 / (0.1 + distance)

            if (
                interoception
                <= self.deficiency_threshold
                # and distance == 0  # amplify reward when satiation is < -1
            ):
                reward *= 10

            elif (
                interoception > self.oversatiation_threshold
                and distance == 0
                # and False
                # and self.prev_interoception == interoception
            ):
                # use amplified negative reward if staying on the food for so long that oversatiation_threshold level is reached or satiation does not increase
                reward += -100

            # penalise consuming last food patch if there has ever been more than one food patch. If there has ever been only one food patch then assume that it does not regrow and consuming it is okay.
            if self.max_len_coordinates > 1 and distance == 0:
                if not predicting and len(coordinates) == 0:
                    reward += (
                        -1000
                    )  # preserving last food is more important than hunger
                elif predicting and len(coordinates) <= 1:
                    reward += (
                        -1000
                    )  # preserving last food is more important than hunger

            rewards += reward

        # / for coordinate in coordinates:

        rewards *= self.weight

        # if not predicting:
        #    print(f"{agent.id} FOOD reward: {format_float(rewards)} interoception: {format_float(interoception)} prev_interoception: {format_float(self.prev_interoception)} agent_coordinate: {agent_coordinate}")

        self.max_len_coordinates = max(
            self.max_len_coordinates, len(coordinates)
        )  # NB! this is updated during prediction phase as well since the observation is still true
        if not predicting:
            self.prev_interoception = interoception

        event_signal = 0
        return rewards, event_signal


class Drink:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.weight = 1.0  # TODO
        self.prev_interoception = None
        self.max_len_coordinates = 0
        self.deficiency_threshold = -1
        self.oversatiation_hysteresis = 1  # stop drinking at this level since drinking one more time will raise the satiation to oversatiation threshold
        self.oversatiation_threshold = 2

    def reset(self):
        pass

    def calc_reward(
        self, agent, state, info, agent_coordinate=(0, 0), predicting=False
    ):
        coordinates = info[INFO_AGENT_OBSERVATION_COORDINATES][DRINK_CHR]

        interoception_index = info[INFO_AGENT_INTEROCEPTION_ORDER].index(
            INTEROCEPTION_DRINK
        )
        interoception_vector = info[INFO_AGENT_INTEROCEPTION_VECTOR]
        interoception = interoception_vector[interoception_index]

        rewards = 0
        for coordinate in coordinates:
            xd = coordinate[0] - agent_coordinate[0]
            yd = coordinate[1] - agent_coordinate[1]
            distance = math.sqrt(xd * xd + yd * yd)

            # we use sum of inverse distances because it is better to be close to one or few drinnks than in the middle of many
            if interoception > self.oversatiation_hysteresis and distance == 0:
                # start moving away from drink when hysteresis level has reached
                reward = -1 / (0.1 + 0.9)

            elif (
                interoception > 0
                and interoception <= self.oversatiation_hysteresis
                and distance == 0
            ):
                # NB! do not provide extra reward for standing on drink if the satiation is > 0 and less than hysteresis value, but do not penalise it either yet
                reward = 1 / (0.1 + 0.9)

            elif interoception <= 0:
                reward = 1 / (0.1 + distance)

            else:  # if the interoception is above zero then value being near drink less
                reward = 0.1 / (0.1 + distance)

            if (
                interoception
                <= self.deficiency_threshold
                # and distance == 0  # amplify reward when satiation is < -1
            ):
                reward *= 10

            elif (
                interoception > self.oversatiation_threshold
                and distance == 0
                # and self.prev_interoception == interoception
            ):
                # use amplified negative reward if staying on the drink for so long that oversatiation_threshold level is reached or satiation does not increase
                reward += -100

            # penalise consuming last food patch if there has ever been more than one food patch. If there has ever been only one food patch then assume that it does not regrow and consuming it is okay.
            if self.max_len_coordinates > 1 and distance == 0:
                if not predicting and len(coordinates) == 0:
                    reward += (
                        -1000
                    )  # preserving last food is more important than hunger
                elif predicting and len(coordinates) <= 1:
                    reward += (
                        -1000
                    )  # preserving last food is more important than hunger

            rewards += reward

        # / for coordinate in coordinates:

        rewards *= self.weight

        # if not predicting:
        #    print(f"{agent.id} DRINK reward: {format_float(rewards)} interoception: {format_float(interoception)} prev_interoception: {format_float(self.prev_interoception)} agent_coordinate: {agent_coordinate}")

        self.max_len_coordinates = max(
            self.max_len_coordinates, len(coordinates)
        )  # NB! this is updated during prediction phase as well since the observation is still true
        if not predicting:
            self.prev_interoception = interoception

        event_signal = 0
        return rewards, event_signal


class Gold:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.weight = 1.0  # NB! could be more than silver  # TODO
        self.log_base = 1.5
        self.collected_amount = 0
        self.food_drink_deficiency_threshold = -1

    def reset(self):
        pass

    def calc_reward(
        self, agent, state, info, agent_coordinate=(0, 0), predicting=False
    ):
        coordinates = info[INFO_AGENT_OBSERVATION_COORDINATES][GOLD_CHR]

        food_interoception_index = info[INFO_AGENT_INTEROCEPTION_ORDER].index(
            INTEROCEPTION_FOOD
        )
        drink_interoception_index = info[INFO_AGENT_INTEROCEPTION_ORDER].index(
            INTEROCEPTION_DRINK
        )
        interoception_vector = info[INFO_AGENT_INTEROCEPTION_VECTOR]
        food_interoception = interoception_vector[food_interoception_index]
        drink_interoception = interoception_vector[drink_interoception_index]

        rewards = 0
        for coordinate in coordinates:
            xd = coordinate[0] - agent_coordinate[0]
            yd = coordinate[1] - agent_coordinate[1]
            distance = math.sqrt(xd * xd + yd * yd)

            # diminishing returns
            prev_total_score = math.log(self.collected_amount + 1, self.log_base)
            next_total_score = math.log(self.collected_amount + 2, self.log_base)
            next_marginal_return = next_total_score - prev_total_score

            # we use sum of inverse distances because it is better to be close to one or few golds than in the middle of many
            inv_distance = 5 / (0.1 + distance)
            reward = inv_distance * next_marginal_return

            # it is much better to actually collect gold than just stand near it!
            if distance == 0:
                reward *= 10

            # reduce gold priority until food and drink satiation is sufficient
            if food_interoception < self.food_drink_deficiency_threshold:
                reward *= 0.01
            # elif food_interoception < 0:
            #    reward *= 0.1

            if drink_interoception < self.food_drink_deficiency_threshold:
                reward *= 0.01
            # elif drink_interoception < 0:
            #    reward *= 0.1

            rewards += reward

            if not predicting and distance == 0:
                self.collected_amount += 1

        rewards *= self.weight

        # if not predicting:
        #    print(f"{agent.id} GOLD reward: {format_float(rewards)} agent_coordinate: {agent_coordinate}")

        event_signal = 0
        return rewards, event_signal


class Silver:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.weight = 1.0  # NB! could be less than gold  # TODO
        self.log_base = 1.5
        self.collected_amount = 0
        self.food_drink_deficiency_threshold = -1

    def reset(self):
        pass

    def calc_reward(
        self, agent, state, info, agent_coordinate=(0, 0), predicting=False
    ):
        coordinates = info[INFO_AGENT_OBSERVATION_COORDINATES][SILVER_CHR]

        food_interoception_index = info[INFO_AGENT_INTEROCEPTION_ORDER].index(
            INTEROCEPTION_FOOD
        )
        drink_interoception_index = info[INFO_AGENT_INTEROCEPTION_ORDER].index(
            INTEROCEPTION_DRINK
        )
        interoception_vector = info[INFO_AGENT_INTEROCEPTION_VECTOR]
        food_interoception = interoception_vector[food_interoception_index]
        drink_interoception = interoception_vector[drink_interoception_index]

        rewards = 0
        for coordinate in coordinates:
            xd = coordinate[0] - agent_coordinate[0]
            yd = coordinate[1] - agent_coordinate[1]
            distance = math.sqrt(xd * xd + yd * yd)

            # diminishing returns
            prev_total_score = math.log(self.collected_amount + 1, self.log_base)
            next_total_score = math.log(self.collected_amount + 2, self.log_base)
            next_marginal_return = next_total_score - prev_total_score

            # we use sum of inverse distances because it is better to be close to one or few silvers than in the middle of many
            inv_distance = 5 / (0.1 + distance)
            reward = inv_distance * next_marginal_return

            # it is much better to actually collect silver than just stand near it!
            if distance == 0:
                reward *= 10

            # reduce silver priority until food and drink satiation is sufficient
            if food_interoception < self.food_drink_deficiency_threshold:
                reward *= 0.01
            # elif food_interoception < 0:
            #    reward *= 0.1

            if drink_interoception < self.food_drink_deficiency_threshold:
                reward *= 0.01
            # elif drink_interoception < 0:
            #    reward *= 0.1

            rewards += reward

            if not predicting and distance == 0:
                self.collected_amount += 1

        rewards *= self.weight

        event_signal = 0
        return rewards, event_signal


class DangerTile:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.weight = 0.1  # danger tiles are passive, so there is no strong need to stay far away from them   # TODO

    def reset(self):
        pass

    def calc_reward(
        self, agent, state, info, agent_coordinate=(0, 0), predicting=False
    ):
        coordinates = info[INFO_AGENT_OBSERVATION_COORDINATES][DANGER_TILE_CHR]
        rewards = 0
        for coordinate in coordinates:
            xd = coordinate[0] - agent_coordinate[0]
            yd = coordinate[1] - agent_coordinate[1]
            distance = math.sqrt(xd * xd + yd * yd)

            # we use sum of inverse distances because it is better to be in the middle of many danger tiles than close to one or few danger tiles
            reward = -1 / (0.1 + distance)
            if distance == 0:
                rewards += (
                    reward * 500
                )  # remove weight downscaling when stepping right into danger tile   # danger tile avoidance should have higher priority than hunger and thirst
            else:
                rewards += reward * self.weight

        event_signal = 0
        return rewards, event_signal


class Predator:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.weight = 1.0  # TODO

    def reset(self):
        pass

    def calc_reward(
        self, agent, state, info, agent_coordinate=(0, 0), predicting=False
    ):
        coordinates = info[INFO_AGENT_OBSERVATION_COORDINATES][PREDATOR_NPC_CHR]
        rewards = 0
        for coordinate in coordinates:
            xd = coordinate[0] - agent_coordinate[0]
            yd = coordinate[1] - agent_coordinate[1]
            distance = math.sqrt(xd * xd + yd * yd)

            # we use sum of inverse distances because it is better to be in the middle of many predators than close to one or few predators
            if distance == 0:
                reward = (
                    -1000
                )  # predator avoidance should have higher priority than hunger and thirst
            elif distance == 1:
                reward = -50 / (
                    0.1 + distance - 1
                )  # NB! reduce subjective distance by one since predators may move. So, distance=1 will be treated as if it was distance 0   # predator avoidance should have higher priority than hunger and thirst
            else:
                reward = -1 / (
                    0.1 + distance - 1
                )  # NB! reduce subjective distance by one since predators may move. So, distance=1 will be treated as if it was distance 0

            rewards += reward

        rewards *= self.weight

        # if not predicting:
        #    print(f"{agent.id} PREDATOR reward: {format_float(rewards)} agent_coordinate: {agent_coordinate}")

        event_signal = 0
        return rewards, event_signal


class Collision:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.weight = 1.0  # there is never point in moving against a wall since NOOP action is available

    def reset(self):
        pass

    def calc_reward(
        self, agent, state, info, agent_coordinate=(0, 0), predicting=False
    ):
        coordinates = info[INFO_AGENT_OBSERVATION_COORDINATES][WALL_CHR]
        coordinates += info[INFO_AGENT_OBSERVATION_COORDINATES][
            ALL_AGENTS_LAYER
        ]  # cannot walk into other agents either
        rewards = 0

        if predicting and tuple(agent_coordinate) != (
            0,
            0,
        ):  # wall avoidance is relevant only during action prediction with movement
            for coordinate in coordinates:
                # no need to calculate distance. Just detect running into walls
                if tuple(coordinate) == tuple(agent_coordinate):
                    rewards += -1

        rewards *= self.weight

        event_signal = 0
        return rewards, event_signal


class AntiMovement:
    def __init__(self, instinct_params={}) -> None:
        self.instinct_params = instinct_params
        self.weight = 1.0

    def reset(self):
        pass

    def calc_reward(
        self, agent, state, info, agent_coordinate=(0, 0), predicting=False
    ):
        rewards = 0

        if predicting and tuple(agent_coordinate) != (
            0,
            0,
        ):  # movement avoidance is relevant only during action prediction with movement
            rewards += -0.0001

        rewards *= self.weight

        event_signal = 0
        return rewards, event_signal


savanna_safetygrid_available_instincts_dict = {
    "food": Food,  # TODO: split into sub-instincts
    "drink": Drink,  # TODO: split into sub-instincts
    "gold": Gold,
    "silver": Silver,
    "danger": DangerTile,
    "predator": Predator,
    "collision": Collision,
    "antimovement": AntiMovement,  # can be merged with collision
}
