import logging

import gymnasium as gym

logger = logging.getLogger("aintelope.environments.env_utils.cleanup")


def cleanup_gym_envs(keyword="savanna"):
    registry_keys = list(gym.envs.registration.registry.keys())
    for env in registry_keys:
        if keyword in env:
            logger.info(f"Removing {env} from registry")
            del gym.envs.registration.registry[env]
