import numpy as np
import pandas as pd
import torch
from torch import nn

from aintelope.agents import (
    Agent,
    GymEnv,
    PettingZooEnv,
    Environment,
    register_agent_class,
)


class InferenceAgent(Agent):
    """Inference class, for data analysis"""

    def __init__(
        self,
        env: Environment,
        model: nn.Module,
    ) -> None:
        self.env = env
        if isinstance(env, GymEnv):
            self.action_space = self.env.action_space
        elif isinstance(env, PettingZooEnv):
            self.action_space = self.env.action_space("agent0")
        else:
            raise TypeError(f"{type(env)} is not a valid environment")
        self.model = model
        self.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.done = False
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]

    def get_action(self, epsilon: float, device: str) -> Tuple[int, torch.Tensor]:
        # TODO: maybe not return as tensor? maybe not overload this function either?
        state = torch.tensor(np.expand_dims(self.state, 0))
        if device not in ["cpu"]:
            state = state.cuda(device)

        q_values = self.model(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())
        return action, q_values

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
        save_path: Optional[str] = None,
    ) -> Tuple[float, bool]:
        return 0.0, False

    def get_history(self) -> pd.DataFrame:
        """
        Method to get the history of the agent. Note that warm_start_steps are excluded.
        """
        return pd.DataFrame(
            columns=[
                "state",
                "action",
                "reward",
                "done",
                "instinct_events",
                "new_state",
            ],
            data=self.history[:],
        )


register_agent_class("inference_agent", InferenceAgent)
