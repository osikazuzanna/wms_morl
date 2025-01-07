import numpy as np
import gymnasium as gym
from gymnasium.spaces.dict import Dict
from core.envs.water_management_system import WaterManagementSystem
from core.models.facility import ControlledFacility


class ReshapeArrayAction(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: WaterManagementSystem):
        # assert isinstance(env.action_space, Dict)

        self.ordered_shapes = {}
        self.slices = {}
        current_index = 0

        for water_system in env.water_systems:
            if isinstance(water_system, ControlledFacility):
                number_of_actions = np.prod(water_system.action_space.shape)

                self.slices[water_system.name] = slice(current_index, current_index + number_of_actions)
                self.ordered_shapes[water_system.name] = water_system.action_space.shape

                current_index += number_of_actions

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

    def action(self, action):
        reshaped_actions = {}

        for name, sub_action_space_shape in self.ordered_shapes.items():
            reshaped_actions[name] = np.reshape(action[self.slices[name]], sub_action_space_shape)

        return reshaped_actions
