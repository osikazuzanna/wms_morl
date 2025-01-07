import gymnasium as gym
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from gymnasium.spaces import Box, Dict, Space
from gymnasium.core import ObsType, RenderFrame
from typing import Any, Union, Optional
from core.models.flow import Flow
from core.models.facility import Facility, ControlledFacility
import time
from gymnasium.spaces import flatten_space

class WaterManagementSystem(gym.Env):
    def __init__(
        self,
        water_systems: list[Union[Facility, ControlledFacility, Flow]],
        rewards: dict,
        start_date: datetime,
        timestep_size: relativedelta,
        seed: int = 42,
        add_timestamp = None,
        custom_obj = None
    ) -> None:
        self.water_systems: list[Union[Facility, ControlledFacility, Flow]] = water_systems
        self.rewards: dict = rewards

        self.start_date: datetime = start_date
        self.current_date: datetime = start_date
        self.timestep_size: relativedelta = timestep_size
        self.timestep: int = 0

        self.seed: int = seed
        self.add_timestamp = add_timestamp
        self.custom_obj = custom_obj

        self.observation_space: Space = self._determine_observation_space()
        self.observation_space = flatten_space(self.observation_space)
        self.action_space: Space = self._determine_action_space()
        self.action_space = flatten_space(self.action_space)

        self.max_capacities = self._determine_capacities()
        
        if self.custom_obj:
            self.reward_space: Space = Box(-1.0, 1.0, shape=(len(self.custom_obj),))
        else:
            self.reward_space: Space = Box(-1.0, 1.0, shape=(len(rewards.keys()),))


        self.observation: np.array = self._determine_observation()

        
        
        

        for water_system in self.water_systems:
            water_system.current_date = self.current_date
            water_system.timestep_size = self.timestep_size

    def _determine_observation(self) -> np.array:
        result = []
        for water_system in self.water_systems:
            if isinstance(water_system, ControlledFacility):
                result.append(water_system.determine_observation())
        result_normalized = list(np.divide(result, self.max_capacities))
        if self.add_timestamp:
                result.append(0)
                result_normalized.append(0)
            
        return np.array(result), np.array(result_normalized)

    def _determine_observation_space(self) -> Dict:
        if self.add_timestamp is not None:
            return Dict(
            {
            **{
                water_system.name: Box(low=0, high=1)
                for water_system in self.water_systems
                if isinstance(water_system, ControlledFacility)
            },
            "timestamp": Box(low=0, high=1)
            }
            )

        else:
            return  Dict(
                {
                    water_system.name: Box(low=0, high=1)
                    for water_system in self.water_systems
                    if isinstance(water_system, ControlledFacility)
                }
            )


    def _determine_action_space(self) -> Dict:
        return Dict(
            {
                water_system.name: water_system.action_space
                for water_system in self.water_systems
                if isinstance(water_system, ControlledFacility)
            }
        )

    def _determine_capacities(self):
        capacities = [water_system.max_capacity for water_system in self.water_systems if isinstance(water_system, ControlledFacility)]
        return capacities
    
    def _is_truncated(self) -> bool:
        return False

    def _determine_info(self) -> dict[str, Any]:
        # TODO: decide on what we wnat to output in the info.
        return {"water_systems": self.water_systems}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[ObsType, dict[str, Any]]:
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)
        self.current_date = self.start_date
        self.timestep = 0

        self.observation, observation_normalized = self._determine_observation()
        # Reset rewards
        for key in self.rewards.keys():
            self.rewards[key] = 0

        for water_system in self.water_systems:
            water_system.current_date = self.start_date
            water_system.reset()
        return observation_normalized, self._determine_info()

    def step(self, action: np.array) -> tuple[np.array, np.array, bool, bool, dict]:
        """
    Execute a single step in the water management simulation.

    This method processes the provided action for each water system in the 
    environment, updating the states of each facility and returning the 
    observations, rewards, termination status, truncation status, and 
    additional information.

    Args:
        action (np.array): An array of actions to be taken for each 
            controlled facility in the simulation. The action should 
            correspond to the facilities' names.

    Returns:
        tuple[np.array, np.array, bool, bool, dict]:
            A tuple containing:

            - np.array: A flattened array of normalized observations for 
              each water system, representing their current states as 
              percentages of their maximum capacities.
            - np.array: A flattened array of final rewards collected from 
              the facilities based on the executed actions.
            - bool: A flag indicating whether the simulation has 
              terminated.
            - bool: A flag indicating whether the simulation has been 
              truncated.
            - dict: A dictionary containing additional information, such 
              as the current date and other relevant data from the 
              water systems.

    Notes:
        - The method resets rewards for each facility at the beginning 
          of the step.
        - If `custom_obj` is specified, only the relevant rewards are 
          returned based on the keys in `custom_obj`.
        - Observations are normalized by dividing by their maximum 
          capacities.
        - The method increments the timestep and updates the current 
          date based on the `timestep_size` attribute.

    """

        final_reward = {}
        

        # Reset rewards
        for key in self.rewards.keys():
            final_reward[key] = 0

        final_observation = {}
        final_terminated = False
        final_truncated = False
        final_info = {"date": self.current_date}

        for water_system in self.water_systems:
            water_system.current_date = self.current_date

            if isinstance(water_system, ControlledFacility):
                observation, reward, terminated, truncated, info = water_system.step(action[water_system.name])

            elif isinstance(water_system, Facility) or isinstance(water_system, Flow):
                observation, reward, terminated, truncated, info = water_system.step()
            else:
                raise ValueError()

            # Set observation for a Controlled Facility.
            if isinstance(water_system, ControlledFacility):
                final_observation[water_system.name] = observation

            # Add reward to the objective assigned to this Facility (unless it is a Flow or the facility has no objectives).
            if isinstance(water_system, Facility) or isinstance(water_system, ControlledFacility):
                if water_system.objective_name:
                    final_reward[water_system.objective_name] += reward

            # Store additional information
            final_info[water_system.name] = info


            # Determine whether program should stop
            final_terminated = final_terminated or terminated
            final_truncated = final_truncated or truncated or self._is_truncated()


        self.timestep += 1
        self.current_date += self.timestep_size

        #check if only a subset of rewards to return
        if self.custom_obj is not None:
            final_reward = [final_reward[key] for key in self.custom_obj]
        else:
            final_reward = list(final_reward.values())

        final_observations = list(final_observation.values())
        #normalize the observation to be percentage of max capacity
        final_observations = list(np.divide(final_observations, self.max_capacities))
        if self.add_timestamp=='m':
            final_observations.append(final_info['date'].month/12)
        elif self.add_timestamp=='h':
            final_observations.append(final_info['date'].hour/24)
             


        return (
            np.array(final_observations).flatten(),
            np.array(final_reward).flatten(),
            final_terminated,
            final_truncated,
            final_info
        )

    def close(self) -> None:
        # TODO: implement if needed, e.g. for closing opened rendering frames.
        pass

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        # TODO: implement if needed, for rendering simulation.
        pass
