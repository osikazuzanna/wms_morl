from core.models.facility import ControlledFacility
from gymnasium.spaces import Box, Space
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
from numpy.core.multiarray import interp as compiled_interp


class Weir(ControlledFacility):
    """
    A class used to represent a Weir, a type of controlled reservoir in the simulation.

    This class models the behavior of a weir, including water storage, outflow release, and reward calculation.
    It incorporates control actions to manage water flow and evaluates performance based on objective functions.

    Attributes
    ----------
    name : str
        Lowercase, non-spaced name of the weir.
    storage_vector : np.array
        Vector holding the volume of water in the weir throughout the simulation horizon (in cubic meters).
    level_vector : np.array
        Vector holding the elevation (height) of the water in the weir throughout the simulation horizon (in meters).
    release_vector : np.array
        Vector holding the actual average release per timestep from the weir (in cubic meters per second).
    evap_rates : np.array
        Monthly evaporation rates for the weir (in centimeters).
    stored_water : float
        The current volume of water stored in the weir (in cubic meters).
    spillage : float
        The amount of water spilling from the weir.
    should_split_release : bool
        Flag indicating if the release should be split into different destinations.
    objective_function : callable
        Function to evaluate the reward based on the stored water.
    objective_name : str
        Name of the objective function.
    integration_timestep_size : relativedelta
        Time resolution for the integration process (typically a month or smaller).
    max_capacity : float
        Maximum water storage capacity of the weir (in cubic meters).
    max_action : list[float]
        Maximum allowable action values for controlling water release.
    observation_space : Box
        Action space for the simulation environment.
    action_space : Box
        Observation space for the simulation environment.
    
    """

    def __init__(
        self,
        name: str,
        max_capacity: float,
        max_action: list[float],
        objective_function,
        integration_timestep_size: relativedelta,
        objective_name: str = "",
        stored_water: float = 0,
        spillage: float = 0,
        observation_space = Box(low=0, high=1),
        action_space = Box(low=0, high=1),

                ) -> None:
        """
        Initializes a Weir object with given parameters.

        Parameters
        ----------
        name : str
            The name of the weir.
        max_capacity : float
            The maximum storage capacity of the weir in cubic meters.
        max_action : list[float]
            The maximum allowable control actions for managing water release.
        objective_function : callable
            Function to evaluate the reward based on the stored water.
        integration_timestep_size : relativedelta
            Time step size for integration.
        objective_name : str, optional
            The name of the objective function (default is an empty string).
        stored_water : float, optional
            The initial amount of water stored in the weir (default is 0).
        spillage : float, optional
            The amount of water spilling from the weir (default is 0).
        observation_space : Box, optional
            The action space for the simulation environment (default range [0, 1]).
        action_space : Box, optional
            The observation space for the simulation environment (default range [0, 1]).
        """
        super().__init__(name, observation_space, action_space, max_capacity, max_action)
        self.stored_water: float = stored_water

        self.should_split_release = True
        
        
        self.storage_vector = []
        self.level_vector = []
        self.release_vector = []

        # Initialise storage vector
        self.storage_vector.append(stored_water)

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.integration_timestep_size: relativedelta = integration_timestep_size
        self.spillage = spillage


    def determine_reward(self) -> float:
        """
        Calculates the reward for the weir based on the stored water.

        Returns
        -------
        float
            The calculated reward value based on the stored water.
        """
        #Pass average inflow (which is stored_water ) to reward function
        return self.objective_function(self.stored_water)

    def determine_outflow(self, actions: np.array) -> list[float]:
        """
        Determines the average monthly water release based on the control actions.

        This method evaluates the inflow to the weir, scales the control actions, and calculates
        the outflow to different destinations.

        Parameters
        ----------
        actions : np.array
            Array of control actions, where each value represents the percentage of water to release to a destination.

        Returns
        -------
        list[float]
            A list containing the average monthly release from the weir.
        """

        destination_1_release = np.empty(0, dtype=np.float64)
        weir_observation_lst = []

        final_date = self.current_date + self.timestep_size

        while self.current_date < final_date:
            next_date = min(final_date, self.current_date + self.integration_timestep_size)

            #See what is the current inflow to weir and scale up the action to the first destination ( the action is a percentage of water going to destination 1)
            weir_observation = self.get_inflow(self.timestep)
            max_action = weir_observation 
            actions_scaled_up = actions*max_action

            destination_1_release = np.append(destination_1_release, actions_scaled_up)

            weir_observation_lst = np.append(weir_observation_lst, weir_observation)
            
            self.current_date = next_date

        #Averaging inflow to weir over last step (usually month) as a potential observation space to be used
        average_release = np.mean(weir_observation_lst, dtype=np.float64)
        self.storage_vector.append(average_release) #TODO does it make sense to keep it?
        self.stored_water = average_release # TODO used in determine_observation
      
        #potential storage (observation space understood as total inflow) is same as the total release
        self.release_vector.append(average_release)

        # Split release for different destinations, action is expected to be in range [0,1]
        self.split_release = [actions, (1-actions)]
        

        return average_release

    def determine_info(self) -> dict:
        """
        Returns key information about the weir.

        The dictionary contains the name of the weir and its average release rate.

        Returns
        -------
        dict
            A dictionary with information about the weir.
        """
        info = {
            "name": self.name,
            "average_release": self.stored_water,
        }
        return info

    def determine_observation(self) -> float:
        """
        Returns the current stored water level as part of the state space.

        Returns
        -------
        float
            The current stored water in the weir.
        """
        if self.stored_water > 0:
            return self.stored_water
        else:
            return 0.0

    def is_terminated(self) -> bool:
        """
        Determines whether the simulation should terminate.

        The simulation terminates if the weir's stored water exceeds its capacity or is below zero.

        Returns
        -------
        bool
            True if the simulation should terminate, otherwise False.
        """
        return self.stored_water > self.max_capacity or self.stored_water < 0
    
    def reset(self) -> None:
        """
        Resets the state of the weir to its initial condition.

        The weir's stored water and other vectors are reset to their initial values at the start of a new simulation.

        Returns
        -------
        None
            This method does not return a value, but modifies the internal state.
        """
        super().reset()
        stored_water = self.storage_vector[0]
        self.storage_vector = [stored_water]
        self.stored_water = stored_water
        self.level_vector = []
        self.release_vector = []
