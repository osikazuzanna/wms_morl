from core.models.facility import ControlledFacility
from gymnasium.spaces import Box, Space
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
from numpy.core.multiarray import interp as compiled_interp
from typing import Callable
import inspect
from core.utils import utils

class ReservoirWithPump(ControlledFacility):
    """
    A class used to represent reservoirs with a pump system that can pump water in and out of the reservoir,
    and manage the water storage and release. It combines functionalities of a reservoir and a pump station, 
    incorporating additional control for managing inflows, evaporation, and releases.

    Attributes
    ----------
    name_reservoir: str
        The lowercase, non-spaced name of the reservoir.
    name_pump: str
        The lowercase, non-spaced name of the pump station.
    storage_vector: np.array (1xH)
        A vector containing the volume of water in the reservoir (in cubic meters) over the simulation horizon.
    level_vector: np.array (1xH)
        A vector holding the elevation (height in meters) of water in the reservoir throughout the simulation horizon.
    release_vector: np.array (1xH)
        A vector that holds the average release rate (in cubic meters per second) of water from the reservoir over time.
    evap_rates: np.array (1x12)
        Monthly evaporation rates (in centimeters) for the reservoir.
    evap_rates_pump: np.array (1x12)
        Monthly evaporation rates (in centimeters) for the pump.
    evap_rates_timestep_size: relativedelta
        The timestep size used for evaporation calculations.
    storage_to_minmax_rel: list[list[float]]
        Relationships between the storage volume and the corresponding minimum and maximum possible releases.
    storage_to_level_rel: list[list[float]]
        Relationships between the storage volume and the corresponding water level (height).
    storage_to_surface_rel: list[list[float]]
        Relationships between the storage volume and the corresponding surface area of the reservoir.
    storage_to_surface_rel_pump: list[list[float]]
        Relationships between the storage volume in the pump and the corresponding surface area.
    storage_to_level_rel_pump: list[list[float]]
        Relationships between the storage volume in the pump and the corresponding water level (height).
    pumping_rules: Callable
        A function that defines the pumping rules for transferring water between the pump and the reservoir.
    inflows_pump: list[float], optional
        A list of inflows to the pump, used to update the water level in the pump over time.
    objective_name: str, optional
        The name of the objective function, used for evaluation purposes.
    stored_water_reservoir: float, optional
        The initial amount of water in the reservoir (in cubic meters).
    stored_water_pump: float, optional
        The initial amount of water in the pump (in cubic meters).
    observation_space: Box, optional
        Defines the observation space for the reservoir and pump system, used for reinforcement learning.
    action_space: Box, optional
        Defines the action space for the reservoir and pump system, used for reinforcement learning.
    spillage: float, optional
        The amount of water that overflows or is spilled from the system.
    """
    def __init__(
        self,
        name: str,
        max_capacity: float,
        max_action: float,
        objective_function,
        integration_timestep_size: relativedelta,
        evap_rates: list[float],
        evap_rates_pump: list[float],
        evap_rates_timestep_size: relativedelta,
        storage_to_minmax_rel: list[list[float]],
        storage_to_level_rel: list[list[float]],
        storage_to_surface_rel: list[list[float]],
        storage_to_surface_rel_pump: list[list[float]],
        storage_to_level_rel_pump: list[list[float]],
        pumping_rules: Callable,
        inflows_pump: list[float] = None,
        objective_name: str = "",
        stored_water_reservoir: float = 0,
        stored_water_pump: float = 0,
        observation_space = Box(low=0, high=1),
        action_space = Box(low=0, high=1),
        spillage: float = 0
                ) -> None:
        """
        Initializes the ReservoirWithPump system, including the pump, reservoir, and their associated properties.
        
        Parameters
        ----------
        name: str
            The name of the reservoir system.
        max_capacity: float
            Maximum storage capacity of the reservoir (m³).
        max_action: float
            Maximum action value allowed for the system.
        objective_function: Callable
            A function used to calculate the objective (reward) based on the reservoir's state.
        integration_timestep_size: relativedelta
            The timestep size for numerical integration.
        evap_rates: list[float]
            A list of evaporation rates for the reservoir over the months.
        evap_rates_pump: list[float]
            A list of evaporation rates for the pump over the months.
        evap_rates_timestep_size: relativedelta
            Timestep for evaporation calculations.
        storage_to_minmax_rel: list[list[float]]
            Relationships between storage volume and release rates.
        storage_to_level_rel: list[list[float]]
            Relationships between storage volume and water level.
        storage_to_surface_rel: list[list[float]]
            Relationships between storage volume and surface area.
        storage_to_surface_rel_pump: list[list[float]]
            Relationships between storage volume in the pump and surface area.
        storage_to_level_rel_pump: list[list[float]]
            Relationships between storage volume in the pump and water level.
        pumping_rules: Callable
            Function defining the pumping rules.
        inflows_pump: list[float], optional
            List of inflows to the pump, optional.
        objective_name: str, optional
            The name of the objective function.
        stored_water_reservoir: float, optional
            Initial stored water volume in the reservoir.
        stored_water_pump: float, optional
            Initial stored water volume in the pump.
        observation_space: Box, optional
            Defines the observation space.
        action_space: Box, optional
            Defines the action space.
        spillage: float, optional
            Amount of water spilled from the system.

        """
        super().__init__(name, observation_space, action_space, max_capacity, max_action)
        self.stored_water: float = stored_water_reservoir
        self.stored_pump: float = stored_water_pump

        self.evap_rates = evap_rates
        self.evap_rates_pump = evap_rates_pump
        self.evap_rates_timestep = evap_rates_timestep_size
        self.storage_to_minmax_rel = storage_to_minmax_rel
        self.storage_to_level_rel = storage_to_level_rel
        self.storage_to_surface_rel = storage_to_surface_rel
        self.storage_to_surface_rel_pump = storage_to_surface_rel_pump
        self.storage_to_level_rel_pump = storage_to_level_rel_pump
        self.spillage = spillage
        self.required_params = ['day_of_the_week', 'hour', 'level_reservoir', 'level_pump', 'storage_reservoir', 'storage_pump']

        self.inflows_pump = inflows_pump
        
        self.storage_vector = []
        self.storage_pump_vector = []
        self.level_vector = []
        self.release_vector = []

        # Initialise storage vector
        self.storage_vector.append(stored_water_reservoir)
        self.storage_pump_vector.append(stored_water_pump)

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.integration_timestep_size: relativedelta = integration_timestep_size
        

        if not callable(pumping_rules):
            raise ValueError("The pumping rules should be defined as a function, which takes as argument storage_level of the reservoir and the pump.")
        
        # Get the signature of the method
        sig = inspect.signature(pumping_rules)

        # Check if the method has the required parameters
        for param in self.required_params:
            if param not in sig.parameters:
                raise ValueError(f"The method must have a parameter named '{param}'.")

        # Assign the provided method to the instance
        self.pumping_rules = pumping_rules


    def determine_reward(self) -> float:

        """
        Calculates the reward based on the current storage of the reservoir.
        
        The reward is determined by passing the water level corresponding to the stored water
        in the reservoir through the objective function.
        
        Returns
        -------
        float
            The computed reward, which is the result of the objective function applied to
            the current water level.
        """
        # Pass water level to reward function
        return self.objective_function(self.storage_to_level(self.stored_water))

    def determine_outflow(self, actions: np.array) -> list[float]:

        """
        Determines the outflow (release rate) from the reservoir and pump system based on
        the given actions. The outflow is influenced by the current storage in both the
        reservoir and pump, as well as evaporation and the pumping/release rules.
        
        Parameters
        ----------
        actions : np.array
            The action values representing the desired release rates (scaled by `max_action`).
            
        Returns
        -------
        list[float]
            A list of release rates (in m³/s), which may be split among multiple destinations.
        """
        #determine current storage for the reservoir
        current_storage = self.storage_vector[-1]
        #determine current storage for the pump
        current_storage_pump = self.storage_pump_vector[-1]
        #check if we are releasing to one destination or more
        if self.should_split_release == True:
            #if it's more destinations, we have to create a list for sub-releases during the integration loop
            sub_releases = []
            actions = np.multiply(actions, self.max_action)
        else:
            sub_releases = np.empty(0, dtype=np.float64)
            actions = actions*self.max_action

            
        final_date = self.current_date + self.timestep_size
        timestep_seconds = (final_date + self.evap_rates_timestep - final_date).total_seconds()
        evaporatio_rate_per_second = self.evap_rates[self.determine_time_idx()] / (100 * timestep_seconds)
        evaporatio_rate_per_second_pump = self.evap_rates_pump[self.determine_time_idx()] / (100 * timestep_seconds)
        
        while self.current_date < final_date:
            next_date = min(final_date, self.current_date + self.integration_timestep_size)
            integration_time_seconds = (next_date - self.current_date).total_seconds()
            
            #pumping/release of the pump

            pumping, release_pump = self.pumping_rules(day = self.current_date.weekday(), 
                                                  hour = self.current_date.hour, 
                                                  level_reservoir = self.storage_to_level(current_storage), 
                                                  level_pump = self.storage_to_level_pump(self.stored_pump),
                                                  storage_reservoir = current_storage, storage_pump = self.stored_pump)


            surface = self.storage_to_surface(current_storage)
            surface_pump = self.storage_to_surface_pump(current_storage_pump)

            evaporation = surface * (evaporatio_rate_per_second * integration_time_seconds)
            evaporation_pump = surface_pump * (evaporatio_rate_per_second_pump * integration_time_seconds)

            current_storage_pump += (self.inflows_pump[self.timestep] + pumping - release_pump) * integration_time_seconds - evaporation_pump


            min_possible_release, max_possible_release = self.storage_to_minmax(current_storage)

            release_per_second = min(max_possible_release, max(min_possible_release, np.sum(actions)))

            #depending if there are multiple outflows, append release decisions for every integration step
            if self.should_split_release == True:
                sub_releases.append(release_per_second)
            else:
                sub_releases = np.append(sub_releases, release_per_second)

            total_addition = (self.get_inflow(self.timestep) + release_pump) * integration_time_seconds

            current_storage += total_addition - evaporation - np.sum(release_per_second) * integration_time_seconds

            self.current_date = next_date

        # Update the amount of water in the Reservoir
        self.storage_vector.append(current_storage)
        self.stored_water = current_storage

        # Update water in the pump
        self.storage_pump_vector.append(current_storage_pump)
        self.stored_pump = current_storage_pump

        # Record level based on storage for time t
        self.level_vector.append(self.storage_to_level(current_storage))

        # Calculate the ouflow of water
        if self.should_split_release == True:
            sub_releases = np.array(sub_releases)
            average_release = np.mean(sub_releases, dtype=np.float64, axis = 0)
        else:
            average_release = np.mean(sub_releases, dtype=np.float64)

        self.release_vector.append(average_release)

        total_action = np.sum(average_release)
        # Split release for different destinations
        if self.should_split_release and total_action != 0:
            
            self.split_release = [(action / total_action) for action in average_release]
            average_release = total_action

        return average_release

    def determine_info(self) -> dict:
        """
        Returns a dictionary containing key information about the current state of the system.
        The dictionary includes the name of the reservoir, stored water volume, current water level,
        current release rate, evaporation rates, and the pump level.
        
        Returns
        -------
        dict
            A dictionary with the following keys:
            - "name": The name of the reservoir.
            - "stored_water": The current volume of water in the reservoir.
            - "current_level": The current water level in the reservoir (if available).
            - "current_release": The current release rate from the reservoir (if available).
            - "evaporation_rates": A list of monthly evaporation rates.
            - "pump_level": The current volume of water in the pump.
        """
        info = {
            "name": self.name,
            "stored_water": self.stored_water,
            "current_level": self.level_vector[-1] if self.level_vector else None,
            "current_release": self.release_vector[-1] if self.release_vector else None,
            "evaporation_rates": self.evap_rates.tolist(),
            "pump_level": self.stored_pump
        }
        return info

    def determine_observation(self) -> float:
        """
        Returns the current state of the reservoir in terms of the stored water volume.
        
        Returns
        -------
        float
            The volume of water currently stored in the reservoir.
        """
        return self.stored_water

    def is_terminated(self) -> bool:
        """
        Checks if the simulation should terminate, based on the current stored water volume.
        The system terminates if the stored water exceeds the maximum capacity or goes below zero.
        
        Returns
        -------
        bool
            True if the system has reached a terminal state (i.e., water volume is out of bounds),
            False otherwise.
        """
        return self.stored_water > self.max_capacity or self.stored_water < 0

    def determine_time_idx(self) -> int:
        """
        Determines the index corresponding to the current date's time step for evaporation calculations.
        
        The method calculates the time index based on the timestep size, which could be in months,
        days, or hours. This is used to retrieve the correct evaporation rate for the current time step.
        
        Returns
        -------
        int
            The index corresponding to the current time step for evaporation.
            
        Raises
        ------
        ValueError
            If the timestep size is unsupported (i.e., not months, days, or hours).
        """
        if self.evap_rates_timestep.months > 0:
            return self.current_date.month - 1
        elif self.evap_rates_timestep.days > 0:
            return self.current_date.timetuple().tm_yday - 1
        elif self.evap_rates_timestep.hours > 0:
            return (self.current_date.timetuple().tm_yday - 1) * 24 + self.current_date.hour - 1
        else:
            raise ValueError('The timestep is not supported, only time series with intervals of months, days, hours are supported')

    def storage_to_level(self, s: float) -> float:
        """
        Converts a storage volume in cubic meters to the corresponding water level in meters
        based on the storage-to-level relationship.
        
        Parameters
        ----------
        s : float
            The storage volume (in cubic meters) to convert to a water level.
            
        Returns
        -------
        float
            The corresponding water level (in meters) for the given storage volume.
        """
        return self.modified_interp(s, self.storage_to_level_rel[0], self.storage_to_level_rel[1])
    
    def storage_to_level_pump(self, s: float) -> float:
        """
        Converts a storage volume in cubic meters to the corresponding water level in the pump
        based on the storage-to-level relationship for the pump.
        
        Parameters
        ----------
        s : float
            The storage volume in the pump (in cubic meters) to convert to a water level.
            
        Returns
        -------
        float
            The corresponding water level (in meters) for the given pump storage volume.
        """
        return self.modified_interp(s, self.storage_to_level_rel_pump[0], self.storage_to_level_rel_pump[1])


    def storage_to_surface(self, s: float) -> float:
        """
        Converts a storage volume in cubic meters to the corresponding surface area
        based on the storage-to-surface relationship.
        
        Parameters
        ----------
        s : float
            The storage volume (in cubic meters) to convert to a surface area.
            
        Returns
        -------
        float
            The corresponding surface area for the given storage volume.
        """
        return self.modified_interp(s, self.storage_to_surface_rel[0], self.storage_to_surface_rel[1])
    
    def storage_to_surface_pump(self, s: float) -> float:
        """
        Converts a storage volume in cubic meters to the corresponding surface area
        in the pump, based on the storage-to-surface relationship for the pump.
        
        Parameters
        ----------
        s : float
            The storage volume in the pump (in cubic meters) to convert to a surface area.
            
        Returns
        -------
        float
            The corresponding surface area in the pump for the given storage volume.
        """
        return self.modified_interp(s, self.storage_to_surface_rel_pump[0], self.storage_to_surface_rel_pump[1])


    def level_to_minmax(self, h) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts a given water level (height) to the corresponding minimum and maximum release rates.
        
        Parameters
        ----------
        h : float
            The water level (height) for which to determine the min and max release rates.
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays: the minimum and maximum release rates corresponding to the given water level.
        """
        return (
            np.interp(h, self.rating_curve[0], self.rating_curve[1]),
            np.interp(h, self.rating_curve[0], self.rating_curve[2]),
        )

    def storage_to_minmax(self, s) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts a given storage volume to the corresponding minimum and maximum release rates.
        
        Parameters
        ----------
        s : float
            The storage volume (in cubic meters) for which to determine the min and max release rates.
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays: the minimum and maximum release rates corresponding to the given storage volume.
        """
        return (
            np.interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[1]),
            np.interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[2]),
        )

    @staticmethod
    def modified_interp(x: float, xp: float, fp: float, left=None, right=None) -> float:
        """
        A helper function that performs linear interpolation with modified behavior for handling
        values outside the interpolation range.
        
        Parameters
        ----------
        x : float
            The point at which to interpolate.
        xp : list[float]
            The list of x-values (storage or height).
        fp : list[float]
            The list of y-values (corresponding water levels or release rates).
        left : optional
            The value to return if x is below the range of xp.
        right : optional
            The value to return if x is above the range of xp.
            
        Returns
        -------
        float
            The interpolated value.
        """
        fp = np.asarray(fp)
        dim = len(xp) - 1
        if x <= xp[0]:
        # if x is smaller than the smallest value on X, interpolate between the first two values
            y = (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]) + fp[0]
            return y
        elif x >= xp[dim]:
        # if x is larger than the largest value, interpolate between the the last two values
            y = fp[dim] + (fp[dim] - fp[dim - 1]) / (xp[dim] - xp[dim - 1]) * (
            x - xp[dim])  # y = Y[dim]
            return y
        else:
            return compiled_interp(x, xp, fp, left, right)

    def reset(self) -> None:
        """
        Resets the system to its initial state.

        Returns
        -------
        None
            This method does not return any value, but it modifies the internal state of the system.
        """
        super().reset()
        stored_water = self.storage_vector[0]
        self.storage_vector = [stored_water]
        self.stored_water = stored_water
        self.level_vector = [self.storage_to_level(stored_water)]
        self.release_vector = []
