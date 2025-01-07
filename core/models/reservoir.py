from core.models.facility import ControlledFacility
from gymnasium.spaces import Box, Space
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
from numpy.core.multiarray import interp as compiled_interp


class Reservoir(ControlledFacility):
    """
    A class representing a reservoir within a water management system.

    This class is used to model the storage, release, and evaporation processes of a reservoir, including
    the management of water levels, inflows, outflows, and associated evaporation. The reservoir interacts
    with the surrounding environment and is controlled by various parameters such as release rates, storage,
    and evaporation rates.

    Attributes
    ----------
    name : str
        A lowercase, non-spaced name for the reservoir.
    stored_water : float
        The current volume of water stored in the reservoir (in m³).
    evap_rates : np.array
        Monthly evaporation rates (in cm).
    evap_rates_timestep : relativedelta
        The timestep for evaporation rate calculation.
    storage_to_minmax_rel : list[list[float]]
        Relationship between storage and minimum/maximum release rates.
    storage_to_level_rel : list[list[float]]
        Relationship between storage and water level (height).
    storage_to_surface_rel : list[list[float]]
        Relationship between storage and surface area of the reservoir.
    storage_vector : list[float]
        List tracking the volume of water in the reservoir over time.
    level_vector : list[float]
        List tracking the elevation (height) of the water in the reservoir over time.
    release_vector : list[float]
        List tracking the actual water release per timestep.
    integration_timestep_size : relativedelta
        The timestep size used for numerical integration of reservoir processes.
    spillage : float
        Amount of water lost due to spillage.
    objective_function : callable
        The objective function to evaluate the reservoir’s performance (based on stored water).
    objective_name : str
        The name of the objective function.
    """

    def __init__(
        self,
        name: str,
        max_capacity: float,
        max_action: list[float],
        objective_function,
        integration_timestep_size: relativedelta,
        evap_rates: list[float],
        evap_rates_timestep_size: relativedelta,
        storage_to_minmax_rel: list[list[float]],
        storage_to_level_rel: list[list[float]],
        storage_to_surface_rel: list[list[float]],
        objective_name: str = "",
        stored_water: float = 0,
        spillage: float = 0,
        observation_space = Box(low=0, high=1),
        action_space = Box(low=0, high=1),

                ) -> None:
        """
        Initializes a Reservoir instance with the specified parameters.

        Args:
            name (str): The name of the reservoir.
            max_capacity (float): The maximum capacity of the reservoir.
            max_action (list[float]): Maximum release actions for the reservoir.
            objective_function (callable): Function that evaluates the reservoir’s objective.
            integration_timestep_size (relativedelta): Timestep for integrating the reservoir’s processes.
            evap_rates (list[float]): Monthly evaporation rates in cm.
            evap_rates_timestep_size (relativedelta): Time step size for the evaporation rates.
            storage_to_minmax_rel (list[list[float]]): Relationship between storage and release rates.
            storage_to_level_rel (list[list[float]]): Relationship between storage and water level.
            storage_to_surface_rel (list[list[float]]): Relationship between storage and surface area.
            objective_name (str, optional): The name of the objective function.
            stored_water (float, optional): The initial amount of stored water.
            spillage (float, optional): The amount of spillage.
            observation_space (Box, optional): The observation space for the environment.
            action_space (Box, optional): The action space for the environment.
        """
        super().__init__(name, observation_space, action_space, max_capacity, max_action)
        self.stored_water: float = stored_water

        self.evap_rates = evap_rates
        self.evap_rates_timestep = evap_rates_timestep_size
        self.storage_to_minmax_rel = storage_to_minmax_rel
        self.storage_to_level_rel = storage_to_level_rel
        self.storage_to_surface_rel = storage_to_surface_rel
        
        self.storage_vector = []
        self.level_vector = []
        self.release_vector = []

        # Initialise storage vector
        self.storage_vector.append(stored_water)

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.integration_timestep_size: relativedelta = integration_timestep_size
        self.spillage = spillage
        # self.water_level = self.storage_to_level(self.stored_water)

    def determine_reward(self) -> float:
        # Pass water level to reward function
        """
        Calculates the reward for the reservoir based on its current stored water level.

        This method uses the objective function to evaluate the reward based on the water level, which is derived
        from the stored water volume.

        Returns:
            float: The calculated reward for the reservoir.
        """
        return self.objective_function(self.storage_to_level(self.stored_water))

    def determine_outflow(self, actions: np.array) -> list[float]:
        """
        Determines the average monthly water release from the reservoir based on the given actions.

        The release is adjusted according to the current storage level, evaporation rates, and actions taken.

        Args:
            actions (np.array): Array of release actions that control the reservoir's water release.

        Returns:
            list[float]: The average release of water per timestep.
        """
        current_storage = self.storage_vector[-1]
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

        while self.current_date < final_date:
            next_date = min(final_date, self.current_date + self.integration_timestep_size)
            integration_time_seconds = (next_date - self.current_date).total_seconds()

            #calculate the surface to get the evaporation
            surface = self.storage_to_surface(current_storage)
            #evaporation per integration timestep
            evaporation = surface * (evaporatio_rate_per_second * integration_time_seconds)
            #get min and max possible release based on the current storage
            min_possible_release, max_possible_release = self.storage_to_minmax(current_storage)
            #release per second is calculated based on the min-max releases which depend on the storage level and the predicted actions
            release_per_second = min(max_possible_release, max(min_possible_release, np.sum(actions)))

            sub_releases = np.append(sub_releases, release_per_second)

            total_addition = self.get_inflow(self.timestep) * integration_time_seconds

            current_storage += total_addition - evaporation - (np.sum(release_per_second) - self.spillage) * integration_time_seconds
            
            self.current_date = next_date

        # Update the amount of water in the Reservoir
        self.storage_vector.append(current_storage)
        self.stored_water = current_storage

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
        Returns information about the current state of the reservoir.

        This includes details such as the current stored water volume, water level, release, and evaporation rates.

        Returns:
            dict: A dictionary containing key information about the reservoir (e.g., name, stored water, 
                current level, current release, evaporation rates).
        """
        info = {
            "name": self.name,
            "stored_water": self.stored_water,
            "current_level": self.level_vector[-1] if self.level_vector else None,
            "current_release": self.release_vector[-1] if self.release_vector else None,
            "evaporation_rates": self.evap_rates.tolist(),
        }
        return info

    def determine_observation(self) -> float:
        """
        Retrieves the current observation of the reservoir, which is the amount of stored water.

        Returns:
            float: The current stored water in the reservoir (in m³). If no water is stored, returns 0.
        """
        if self.stored_water > 0:
            return self.stored_water
        else:
            return 0.0

    def is_terminated(self) -> bool:
        """
        Checks if the reservoir's simulation has reached a termination condition.

        The simulation is terminated if the stored water exceeds the maximum capacity or falls below 0.

        Returns:
            bool: True if the simulation should be terminated (due to overflow or depletion), otherwise False.
        """
        return self.stored_water > self.max_capacity or self.stored_water < 0
    


    def determine_time_idx(self) -> int:
        """
        Determines the index for the evaporation rate based on the current date and timestep.

        The function maps the current date to the appropriate index in the evaporation rate array, considering
        the timestep size (months, days, or hours).

        Returns:
            int: The index corresponding to the current date in the evaporation rate array.

        Raises:
            ValueError: If the timestep size is unsupported (i.e., not months, days, or hours).
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
        Converts a given storage value to the corresponding water level (height) based on predefined relationships.

        Args:
            s (float): The storage value (in m³) to be converted.

        Returns:
            float: The corresponding water level (height) in meters.
        """
        return self.modified_interp(s, self.storage_to_level_rel[0], self.storage_to_level_rel[1])

    def storage_to_surface(self, s: float) -> float:
        """
        Converts a given storage value to the corresponding surface area based on predefined relationships.

        Args:
            s (float): The storage value (in m³) to be converted.

        Returns:
            float: The corresponding surface area (in m²).
        """
        return self.modified_interp(s, self.storage_to_surface_rel[0], self.storage_to_surface_rel[1])

    def level_to_minmax(self, h) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the minimum and maximum possible release rates based on the given water level.

        Args:
            h (float): The water level (height) to calculate the release rates for.

        Returns:
            tuple: A tuple containing two arrays - the minimum and maximum release rates.
        """
        return (
            np.interp(h, self.rating_curve[0], self.rating_curve[1]),
            np.interp(h, self.rating_curve[0], self.rating_curve[2]),
        )

    def storage_to_minmax(self, s) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the minimum and maximum possible release rates based on the given storage.

        Args:
            s (float): The storage value (in m³) to calculate the release rates for.

        Returns:
            tuple: A tuple containing two arrays - the minimum and maximum release rates.
        """
        return (
            np.interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[1]),
            np.interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[2]),
        )

    @staticmethod
    def modified_interp(x: float, xp: float, fp: float, left=None, right=None) -> float:
        """
        Performs linear interpolation between two points with custom handling for out-of-bound values.

        Args:
            x (float): The input value to interpolate.
            xp (float): The known input values for interpolation.
            fp (float): The corresponding output values for interpolation.
            left (optional): The value to return if x is less than the smallest value in xp.
            right (optional): The value to return if x is greater than the largest value in xp.

        Returns:
            float: The interpolated value based on the provided input values.
        """
        fp = np.asarray(fp)

        return compiled_interp(x, xp, fp, left, right)

    # def modified_interp(x: float, xp: float, fp: float, left=None, right=None) -> float:
    #     fp = np.asarray(fp)
    #     dim = len(xp) - 1
    #     if x <= xp[0]:
    #     # if x is smaller than the smallest value on X, interpolate between the first two values
    #         y = (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]) + fp[0]
    #         return y
    #     elif x >= xp[dim]:
    #     # if x is larger than the largest value, interpolate between the the last two values
    #         y = fp[dim] + (fp[dim] - fp[dim - 1]) / (xp[dim] - xp[dim - 1]) * (
    #         x - xp[dim])  # y = Y[dim]
    #         return y
    #     else:
    #         return compiled_interp(x, xp, fp, left, right)


    def reset(self) -> None:
        """
        Resets the state of the reservoir to its initial conditions for a new simulation.

        This method resets the storage, release, and level vectors, and restores the initial stored water volume.
        """
        super().reset()
        stored_water = self.storage_vector[0]
        self.storage_vector = [stored_water]
        self.stored_water = stored_water
        self.level_vector = []
        self.release_vector = []
