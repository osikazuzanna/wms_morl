from core.models.facility import Facility


class Catchment(Facility):
    """
    A class representing a catchment facility that accumulates water over time and manages inflows and outflows.

    Attributes
    ----------
    name : str
        The identifier for the catchment facility.
    all_water_accumulated : list[float]
        List of water accumulation values over time.
    """

    def __init__(self, name: str, all_water_accumulated: list[float]) -> None:
        """
        Initializes a Catchment instance with the given name and water accumulation data.

        Args:
            name (str): The identifier for the catchment facility.
            all_water_accumulated (list[float]): A list of accumulated water values over time.
        """
        super().__init__(name)
        self.all_water_accumulated: list[float] = all_water_accumulated

    def determine_reward(self) -> float:
        """
        Determines the reward for the catchment. In this implementation, the reward is always 0.

        Returns:
            float: The reward for the current timestep, always 0 in this case.
        """
        return 0

    def get_inflow(self, timestep: int) -> float:
        """
        Retrieves the inflow value for a given timestep.

        Args:
            timestep (int): The timestep index for which to retrieve the inflow value.

        Returns:
            float: The inflow value for the specified timestep, based on the water accumulation list.
            
        Notes:
            If the timestep exceeds the length of `all_water_accumulated`, it wraps around using modulus.
        """
        return self.all_water_accumulated[timestep % len(self.all_water_accumulated)]

    def determine_consumption(self) -> float:
        """
        Determines the water consumption for the catchment. In this implementation, consumption is always 0.

        Returns:
            float: The water consumption for the current timestep, always 0 in this case.
        """        
        return 0

    def is_truncated(self) -> bool:
        """
        Determines if the simulation should be truncated based on the timestep.

        Returns:
            bool: True if the current timestep is greater than or equal to the length of `all_water_accumulated`,
                indicating the simulation should be truncated, otherwise False.
        """
        return self.timestep >= len(self.all_water_accumulated)

    def determine_info(self) -> dict:
        """
        Returns information about the catchment's state, particularly the water consumption.

        Returns:
            dict: A dictionary containing the water consumption, which is always 0 in this case.
        """
        return {"water_consumption": self.determine_consumption()}
