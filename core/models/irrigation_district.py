from core.models.facility import Facility


class IrrigationDistrict(Facility):
    """
    Represents an irrigation district with specific water demand, consumption, and deficit tracking.

    Attributes
    ----------
    name : str
        Identifier for the irrigation district.
    all_demand : list[float]
        Monthly water demand values for the irrigation district.
    total_deficit : float
        Cumulative water deficit experienced by the district over time.
    all_deficit : list[float]
        Monthly record of deficits, calculated as demand minus consumption.
    normalize_objective : float
        Normalization factor for the objective function reward. It should be the highest monthly value in a year. Default is 0.0

    """
    def __init__(self, name: str, all_demand: list[float], objective_function, objective_name: str, normalize_objective:float = 0.0) -> None:
        """
        Initializes an Irrigation District instance.

        Parameters:
            name : str
                Identifier for the irrigation district.
            all_demand : list[float]
                Monthly water demand values for the irrigation district.
            objective_function : callable
                Function to evaluate the district’s performance.
            objective_name : str
                Name of the objective.
            normalize_objective : float, optional
                Maximum value for normalizing the objective, it should be the highest monthly demand in the whole year. By default 0.0.

        """
        super().__init__(name, objective_function, objective_name, normalize_objective)
        self.all_demand: list[float] = all_demand
        self.total_deficit: float = 0
        self.all_deficit: list[float] = []

    def get_current_demand(self) -> float:
        """
        Returns the demand value for the current timestep.

        Returns
        -------
        float
            Demand for the current timestep.
        
        """
        return self.all_demand[self.timestep % len(self.all_demand)]

    def determine_deficit(self) -> float:
        """
        Calculates the reward (irrigation deficit) given the values of its attributes

        Returns
        -------
        float
            Water deficit of the irrigation district
        """
        consumption = self.determine_consumption()
        deficit = self.get_current_demand() - consumption
        self.total_deficit += deficit
        self.all_deficit.append(deficit)
        return deficit

    def determine_reward(self) -> float:
        """
        Calculates the reward for the irrigation district based on the objective function.

        Returns
        -------
        float
            Reward as calculated by the objective function.
        """
        return self.objective_function(self.get_current_demand(), self.get_inflow(self.timestep))

    def determine_consumption(self) -> float:
        """
        Calculates the water consumption for the irrigation district based on current demand and inflow.

        Returns
        -------
        float
            Water consumption for the current timestep.
        """
        return min(self.get_current_demand(), self.get_inflow(self.timestep))

    def is_truncated(self) -> bool:
        """
        Checks if the simulation has reached the end of the demand data.

        Returns:
            bool: 
                True if the simulation has no more demand data to process, False otherwise.
        """
        return self.timestep >= len(self.all_demand)

    def determine_info(self) -> dict:
        """
        Returns information about the irrigation district.

        Returns:
            dict: 
                A dictionary containing key metrics for the district.
        """
        return {
            "name": self.name,
            "inflow": self.get_inflow(self.timestep),
            "outflow": self.get_outflow(self.timestep),
            "demand": self.get_current_demand(),
            "total_deficit": self.total_deficit,
            "list_deficits": self.all_deficit,
        }

    def reset(self) -> None:
        """
        Resets the irrigation district's deficit attributes and inherited attributes.

        Returns:
            None

        """        
        super().reset()
        self.total_deficit = 0
        self.all_deficit = []
