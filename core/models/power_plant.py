from core.models.facility import Facility
from core.models.reservoir import Reservoir
from core.utils import utils
from scipy.constants import g
import numpy as np


class PowerPlant(Facility):
    """
    Class to represent a Hydro-energy Powerplant.

    This class models a hydroelectric power plant, including its power production capabilities,
    water consumption, and the relationship between water levels and energy generation.

    Attributes
    ----------
    name : str
        Identifier for the power plant.
    efficiency : float
        Efficiency coefficient (mu) used in hydropower formula.
    max_turbine_flow : float
        Maximum possible flow that can be passed through the turbines for the purpose of hydroenergy production.
    head_start_level : float
        Minimum elevation of water level that is used to calculate hydraulic head for hydropower production.
    max_capacity : float
        Total design capacity (mW) of the plant.
    water_level_coeff : float
        Coefficient that determines the water level based on the volume of outflow.
    water_usage : float
        Fraction of water used by the power plant for production.
    production_vector : np.ndarray
        Vector that stores the power production history of the plant.
    tailwater : np.array
        Array of tailwater data used for detailed power production calculations.
    turbines : np.array
        Array representing turbine flow data.
    n_turbines : int
        Number of turbines in the power plant.
    energy_prices : np.array
        Energy price data used for production cost calculations.
    """

    def __init__(
        self,
        name: str,
        objective_function,
        objective_name: str,
        efficiency: float,
        min_turbine_flow: float = 0.0,
        normalize_objective: float = 0.0,
        max_turbine_flow: float = 0.0,
        head_start_level: float = 0.0,
        max_capacity: float = 0.0,
        reservoir: Reservoir = None,
        water_usage: float = 0.0,
        tailwater: np.array = None,
        turbines: np.array = None,
        n_turbines: int = 0,
        energy_prices: np.array = None
    ) -> None:
        """
        Initializes a Hydro-energy Powerplant object.

        Parameters
        ----------
        name : str
            The name/identifier for the power plant.
        objective_function : callable
            Function to calculate the reward based on the power production.
        objective_name : str
            Name of the objective.
        efficiency : float
            Efficiency of the plant in converting water flow to energy.
        min_turbine_flow : float, optional
            Minimum turbine flow, default is 0.0.
        normalize_objective : float, optional
            Normalization factor for the objective function, default is 0.0.
        max_turbine_flow : float, optional
            Maximum turbine flow, default is 0.0.
        head_start_level : float, optional
            Minimum water level required for generating power, default is 0.0.
        max_capacity : float, optional
            Maximum design capacity of the plant in MW, default is 0.0.
        reservoir : Reservoir, optional
            Reservoir object associated with the power plant, default is None.
        water_usage : float, optional
            Water usage factor for the plant, default is 0.0.
        tailwater : np.array, optional
            Tailwater data for detailed turbine calculations, default is None.
        turbines : np.array, optional
            Turbine capacity data, default is None.
        n_turbines : int, optional
            Number of turbines in the plant, default is 0.
        energy_prices : np.array, optional
            Array of energy price data, default is None.
        """
        super().__init__(name, objective_function, objective_name, normalize_objective)
        self.efficiency: float = efficiency
        self.max_turbine_flow: float = max_turbine_flow
        self.head_start_level: float = head_start_level
        self.min_turbine_flow: float = min_turbine_flow
        self.max_capacity: float = max_capacity
        self.reservoir: Reservoir = reservoir
        self.water_usage: float = water_usage
        self.production_vector: np.ndarray = np.empty(0, dtype=np.float64)
        self.tailwater = tailwater
        self.turbines = turbines
        self.n_turbines = n_turbines
        self.energy_prices = energy_prices

    def determine_turbine_flow(self) -> float:
        """
        Determines the flow through the turbines, bounded by the minimum and maximum turbine flow.

        Returns
        -------
        float
            Flow through the turbines, in cubic meters per second.
        """
        return max(self.min_turbine_flow, min(self.max_turbine_flow, self.get_inflow(self.timestep)))

    # Constants are configured as parameters with default values
    def determine_production(self) -> float:
        """
        Calculates the power production (in MWh) for the plant when detailed turbine and tailwater data is not available.

        The calculation is based on the turbine flow, hydraulic head, and efficiency of the plant.

        Returns
        -------
        float
            The power production of the plant in MWh for the current timestep.
        """
        m3_to_kg_factor: int = 1000
        w_Mw_conversion: float = 1e-6
        # Turbine flow is equal to outflow, as long as it does not exceed maximum turbine flow
        turbine_flow = self.determine_turbine_flow()

        # Uses water level from reservoir to determine water level
        water_level = self.reservoir.level_vector[-1] if self.reservoir.level_vector else 0
        # Calculate at what level the head will generate power, using water_level of the outflow and head_start_level
        head = max(0.0, water_level - self.head_start_level)

        # Calculate power in mW, has to be lower than or equal to capacity
        power_in_mw = min(
            self.max_capacity,
            turbine_flow * head * m3_to_kg_factor * g * self.efficiency * w_Mw_conversion,
        )

        # Calculate the numbe rof hours the power plant has been running.
        final_date = self.current_date + self.timestep_size
        timestep_hours = (final_date - self.current_date).total_seconds() / 3600

        # Hydro-energy power production in mWh
        production = power_in_mw * timestep_hours
        self.production_vector = np.append(self.production_vector, production)

        return production
    



    def determine_production_detailed(self) -> float:
        """
        Calculates the power production (in MWh) when detailed turbine and tailwater data is available.

        This method takes into account the turbine characteristics and the tailwater level to compute 
        more precise power production.

        Returns
        -------
        float
            The detailed power production of the plant in MWh for the current timestep.
        """

        cubicFeetToCubicMeters = 0.0283  # 1 cf = 0.0283 m3
        feetToMeters = 0.3048  # 1 ft = 0.3048 m
        m3_to_kg_factor = 1000
        p = 0.0
        water_level = self.reservoir.level_vector[-1] if self.reservoir.level_vector else 0
        turbine_flow = self.determine_turbine_flow()

        deltaH = water_level - utils.interpolate_tailwater_level(
            self.tailwater[0], self.tailwater[1], turbine_flow
        )

        q_split = turbine_flow

        for j in range(0, self.n_turbines):
            if q_split < self.turbines[1][j]:
                qturb = 0.0
            elif q_split > self.turbines[0][j]:
                qturb = self.turbines[0][j]
            else:
                qturb = q_split
            q_split = q_split - qturb
            p = p + (
                self.efficiency
                * g
                * m3_to_kg_factor
                * (cubicFeetToCubicMeters * qturb)
                * (feetToMeters * deltaH)
                * 3600
                / (3600 * 1000)
            )  


        # Calculate the numbe rof hours the power plant has been running.
        final_date = self.current_date + self.timestep_size
        timestep_hours = (final_date - self.current_date).total_seconds() / 3600

        production = p * timestep_hours
        self.production_vector = np.append(self.production_vector, production)

        return production



    def determine_reward(self) -> float:
        """
        Calculates the reward based on the power production of the plant.

        The reward is determined by evaluating the power production through either the general production
        method or the detailed method based on the availability of tailwater and turbine data.

        Returns
        -------
        float
            The calculated reward based on the power production.
        """

        if self.turbines is not None and self.tailwater is not None:
            return self.objective_function(self.determine_production_detailed())
        else:
            return self.objective_function(self.determine_production())

    def determine_consumption(self) -> float:
        """
        Calculates the water consumption of the plant.

        The consumption is based on the turbine flow and the water usage factor.

        Returns
        -------
        float
            The amount of water consumed by the plant in cubic meters per second.
        """
        return self.determine_turbine_flow() * self.water_usage

    def determine_info(self) -> dict:
        """
        Returns key information about the hydro-energy power plant.

        This includes the name, inflow, outflow, monthly production, water usage, and total production.

        Returns
        -------
        dict
            A dictionary with the power plant's key information.
        """
        return {
            "name": self.name,
            "inflow": self.get_inflow(self.timestep),
            "outflow": self.get_outflow(self.timestep),
            "monthly_production": self.production_vector[-1],
            "water_usage": self.water_usage,
            "total production (MWh)": sum(self.production_vector),
        }

    def determine_month(self) -> int:
        """
        Returns the current month based on the timestep.

        The method assumes the timestep is provided as a monthly index.

        Returns
        -------
        int
            The current month as an integer (0-11).
        """
        return self.timestep % 12

    def reset(self) -> None:
        """
        Resets the power plant state, including clearing the power production history.

        Returns
        -------
        None
            This method does not return a value but modifies the internal state.
        """
        super().reset()
        self.production_vector = np.empty(0, dtype=np.float64)
