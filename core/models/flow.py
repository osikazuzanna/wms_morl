from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Union, Optional
from core.models.facility import Facility, ControlledFacility
from gymnasium.core import ObsType


class Flow:

    """
    A class to model water flow between sources and destinations with optional evaporation, delay, and capacity constraints.

    The Flow class simulates water transfer from source facilities to destination facilities, taking into account
    factors such as maximum capacity, evaporation rate, and delay in flow arrival.

    Attributes
    ----------
    name : str
        The name of the flow.
    sources : list[Union[Facility, ControlledFacility]]
        A list of source facilities providing the flow.
    destinations : Union[Facility, ControlledFacility, dict[Facility | ControlledFacility, float]]
        A dictionary where keys are destination facilities and values are inflow ratios (weights),
        or a single destination facility if only one is present.
    max_capacity : float
        The maximum capacity of the flow, which, if exceeded, may trigger truncation or termination.
    evaporation_rate : float, optional
        A rate at which water evaporates during the flow, expressed as a decimal fraction (default is 0.0).
    delay : int, optional
        Delay in the flow, measured in timesteps (default is 0).
    default_outflow : Optional[float], optional
        Default outflow rate when delayed (default is None).
    current_date : Optional[datetime]
        Current date in the simulation, used to manage time-based calculations.
    timestep_size : Optional[relativedelta]
        The size of each simulation timestep.
    timestep : int
        Current timestep in the simulation.
    """

    def __init__(
        self,
        name: str,
        sources: list[Union[Facility, ControlledFacility]],
        destinations: Facility | ControlledFacility | dict[Facility | ControlledFacility, float],
        max_capacity: float,
        evaporation_rate: float = 0.0,
        delay: int = 0,
        default_outflow: Optional[float] = None,
    ) -> None:
        """
        Initializes a Flow object with specified sources, destinations, and flow properties.

        Parameters
        ----------
        name : str
            The name of the flow.
        sources : list[Union[Facility, ControlledFacility]]
            A list of source facilities for the flow.
        destinations : Facility | ControlledFacility | dict[Facility | ControlledFacility, float]
            Destinations with inflow ratios or a single destination facility.
        max_capacity : float
            Maximum capacity of the flow.
        evaporation_rate : float, optional
            Evaporation rate for the flow (default is 0.0).
        delay : int, optional
            Delay in timesteps before flow reaches the destination (default is 0).
        default_outflow : Optional[float], optional
            Default outflow rate during delay periods (default is None).
        """
        self.name: str = name
        self.sources: list[Union[Facility, ControlledFacility]] = sources

        if isinstance(destinations, Facility) or isinstance(destinations, ControlledFacility):
            self.destinations = {destinations: 1.0}
        else:
            self.destinations: dict[Union[Facility, ControlledFacility], float] = destinations

        self.max_capacity: float = max_capacity
        self.evaporation_rate: float = evaporation_rate

        self.delay: int = delay
        self.default_outflow: Optional[float] = default_outflow

        self.current_date: Optional[datetime] = None
        self.timestep_size: Optional[relativedelta] = None
        self.timestep: int = 0

    def determine_source_outflow(self) -> float:
        """
        Calculates the total outflow from all sources, considering delay and default outflow conditions.

        Returns
        -------
        float
            The total outflow from sources after applying delay and any default outflow conditions.
        """
        if self.timestep - self.delay < 0 and self.default_outflow:
            return self.default_outflow
        else:
            timestep_after_delay_clipped = max(0, self.timestep - self.delay)

            return sum(source.get_outflow(timestep_after_delay_clipped) for source in self.sources)

    def determine_source_outflow_by_destination(self, destination_index: int, destination_inflow_ratio: float) -> float:
        """
        Calculates source outflow specific to a destination, adjusting for custom split policies and inflow ratios.

        Parameters
        ----------
        destination_index : int
            Index of the destination in the list of destinations.
        destination_inflow_ratio : float
            The ratio of total inflow directed to this destination.

        Returns
        -------
        float
            The calculated source outflow for the specified destination.
        """
        if self.timestep - self.delay < 0 and self.default_outflow:
            return self.default_outflow
        else:
            timestep_after_delay_clipped = max(0, self.timestep - self.delay)
            total_source_outflow = 0

            # Calculate each source contribution to the destination
            for source in self.sources:
                source_outflow = source.get_outflow(timestep_after_delay_clipped)

                # Determine if source has custom split policy
                if source.split_release:
                    total_source_outflow += source_outflow * source.split_release[destination_index]
                else:
                    total_source_outflow += source_outflow * destination_inflow_ratio

            return total_source_outflow

    def set_destination_inflow(self) -> None:
        """
        Sets the inflow for each destination based on calculated source outflow and evaporation rates.
        """
        for destination_index, (destination, destination_inflow_ratio) in enumerate(self.destinations.items()):
            destination_inflow = self.determine_source_outflow_by_destination(
                destination_index, destination_inflow_ratio
            )

            destination.set_inflow(self.timestep, destination_inflow * (1.0 - self.evaporation_rate))

    def is_truncated(self) -> bool:
        """
        Determines if the flow is truncated.

        Returns
        -------
        bool
            Always returns False in the base class.
        """
        return False

    def determine_info(self) -> dict:
        """
        Returns information about the flow.

        Returns
        -------
        dict
            A dictionary containing the flow name and the current flow rate.
        """
        return {"name": self.name, "flow": self.determine_source_outflow()}

    def step(self) -> tuple[Optional[ObsType], float, bool, bool, dict]:

        """
        Advances the simulation by one timestep, updating destination inflows and checking termination and truncation.

        Returns
        -------
        tuple[Optional[ObsType], float, bool, bool, dict]
            A tuple containing the observation, reward, termination status, truncation status, and additional information.
        """
        self.set_destination_inflow()

        terminated = self.determine_source_outflow() > self.max_capacity
        truncated = self.is_truncated()
        reward = float("-inf") if terminated else 0.0 
        info = self.determine_info()

        self.timestep += 1

        return None, reward, terminated, truncated, info

    def reset(self) -> None:
        """
        Resets the simulation, setting the timestep to zero.
        """
        self.timestep = 0


class Inflow(Flow):
    """
    A subclass of Flow representing a fixed inflow over time.

    Attributes
    ----------
    all_inflow : list[float]
        A list containing inflow values for each timestep.
    """
    def __init__(
        self,
        name: str,
        destinations: Facility | ControlledFacility | dict[Facility | ControlledFacility, float],
        max_capacity: float,
        all_inflow: list[float],
        evaporation_rate: float = 0.0,
        delay: int = 0,
        default_outflow: Optional[float] = None,
    ) -> None:
        """
        Initializes an Inflow object with predefined inflow values.

        Parameters
        ----------
        name : str
            The name of the inflow.
        destinations : Facility | ControlledFacility | dict[Facility | ControlledFacility, float]
            Destinations for the inflow.
        max_capacity : float
            Maximum capacity of the inflow.
        all_inflow : list[float]
            List of inflow values for each timestep.
        evaporation_rate : float, optional
            Evaporation rate for the inflow (default is 0.0).
        delay : int, optional
            Delay before inflow reaches destination (default is 0).
        default_outflow : Optional[float], optional
            Default outflow when delayed (default is None).
        """
        super().__init__(name, None, destinations, max_capacity, evaporation_rate, delay, default_outflow)
        self.all_inflow: list[float] = all_inflow

    def determine_source_outflow(self) -> float:
        """
        Determines the inflow for the current timestep, taking into account delay and default outflow.

        If the delay period has not been reached, returns the default outflow value if provided.
        Otherwise, it calculates the inflow using the all_inflow list, looping back if the timestep 
        exceeds the length of all_inflow.

        Returns
        -------
        float
            The inflow amount for the current timestep, or the default outflow during the delay period.
        """
        if self.timestep - self.delay < 0 and self.default_outflow:
            return self.default_outflow
        else:
            timestep_after_delay_clipped = max(0, self.timestep - self.delay) % len(self.all_inflow)

            return self.all_inflow[timestep_after_delay_clipped]

    def determine_source_outflow_by_destination(self, destination_index: int, destination_inflow_ratio: float) -> float:
        """
        Determines the inflow for a specific destination, applying destination ratios and considering delay.

        If the delay period has not been reached, returns the default outflow value if provided.
        Otherwise, calculates the inflow to the destination by applying the inflow ratio 
        to the current timestep’s inflow value.

        Parameters
        ----------
        destination_index : int
            Index of the destination in the list of destinations.
        destination_inflow_ratio : float
            Ratio of the total inflow directed to this destination.

        Returns
        -------
        float
            The amount of inflow directed to the specified destination for the current timestep.
        """        
        if self.timestep - self.delay < 0 and self.default_outflow:
            return self.default_outflow
        else:
            timestep_after_delay_clipped = max(0, self.timestep - self.delay) % len(self.all_inflow)

            return self.all_inflow[timestep_after_delay_clipped] * destination_inflow_ratio

    def is_truncated(self) -> bool:
        """
        Checks if the inflow sequence has reached its end, indicating a truncated state.

        Returns
        -------
        bool
            True if the timestep has exceeded the length of the all_inflow list, otherwise False.
        """
        return self.timestep >= len(self.all_inflow)


class Outflow(Flow):
    """
    A subclass of Flow specifically for outflow management, without any set destinations.

    Attributes
    ----------
    None

    """
    def __init__(
        self,
        name: str,
        sources: list[Union[Facility, ControlledFacility]],
        max_capacity: float,
    ) -> None:
        """
        Initializes an Outflow object without destinations.

        Parameters
        ----------
        name : str
            The name of the outflow.
        sources : list[Union[Facility, ControlledFacility]]
            List of source facilities.
        max_capacity : float
            Maximum capacity of the outflow.
        """
        super().__init__(name, sources, None, max_capacity)

    def set_destination_inflow(self) -> None:
        """Overrides the destination inflow method to do nothing, as Outflow has no destinations."""
        pass
