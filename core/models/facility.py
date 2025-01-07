import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from dateutil.relativedelta import relativedelta
from gymnasium.spaces import Space, Box
from gymnasium.core import ObsType, ActType
from typing import SupportsFloat, Optional
from core.models.objective import Objective


class Facility(ABC):
    """
    Abstract base class representing a facility with inflow and outflow management, along with reward determination.

    Attributes
    ----------
    name : str
        Identifier for the facility.
    all_inflow : list[float]
        Historical inflow values recorded over time.
    all_outflow : list[float]
        Historical outflow values recorded over time.
    objective_function : Callable
        Function to evaluate the facility’s performance based on defined objectives.
    objective_name : str
        Name of the objective function used.
    current_date : Optional[datetime]
        Date associated with the current timestep of the facility.
    timestep_size : Optional[relativedelta]
        Size of the timestep for simulation. Usually 1 month.
    timestep : int
        Current timestep index for the facility simulation.
    split_release : Optional
        Placeholder for managing release strategies (usage to be defined).
    normalize_objective : float
        Normalization factor for the objective reward.
    """
    def __init__(self, name: str, objective_function=Objective.no_objective, objective_name: str = "", normalize_objective=0.0) -> None:
        """
        Initializes a Facility instance.

        Args:
            name (str): Identifier for the facility.
            objective_function (Callable): Function to evaluate the facility’s performance.
            objective_name (str): Name of the objective function.
            normalize_objective (float): Maximum value for normalizing the reward; defaults to 0.0.
        """
        self.name: str = name
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.current_date: Optional[datetime] = None
        self.timestep_size: Optional[relativedelta] = None
        self.timestep: int = 0

        self.split_release = None
        self.normalize_objective = normalize_objective

    @abstractmethod
    def determine_reward(self) -> float:
        """
        Abstract method to compute the reward based on the facility's performance.

        Returns:
            float: The computed reward for the current timestep.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def determine_consumption(self) -> float:
        """
        Abstract method to calculate the facility's water consumption.

        Returns:
            float: The calculated water consumption for the current timestep.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError()

    def is_terminated(self) -> bool:
        """
        Checks if the facility simulation has reached a terminating condition.

        Returns:
            bool: Always returns False for the base Facility class; override in subclasses as needed.
        """
        return False

    def is_truncated(self) -> bool:
        """
        Checks if the facility simulation has been truncated.

        Returns:
            bool: Always returns False for the base Facility class; override in subclasses as needed.
        """
        return False

    def get_inflow(self, timestep: int) -> float:
        """
        Retrieves the inflow value for a specific timestep.

        Args:
            timestep (int): The timestep index for which to retrieve the inflow.

        Returns:
            float: The inflow value for the specified timestep.

        Raises:
            IndexError: If the timestep is out of bounds.
        """
        return self.all_inflow[timestep]

    def set_inflow(self, timestep: int, inflow: float) -> None:
        """
        Sets the inflow value for a specific timestep, adjusting if necessary.

        Args:
            timestep (int): The timestep index at which to set the inflow.
            inflow (float): The inflow value to set.

        Raises:
            IndexError: If the timestep index is invalid.
        """
        if len(self.all_inflow) == timestep:
            self.all_inflow.append(inflow)
        elif len(self.all_inflow) > timestep:
            self.all_inflow[timestep] += inflow
        else:
            raise IndexError

    def determine_outflow(self) -> float:
        """
        Calculates the outflow based on the current inflow and consumption.

        Returns:
            float: The calculated outflow for the current timestep.
        """
        return self.get_inflow(self.timestep) - self.determine_consumption()

    def get_outflow(self, timestep: int) -> float:
        """
        Retrieves the outflow value for a specific timestep.

        Args:
            timestep (int): The timestep index for which to retrieve the outflow.

        Returns:
            float: The outflow value for the specified timestep.

        Raises:
            IndexError: If the timestep is out of bounds.
        """
        return self.all_outflow[timestep]

    def step(self) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Advances the simulation by one timestep, calculating outflows and rewards.

        Returns:
            tuple[ObsType, float, bool, bool, dict]: A tuple containing:
                - ObsType: Placeholder for observation type (to be defined).
                - float: The reward for the current timestep.
                - bool: Indicates if the simulation has terminated.
                - bool: Indicates if the simulation has been truncated.
                - dict: Additional information about the facility's state.
        """
        self.all_outflow.append(self.determine_outflow())
        # TODO: Determine if we need to satisy any terminating codnitions for facility.
        reward = self.determine_reward()
        if self.normalize_objective>0.0:
            reward = reward/self.normalize_objective
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.determine_info()

        self.timestep += 1

        return None, reward, terminated, truncated, info

    def reset(self) -> None:
        """
        Resets the facility to its initial state for a new simulation run.

        Returns:
            None
        """
        self.timestep: int = 0
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

    def determine_info(self) -> dict:
        """
        Method to gather information about the facility's state.

        Returns:
            dict: A dictionary containing information about the facility.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError()

    def __eq__(self, other):
        """
        Checks equality between two Facility instances based on their names.

        Args:
            other (Facility): The other facility to compare against.

        Returns:
            bool: True if both facilities are of the same class and have the same name, False otherwise.
        """
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        """
        Returns a hash of the facility based on its name.

        Returns:
            int: The hash value of the facility.
        """
        return hash(self.name)


class ControlledFacility(ABC):
    """
    Abstract base class representing a facility with inflow and outflow management, along with reward determination.
    
    Attributes
    ----------
    name : str
        Identifier for the controlled facility.
    all_inflow : list[float]
        Historical inflow values recorded over time.
    all_outflow : list[float]
        Historical outflow values recorded over time.
    max_capacity : float
        Maximum capacity of the facility.
    max_action : float
        Maximum allowable action for inflow/outflow control.
    observation_space : Space
        The space representing the possible observations from the facility.
    action_space : Space
        The space representing the possible actions that can be taken to control the facility.
    objective_function : Callable
        Function used to evaluate the facility's performance based on predefined objectives.
    objective_name : str
        The name of the objective function used.
    current_date : Optional[datetime]
        Date corresponding to the current timestep.
    timestep_size : Optional[relativedelta]
        Size of the simulation timestep (e.g., 1 month).
    timestep : int
        The current timestep index in the simulation.
    should_split_release : bool
        Determines whether the release needs to be split across multiple actions.
    split_release : Optional
        Placeholder for managing split release strategies (if applicable).

    """
    def __init__(
        self,
        name: str,
        observation_space: Space,
        action_space: ActType,
        max_capacity: float, 
        max_action: float,

        objective_function=Objective.no_objective,
        objective_name: str = "",
    ) -> None:
        
        """
        Initializes a ControlledFacility instance.

        Args:
            name (str): The identifier for the facility.
            observation_space (Space): The space representing possible observations from the facility.
            action_space (ActType): The space representing possible actions that control the facility.
            max_capacity (float): Maximum capacity of the facility.
            max_action (float): Maximum action value for inflow/outflow control.
            objective_function (Callable): Function to evaluate the facility’s performance (default is no objective).
            objective_name (str): Name of the objective function used (default is an empty string).
        """
        self.name: str = name
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []
        self.max_capacity: float = max_capacity
        self.max_action: float = max_action


        self.observation_space: Space = observation_space
        #check if there is more than one outflow from a reservoir
        if len(self.max_action)>1:
            self.action_space: Space = Box(low=0, high=1, shape=(len(self.max_action),))
        else:
            self.action_space: Space = action_space

        self.objective_function = objective_function
        self.objective_name = objective_name


        self.current_date: Optional[datetime] = None
        self.timestep_size: Optional[relativedelta] = None
        self.timestep: int = 0

        self.should_split_release = np.prod(self.action_space.shape) > 1
        self.split_release = None

    @abstractmethod
    def determine_reward(self) -> float:
        """
        Abstract method for calculating the reward based on the facility's performance.

        Returns:
            float: The computed reward for the current timestep.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def determine_outflow(action: ActType) -> float:
        """
        Abstract method for determining the outflow based on a given action.

        Args:
            action (ActType): The action taken to control the outflow.

        Returns:
            float: The calculated outflow based on the provided action.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def determine_observation(self) -> ObsType:
        """
        Abstract method for determining the current observation from the facility.

        Returns:
            ObsType: The current observation.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_terminated(self) -> bool:
        """
        Abstract method to check if the simulation has reached a termination condition.

        Returns:
            bool: Returns True if the simulation has reached a termination condition, False otherwise.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError()

    def is_truncated(self) -> bool:
        """
        Determines if the simulation has been truncated.

        Returns:
            bool: Always returns False for the base ControlledFacility class. Override in subclasses if needed.
        """
        return False

    def get_inflow(self, timestep: int) -> float:
        """
        Retrieves the inflow value for a given timestep.

        Args:
            timestep (int): The timestep index for which to retrieve the inflow.

        Returns:
            float: The inflow value for the specified timestep.

        Raises:
            IndexError: If the timestep is out of bounds.
        """
        return self.all_inflow[timestep]

    def set_inflow(self, timestep: int, inflow: float) -> None:
        """
        Sets the inflow value for a given timestep, either adding to or replacing the current value.

        Args:
            timestep (int): The timestep index at which to set the inflow.
            inflow (float): The inflow value to set.

        Raises:
            IndexError: If the timestep index is invalid.
        """
        if len(self.all_inflow) == timestep:
            self.all_inflow.append(inflow)
        elif len(self.all_inflow) > timestep:
            self.all_inflow[timestep] += inflow
        else:
            raise IndexError

    def get_outflow(self, timestep: int) -> float:
        """
        Retrieves the outflow value for a given timestep.

        Args:
            timestep (int): The timestep index for which to retrieve the outflow.

        Returns:
            float: The outflow value for the specified timestep.

        Raises:
            IndexError: If the timestep is out of bounds.
        """
        return self.all_outflow[timestep]

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        """
        Advances the simulation by one timestep based on the given action and computes the reward.

        Args:
            action (ActType): The action to be taken to control the facility.

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, dict]: A tuple containing:
                - ObsType: The updated observation.
                - float: The reward for the current timestep.
                - bool: Indicates if the simulation has terminated.
                - bool: Indicates if the simulation has been truncated.
                - dict: Additional information about the facility's state.
        """
        self.all_outflow.append(self.determine_outflow(action))
        # TODO: Change stored_water to multiple outflows.

        observation = self.determine_observation()
        reward = self.determine_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.determine_info()

        self.timestep += 1

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self) -> None:
        """
        Resets the facility to its initial state for a new simulation run.

        Returns:
            None
        """
        self.timestep: int = 0
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

    def determine_info(self) -> dict:
        """
        Retrieves additional information about the facility's state.

        Returns:
            dict: A dictionary containing information about the facility.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise {}

    def __eq__(self, other):
        """
        Checks equality between two ControlledFacility instances based on their names.

        Args:
            other (ControlledFacility): The other facility to compare against.

        Returns:
            bool: True if both facilities are of the same class and have the same name, False otherwise.
        """
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        """
        Returns a hash of the facility based on its name.

        Returns:
            int: The hash value of the facility.
        """
        return hash(self.name)
