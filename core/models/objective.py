class Objective:
    """
    A class containing static methods representing various objective functions
    used to evaluate water management strategies based on demand, supply, and custom conditions.

    """
    @staticmethod
    def no_objective(*args):
        """
        A placeholder objective that returns a fixed value of 0.0, used when no objective function is specified.

        Parameters
        ----------
        *args
            Any arguments passed will be ignored.

        Returns
        -------
        float
            The value 0.0 as a default objective output.
        """
        return 0.0

    @staticmethod
    def identity(value: float) -> float:
        """
        Returns the input value unchanged.

        Parameters
        ----------
        value : float
            The input value to return.

        Returns
        -------
        float
            The input value itself.
        """
        return value

    @staticmethod
    def is_greater_than_minimum(minimum_value: float) -> float:
        """
        Checks if a value meets or exceeds a specified minimum value.

        Parameters
        ----------
        minimum_value : float
            The minimum threshold value.

        Returns
        -------
        float
            1.0 if the value is greater than or equal to minimum_value, 0.0 otherwise.
        """
        return lambda value: 1.0 if value >= minimum_value else 0.0

    @staticmethod
    def is_greater_than_minimum_with_condition(minimum_value: float) -> float:
        """
        Checks if a value meets a minimum threshold, contingent on a condition.

        Parameters
        ----------
        minimum_value : float
            The minimum threshold value.

        Returns
        -------
        float
            1.0 if the condition is True and value is above the minimum, 0.0 otherwise.
        """
        return lambda condition, value: 1.0 if condition and value >= minimum_value else 0.0

    @staticmethod
    def deficit_minimised(demand: float, received: float) -> float:
        """
        Returns the negative deficit between demand and received amounts, penalizing any shortfall.

        Parameters
        ----------
        demand : float
            The required demand amount.
        received : float
            The amount actually received.

        Returns
        -------
        float
            Negative deficit calculated as -(demand - received), or 0 if received >= demand.
        """
        return -max(0.0, demand - received)

    @staticmethod
    def deficit_squared_ratio_minimised(demand: float, received: float) -> float:
        """
        Returns the negative squared ratio of the deficit to the demand, penalizing larger deficits more heavily.

        Parameters
        ----------
        demand : float
            The required demand amount.
        received : float
            The amount actually received.

        Returns
        -------
        float
            The negative squared ratio of the deficit, calculated as -((demand - received) / demand)^2,
            or 0 if received >= demand.
        """
        return -((max(0.0, demand - received) / demand) ** 2)

    @staticmethod
    def supply_ratio_maximised(demand: float, received: float) -> float:
        """
        Returns the ratio of received to demand, capped at 1.0 to maximize supply relative to demand.

        Parameters
        ----------
        demand : float
            The required demand amount.
        received : float
            The amount actually received.

        Returns
        -------
        float
            The supply-to-demand ratio, capped at 1.0 if received exceeds demand.
        """
        return received / demand if received / demand < 1.0 else 1.0

    @staticmethod
    def scalar_identity(scalar: float) -> float:
        """
        Scales an input value by a specified scalar.

        Parameters
        ----------
        scalar : float
            The scaling factor to apply to the input value.

        Returns
        -------
        float
            The input value multiplied by the scalar.
        """
        return lambda value: value * scalar

    @staticmethod
    def sequential_scalar(scalar: list[float]) -> float:
        """
        Scales an input value by a scalar from a specified list based on an index.

        Parameters
        ----------
        scalar : list[float]
            A list of scalars to be applied sequentially.

        Returns
        -------
        float
            The input value multiplied by the scalar at the specified index.
        """
        return lambda index, value: value * scalar[index]
