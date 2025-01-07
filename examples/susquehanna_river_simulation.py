import numpy as np
from pathlib import Path
from gymnasium.spaces import Box
from gymnasium.wrappers.time_limit import TimeLimit
from datetime import datetime
from dateutil.relativedelta import relativedelta
from core.envs.water_management_system import WaterManagementSystem
from core.models.reservoir import Reservoir
from core.models.reservoir_with_pump import ReservoirWithPump
from core.models.flow import Flow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_district import IrrigationDistrict
from core.wrappers.transform_action import ReshapeArrayAction
from core.models.facility import Facility, ControlledFacility
import time
from gymnasium.envs.registration import register
from typing import Any, Union, Optional
from gymnasium.core import ObsType, RenderFrame


data_directory = Path(__file__).parents[1] / "examples" / "data" / "susquehanna_river"





register(
    id='susquehanna-v0',
    entry_point='examples.susquehanna_river_simulation:create_susquehanna_river_env',
)

register(
    id='susquehanna2-v0',
    entry_point='examples.susquehanna_river_simulation:create_susquehanna_river_env',
    kwargs={'custom_obj': ['water_supply_Baltimore', 'water_supply_Chester']}
)

register(
    id='susquehanna3-v0',
    entry_point='examples.susquehanna_river_simulation:create_susquehanna_river_env',
    kwargs={'custom_obj': ['water_supply_Baltimore', 'water_supply_Chester', 'water_supply_Atomic']}
)

register(
    id='susquehanna4-v0',
    entry_point='examples.susquehanna_river_simulation:create_susquehanna_river_env',
    kwargs={'custom_obj': ['water_supply_Baltimore', 'water_supply_Chester', 'water_supply_Atomic', 'recreation']}
)


# Custom rules which define how the pump works, so how much water si pumped and released from the pump to the reservoir.
#Class reservoiwWithPump takes this function as a requred argument 
def muddyrun_pumpturb_(
                        day_of_the_week, hour, level_reservoir = 0.0, 
                        level_pump = 0.0, 
                        storage_reservoir = 0.0, 
                        storage_pump = 0.0):
    # Determines the pumping and turbine release volumes in a day based
    # on the hour and day of week for muddy run
    QP = 24800  # cfs
    QT = 32000  # cfs

    # active storage = sMR - deadStorage
    qM = (
        storage_pump - 994779720.0
    ) / 3600
    qp = 0.0
    qt = 0.0
    if day_of_the_week == 6:  # sunday
        if hour < 5 or hour >= 22:
            qp = QP
    elif 0 <= day_of_the_week <= 3:  # monday to thursday
        if hour <= 6 or hour >= 21:
            qp = QP
        if (7 <= hour <= 11) or (17 <= hour <= 20):
            qt = min(QT, qM)
    elif day_of_the_week == 4:  # friday
        if (7 <= hour <= 11) or (17 <= hour <= 20):
            qt = min(QT, qM)
    elif day_of_the_week == 5:  # saturday
        if hour <= 6 or hour >= 22:
            qp = QP
    # water pumping stops to Muddy Run beyond this point.
    # However, according to the conowingo authorities 800 cfs will be
    # released as emergency credits in order to keep the facilities from
    # running
    # Q: The level in Conowingo impacts the pumping in Muddy Run. How?
    if level_reservoir < 104.7:  # if True cavitation problems in pumping
        qp = 0.0

    if level_pump < 470.0:
        qt = 0.0
    return qp, qt  # pumping, Turbine release


class WaterManagementSystemWithWaterLevels(WaterManagementSystem):
    def __init__(self, *args, **kwargs):
            # Call the parent class's __init__ method to inherit its initialization
            super().__init__(*args, **kwargs)

            # Initialize the max_capacity_level attribute
            self.max_capacity_level = self.water_systems[2].storage_to_level(self.water_systems[2].max_capacity)

    #we have to overwrite reset method as susquehanna simulation takes water levels and not storage volume as observation space
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[ObsType, dict[str, Any]]:
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)
        self.current_date = self.start_date
        self.timestep = 0

        self.observation, _ = self._determine_observation()
        #We add this recalculation from storage to levels
        observation = [self.water_systems[2].storage_to_level(self.observation[0])/self.max_capacity_level]
        observation.append(0.0)
        # Reset rewards
        for key in self.rewards.keys():
            self.rewards[key] = 0

        for water_system in self.water_systems:
            water_system.current_date = self.start_date
            water_system.reset()
        return np.array(observation).flatten(), self._determine_info()

    #We overwrite the step method
    def step(self, action: np.array) -> tuple[np.array, np.array, bool, bool, dict]:
        """
        Overwrites the step method to manage water system dynamics.
        TO BE CORRECTED
        Parameters
        ----------
        action : np.array
            The action taken for the controlled facilities.

        Returns
        -------
        tuple[np.array, np.array, bool, bool, dict]
            - An array of observations.
            - An array of rewards.
            - A boolean indicating whether the episode is terminated.
            - A boolean indicating whether the episode is truncated.
            - A dictionary containing additional info.

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

            # Add reward to the objective assigned to this Facility (unless it is a Flow).
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

        #We change the final observation as we model the problem with water levels and not the storage volume as is by default
        final_observations = [final_info['Conowingo']['current_level']/self.max_capacity_level]
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


def create_susquehanna_river_env(custom_obj = None, render_mode=None) -> WaterManagementSystemWithWaterLevels:


    class ReservoirwithPumpDateDependendObjetive(ReservoirWithPump):

        

        def determine_reward(self) -> float:
            is_weekend = self.current_date.weekday() < 5
            is_summer = 5 < self.current_date.month < 9
            summer_weekend = is_weekend and is_summer
            return self.objective_function(summer_weekend, self.storage_to_level(self.stored_water))
        


        def muddyrun_pumpturb(self,
                              day_of_the_week, hour, level_reservoir = 0.0, 
                              level_pump = 0.0, 
                              storage_reservoir = 0.0, 
                              storage_pump = 0.0):
            # Determines the pumping and turbine release volumes in a day based
            # on the hour and day of week for muddy run
            QP = 24800  # cfs
            QT = 32000  # cfs

            # active storage = sMR - deadStorage
            qM = (
                storage_pump - 994779720.0
            ) / 3600
            qp = 0.0
            qt = 0.0
            if day_of_the_week == 6:  # sunday
                if hour < 5 or hour >= 22:
                    qp = QP
            elif 0 <= day_of_the_week <= 3:  # monday to thursday
                if hour <= 6 or hour >= 21:
                    qp = QP
                if (7 <= hour <= 11) or (17 <= hour <= 20):
                    qt = min(QT, qM)
            elif day_of_the_week == 4:  # friday
                if (7 <= hour <= 11) or (17 <= hour <= 20):
                    qt = min(QT, qM)
            elif day_of_the_week == 5:  # saturday
                if hour <= 6 or hour >= 22:
                    qp = QP
            # water pumping stops to Muddy Run beyond this point.
            # However, according to the conowingo authorities 800 cfs will be
            # released as emergency credits in order to keep the facilities from
            # running
            if level_reservoir < 104.7:  # if True cavitation problems in pumping
                qp = 0.0

            if level_pump < 470.0:
                qt = 0.0
            return qp, qt  # pumping, Turbine release


        #predifined release constraints, depending on the days of the year, different for each release as it affects differect capacities (baltimore, app, chester, downstream)
        def actual_release(self, actions, level_Co, day_of_year):
            # Check if it doesn't exceed the spillway capacity
            Tcap = 85412  # total turbine capacity (cfs)
            # maxSpill = 1242857.0 # total spillway combined (cfs)
            w_atomic = np.loadtxt(data_directory / "reservoirs" / "wAtomic.txt")
            w_baltimore = np.loadtxt(data_directory / "reservoirs" / "wBaltimore.txt")
            w_chester = np.loadtxt(data_directory / "reservoirs" / "wChester.txt")
            spillways = np.loadtxt(data_directory / "reservoirs" / "spillways_Conowingo.txt")
            # minimum discharge values at APP, Balitomore, Chester and downstream
            qm_A = 0.0
            qm_B = 0.0
            qm_C = 0.0
            qm_D = 0.0

            # maximum discharge values. The max discharge can be as much as the
            # demand in that area
            qM_A = w_atomic[day_of_year]
            qM_B = w_baltimore[day_of_year]
            qM_C = w_chester[day_of_year]
            qM_D = Tcap

            # reservoir release constraints (minimum level app)
            if level_Co <= 98.5:
                qM_A = 0.0
            else:
                qM_A = w_atomic[day_of_year]

            if level_Co <= 90.8: #minumum level baltimore
                qM_B = 0.0
            else:
                qM_B = w_baltimore[day_of_year]
            if level_Co <= 99.8: #minimum level chester (ft of water)
                qM_C = 0.0
            else:
                qM_C = w_chester[day_of_year]

            if level_Co > 110.2:  # spillways activated
                qM_D = (
                    self.modified_interp(level_Co, spillways[0], spillways[1])
                    + Tcap
                )  # Turbine capacity + spillways
                qm_D = (
                    self.modified_interp(level_Co, spillways[0], spillways[1])
                    + Tcap
                )  

            # different from flood model
            if level_Co < 105.5:
                qM_D = 0.0
            elif level_Co < 103.5:
                qM_A = 0.0
            elif level_Co < 100.5:
                qM_C = 0.0
            elif level_Co < 91.5:
                qM_B = 0.0

            # actual release
            rr = []
            rr.append(min(qM_A, max(qm_A, actions[0])))
            rr.append(min(qM_B, max(qm_B, actions[1])))
            rr.append(min(qM_C, max(qm_C, actions[2])))
            rr.append(min(qM_D, max(qm_D, actions[3])))
            return np.array(rr)
        

        def determine_outflow(self, actions: np.array) -> list[float]:
            

            current_storage = self.storage_vector[-1]
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
            evaporatio_rate_per_second = self.evap_rates[self.determine_time_idx()] / (timestep_seconds)
            evaporatio_rate_per_second_pump = self.evap_rates_pump[self.determine_time_idx()] / (timestep_seconds)

            while self.current_date < final_date:
                next_date = min(final_date, self.current_date + self.integration_timestep_size)
                integration_time_seconds = (next_date - self.current_date).total_seconds()
                
                #pumping/release of the pump

                pumping, release_pump = self.pumping_rules(day_of_the_week = self.current_date.weekday(), 
                                                    hour = self.current_date.hour, 
                                                    level_reservoir = self.storage_to_level(current_storage), 
                                                    level_pump = self.storage_to_level_pump(current_storage_pump),
                                                    storage_reservoir = current_storage, 
                                                    storage_pump = current_storage_pump)


                surface = self.storage_to_surface(current_storage)
                surface_pump = self.storage_to_surface_pump(current_storage_pump)

                evaporation = surface * evaporatio_rate_per_second
                evaporation_pump = surface_pump * evaporatio_rate_per_second_pump 

                current_storage_pump = current_storage_pump + (self.inflows_pump[self.timestep] + pumping - release_pump - evaporation_pump) * integration_time_seconds 
                day_of_year = self.current_date.timetuple().tm_yday - 1 #to read from a np array indexed at 1
                release_per_second = self.actual_release(actions, self.storage_to_level(current_storage), day_of_year)

                if self.should_split_release == True:
                    sub_releases.append(release_per_second)
                else:
                    sub_releases = np.append(sub_releases, release_per_second)

                total_addition = (self.get_inflow(self.timestep) + release_pump) 

                current_storage = current_storage + integration_time_seconds*(total_addition
                                                                              -np.sum(release_per_second)
                                                                              -evaporation
                                                                              -pumping
                                                                              -self.spillage)
                
                self.current_date = next_date
                

            
            # Update the amount of water in the Reservoir
            self.storage_vector.append(current_storage)
            self.stored_water = current_storage

            # Update water in the pump
            self.storage_pump_vector.append(current_storage_pump)
            self.stored_pump = current_storage_pump


            # Record level based on storage for time t
            self.level_vector.append(self.storage_to_level(current_storage))

            # Calculate the ouflow of water depending if the releases are split (an array with lists of releases to different destinations from one reservoir or an array with scalar values)
            if self.should_split_release == True:
                sub_releases = np.array(sub_releases)
                average_release = np.mean(sub_releases, dtype=np.float64, axis = 0)
            else:
                average_release = np.mean(sub_releases, dtype=np.float64)

            self.release_vector.append(average_release)

            total_action = np.sum(average_release)
            # Split release for different destinations - calculate ratio of the releases to different destinations from the total action
            if self.should_split_release and total_action != 0:
                self.split_release = [(action / total_action) for action in average_release]
                average_release = total_action
            elif self.should_split_release and total_action == 0:
                self.split_release = [(action) for action in average_release]
                average_release = total_action



            return average_release



    class PowerPlantSequentialObjetive(PowerPlant):
        def determine_reward(self) -> float:
            return self.objective_function(self.timestep, self.determine_production_detailed())

    Conowingo_reservoir = ReservoirwithPumpDateDependendObjetive(
        name="Conowingo",
        integration_timestep_size=relativedelta(hours=1),
        objective_function=Objective.is_greater_than_minimum_with_condition(106.5),
        objective_name="recreation",
        stored_water_reservoir=2641905256.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_Conowingo.txt"),
        evap_rates_timestep_size=relativedelta(days=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_Conowingo.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_Conowingo.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_sur_rel_Conowingo.txt"),
        pumping_rules=muddyrun_pumpturb_,
        stored_water_pump = 1931101920.0,
        evap_rates_pump = np.loadtxt(data_directory / "reservoirs" / "evap_Muddy.txt"),
        storage_to_surface_rel_pump = np.loadtxt(data_directory / "reservoirs" / "storage_surface_rel_MR.txt"),
        storage_to_level_rel_pump = np.loadtxt(data_directory / "reservoirs" / "storage_level_rel_MR.txt"),
        inflows_pump = np.loadtxt(data_directory / "inflows" / "InflowMuddy.txt"),
        spillage = 800,
        max_capacity=6962264495.999999,
        max_action=[41.302169, 464.16667, 54.748458, 85412]
                )

    Power_plant = PowerPlantSequentialObjetive(
        name="Power_plant",
        objective_function=Objective.sequential_scalar(np.loadtxt(data_directory / "reservoirs" / "avg_energy_prices.txt")),
        objective_name="energy_revenue",
        normalize_objective=198445191.2586579,
        efficiency=0.79,
        min_turbine_flow=210.0, #this assumes there is only one turbine!
        max_turbine_flow=85412.0, #this assumes there is only one turbine!
        head_start_level=0,
        max_capacity=float("inf"),
        reservoir=Conowingo_reservoir,
        turbines = np.loadtxt(data_directory / "reservoirs" / "turbines_Conowingo2.txt"),
        n_turbines = 13,
        tailwater =  np.loadtxt(data_directory / "reservoirs" / "tailwater.txt")
    )

    Atomic_system = IrrigationDistrict(
        name="Atomic",
        all_demand=np.loadtxt(data_directory / "demands" / "Atomic.txt"),
        objective_function=Objective.supply_ratio_maximised,
        objective_name="water_supply_Atomic",
    )

    Baltimore_system = IrrigationDistrict(
        name="Baltimore",
        all_demand=np.loadtxt(data_directory / "demands" / "Baltimore.txt"),
        objective_function=Objective.supply_ratio_maximised,
        objective_name="water_supply_Baltimore",
    )

    Chester_system = IrrigationDistrict(
        name="Chester",
        all_demand=np.loadtxt(data_directory / "demands" / "Chester.txt"),
        objective_function=Objective.supply_ratio_maximised,
        objective_name="water_supply_Chester",
    )

    Downstream_system = IrrigationDistrict(
        name="Downstream",
        all_demand=np.loadtxt(data_directory / "demands" / "Downstream.txt"),
        objective_function=Objective.deficit_squared_ratio_minimised,
        objective_name="enviromental_shortage",
    )

    Conowingo_inflow_main = Inflow(
        name="conowingo_inflow_main",
        destinations=Conowingo_reservoir,
        max_capacity=float("inf"),
        all_inflow=np.loadtxt(data_directory / "inflows" / "InflowConowingoMain.txt"),
    )

    Conowingo_inflow_lateral = Inflow(
        name="conowingo_inflow_lateral",
        destinations=Conowingo_reservoir,
        max_capacity=float("inf"),
        all_inflow=np.loadtxt(data_directory / "inflows" / "InflowConowingoLateral.txt"),
    )

    Conowingo_outflow = Flow(
        name="conowingo_outflow",
        sources=[Conowingo_reservoir],
        destinations={Atomic_system: 0.25, Baltimore_system: 0.25, Chester_system: 0.25, Power_plant: 0.25}, 
        max_capacity=float("inf"),
    )

    Donwstream_inflow = Flow(
        name="donwstream_inflow",
        sources=[Power_plant],
        destinations=Downstream_system,
        max_capacity=float("inf"),
    )

    # Muddy_reservoir = Reservoir(
    #     name="Muddy",
    #     observation_space=Box(low=0, high=2471202360),
    #     action_space=Box(low=0, high=32000),
    #     integration_timestep_size=relativedelta(hours=1),
    #     objective_function=Objective.no_objective,
    #     objective_name="",
    #     stored_water=0,
    #     evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_Muddy.txt"),
    #     storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_Muddy.txt"),
    #     storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_Muddy.txt"),
    #     storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_sur_rel_Muddy.txt"),
    # )

    water_management_system = WaterManagementSystemWithWaterLevels(
        water_systems=[
            Conowingo_inflow_main,
            Conowingo_inflow_lateral,
            Conowingo_reservoir,
            Conowingo_outflow,
            Atomic_system,
            Baltimore_system,
            Chester_system,
            Power_plant,
            Donwstream_inflow,
            Downstream_system,
        ],
        rewards={
            "recreation": 0,
            "energy_revenue": 0,
            "water_supply_Baltimore": 0,
            "water_supply_Atomic": 0,
            "water_supply_Chester": 0,
            "enviromental_shortage": 0,
        },
        start_date=datetime(2021, 1, 1),
        timestep_size=relativedelta(hours=4),
        seed=42,
        add_timestamp='h',
        custom_obj = custom_obj
    )

    water_management_system = ReshapeArrayAction(water_management_system)
    water_management_system = TimeLimit(water_management_system, max_episode_steps=2190)

    return water_management_system
