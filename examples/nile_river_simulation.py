import numpy as np
from pathlib import Path
from gymnasium.spaces import Box
from gymnasium.wrappers.time_limit import TimeLimit
from core.envs.water_management_system import WaterManagementSystem
from core.models.reservoir import Reservoir
from core.models.flow import Flow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_district import IrrigationDistrict
from core.models.catchment import Catchment
from core.wrappers.transform_action import ReshapeArrayAction
from datetime import datetime
from dateutil.relativedelta import relativedelta
from gymnasium.envs.registration import register

data_directory = Path(__file__).parents[1] / "examples" / "data" / "nile_river"

register(
    id='nile-v0',
    entry_point='examples.nile_river_simulation:create_nile_river_env',
)

register(
    id='nile2-v0',
    entry_point='examples.nile_river_simulation:create_nile_river_env',
    kwargs={'custom_obj': ['ethiopia_power', 'egypt_deficit_minimised']}
)

register(
    id='nile3-v0',
    entry_point='examples.nile_river_simulation:create_nile_river_env',
    kwargs={'custom_obj': ['ethiopia_power', 'egypt_deficit_minimised', 'sudan_deficit_minimised']}
)


def create_nile_river_env(custom_obj = None, render_mode=None) -> WaterManagementSystem:
    # Ethiopia
    GERD_reservoir = Reservoir(
        "GERD",
        max_capacity=117500000000.0,
        max_action=[10000],
        integration_timestep_size=relativedelta(minutes=240),
        objective_function=Objective.no_objective,
        stored_water=15000000000.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_GERD.txt"),
        evap_rates_timestep_size=relativedelta(months=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_GERD.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_GERD.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_sur_rel_GERD.txt"),
    )
    GERD_power_plant = PowerPlant(
        "GERD_power_plant",
        Objective.scalar_identity(1),
        "ethiopia_power",
        efficiency=0.93,
        min_turbine_flow=0,
        max_turbine_flow=4320,
        head_start_level=507,
        max_capacity=6000,
        reservoir=GERD_reservoir,
        normalize_objective=4464000.0 #the max amount of electricity produced possible in one month based on the max capacity
    )
    # Sudan
    DSSennar_irr_system = IrrigationDistrict(
        "DSSennar_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_DSSennar.txt"),
        Objective.deficit_minimised,
        "sudan_deficit_minimised",
        normalize_objective=12.73148148
    )
    Gezira_irr_system = IrrigationDistrict(
        "Gezira_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Gezira.txt"),
        Objective.deficit_minimised,
        "sudan_deficit_minimised",
        normalize_objective=388.2915173
    )
    Hassanab_irr_system = IrrigationDistrict(
        "Hassanab_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Hassanab.txt"),
        Objective.deficit_minimised,
        "sudan_deficit_minimised",
        normalize_objective=38.96604938
    )
    Tamaniat_irr_system = IrrigationDistrict(
        "Tamaniat_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Tamaniat.txt"),
        Objective.deficit_minimised,
        "sudan_deficit_minimised",
        normalize_objective=25.84876543
    )
    USSennar_irr_system = IrrigationDistrict(
        "USSennar_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_USSennar.txt"),
        Objective.deficit_minimised,
        "sudan_deficit_minimised",
        normalize_objective=148.2228196
    )
    Roseires_reservoir = Reservoir(
        "Roseires",
        max_capacity=6095000000.0,
        max_action=[15000.0],
        integration_timestep_size=relativedelta(minutes=240),
        objective_function=Objective.no_objective,
        stored_water=4571250000.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_Roseires.txt"),
        evap_rates_timestep_size=relativedelta(months=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_Roseires.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_Roseires.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_sur_rel_Roseires.txt"),
    )
    Sennar_reservoir = Reservoir(
        "Sennar",
        max_capacity=579900000.0,
        max_action = [7000.0],
        integration_timestep_size=relativedelta(minutes=240),
        objective_function=Objective.no_objective,
        stored_water=434925000.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_Sennar.txt"),
        evap_rates_timestep_size=relativedelta(months=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_Sennar.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_Sennar.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_sur_rel_Sennar.txt"),
    )
    # Egypt
    Egypt_irr_system = IrrigationDistrict(
        "Egypt_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Egypt.txt"),
        Objective.deficit_minimised,
        "egypt_deficit_minimised",
        normalize_objective=2523.894863
    )
    HAD_reservoir = Reservoir(
        "HAD",
        max_capacity=182700000000.0,
        max_action=[7000.0],
        integration_timestep_size=relativedelta(minutes=240),
        objective_function=Objective.is_greater_than_minimum(159),
        objective_name="HAD_minimum_water_level",
        stored_water=137025000000.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_HAD.txt"),
        evap_rates_timestep_size=relativedelta(months=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_HAD.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_HAD.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_sur_rel_HAD.txt"),
    )
    # Create 'edges' between Facilities.
    # TODO: determine max capacity for flows
    GERD_inflow = Inflow(
        "gerd_inflow",
        GERD_reservoir,
        float("inf"),
        np.loadtxt(data_directory / "catchments" / "InflowBlueNile.txt"),
    )

    GerdToRoseires_catchment = Catchment(
        "GerdToRoseires_catchment", np.loadtxt(data_directory / "catchments" / "InflowGERDToRoseires.txt")
    )
    # TODO: add catchment 1 inflow to sources of Roseires (inflow with destination Roseires)

    Power_plant_flow = Flow("power_plant_flow", [GERD_reservoir], GERD_power_plant, float("inf"))

    Roseires_flow = Flow(
        "roseires_flow", [GERD_power_plant, GerdToRoseires_catchment], Roseires_reservoir, float("inf")
    )

    RoseiresToAbuNaama_catchment = Catchment(
        "RoseiresToAbuNaama_catchment", np.loadtxt(data_directory / "catchments" / "InflowRoseiresToAbuNaama.txt")
    )

    # TODO: add catchment 2 inflow to sources of USSennar (inflow with destination USSennar)
    upstream_Sennar_received_flow = Flow(
        "upstream_Sennar_received_flow",
        [Roseires_reservoir, RoseiresToAbuNaama_catchment],
        USSennar_irr_system,
        float("inf"),
    )

    SukiToSennar_catchment = Catchment(
        "SukiToSennar_catchment", np.loadtxt(data_directory / "catchments" / "InflowSukiToSennar.txt")
    )

    # TODO: add catchment 3 inflow to sources of Sennar (inflow with destination USSennar)
    Sennar_flow = Flow("sennar_flow", [USSennar_irr_system, SukiToSennar_catchment], Sennar_reservoir, float("inf"))

    Gezira_received_flow = Flow("gezira_received_flow", [Sennar_reservoir], Gezira_irr_system, float("inf"))

    Dinder_catchment = Catchment("dinder_catchment", np.loadtxt(data_directory / "catchments" / "InflowDinder.txt"))

    Rahad_catchment = Catchment("rahad_catchment", np.loadtxt(data_directory / "catchments" / "InflowRahad.txt"))

    downstream_Sennar_received_flow = Flow(
        "downstream_sennar_received_flow",
        [Gezira_irr_system, Dinder_catchment, Rahad_catchment],
        DSSennar_irr_system,
        float("inf"),
    )
    WhiteNile_catchment = Catchment(
        "whitenile_catchment",
        np.loadtxt(data_directory / "catchments" / "InflowWhiteNile.txt"),
    )
    Taminiat_received_flow = Flow(
        "taminiat_received_flow",
        [DSSennar_irr_system, WhiteNile_catchment],
        Tamaniat_irr_system,
        float("inf"),
    )

    Atbara_catchment = Catchment("atbara_catchment", np.loadtxt(data_directory / "catchments" / "InflowAtbara.txt"))

    # TODO: change Hassanab received flow to depend on leftover flow from Taminiat in previous month (see A.2.8)
    Hassanab_received_flow = Flow(
        "hassanab_received_flow",
        [Tamaniat_irr_system, Atbara_catchment],
        Hassanab_irr_system,
        float("inf"),
        delay=1,
        default_outflow=934.2,
    )
    HAD_flow = Flow("had_flow", [Hassanab_irr_system], HAD_reservoir, float("inf"))
    Egypt_flow = Flow("egypt_flow", [HAD_reservoir], Egypt_irr_system, float("inf"))
    # Create water management system. Add Facilities in the topological order (in the list).
    # Egypt deficit reward goes negative when there is a deficit. Otherwise is 0.
    water_management_system = WaterManagementSystem(
        water_systems=[
            GERD_inflow,
            GERD_reservoir,
            Power_plant_flow,
            GERD_power_plant,
            GerdToRoseires_catchment,
            Roseires_flow,
            Roseires_reservoir,
            RoseiresToAbuNaama_catchment,
            upstream_Sennar_received_flow,
            USSennar_irr_system,
            SukiToSennar_catchment,
            Sennar_flow,
            Sennar_reservoir,
            Gezira_received_flow,
            Gezira_irr_system,
            Dinder_catchment,
            Rahad_catchment,
            downstream_Sennar_received_flow,
            DSSennar_irr_system,
            WhiteNile_catchment,
            Taminiat_received_flow,
            Tamaniat_irr_system,
            Atbara_catchment,
            Hassanab_received_flow,
            Hassanab_irr_system,
            HAD_flow,
            HAD_reservoir,
            Egypt_flow,
            Egypt_irr_system,
        ],
        rewards={
            "ethiopia_power": 0,
            "sudan_deficit_minimised": 0,
            "egypt_deficit_minimised": 0,
            "HAD_minimum_water_level": 0,
        },
        start_date=datetime(2025, 1, 1),
        timestep_size=relativedelta(months=1),
        seed=42
        ,add_timestamp='m',
        custom_obj=custom_obj
        
    )

    water_management_system = ReshapeArrayAction(water_management_system)
    water_management_system = TimeLimit(water_management_system, max_episode_steps=12)

    return water_management_system
