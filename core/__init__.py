from gymnasium.envs.registration import register

register(
    id="WaterManagementSystem-v0",
    entry_point="core.envs:WaterManagementSystem",
)
