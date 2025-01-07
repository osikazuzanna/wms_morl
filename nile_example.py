import numpy as np
import mo_gymnasium
import examples.nile_river_simulation



water_management_system = mo_gymnasium.make('nile-v0')


def run_nile():

    #reset
    obs, info = water_management_system.reset()
    print(f'Initial Obs: {obs}')

    final_truncated = False
    final_terminated = False
    for t in range(248):
        if not final_terminated and not final_truncated:
            action = water_management_system.action_space.sample()
            print(f'Action for month: {t}: {action}')

            (
                        final_observation,
                        final_reward,
                        final_terminated,
                        final_truncated,
                        final_info
                    ) = water_management_system.step(action)
            print(f'Observation: {final_observation}')
            print(f'Reward: {final_reward}')
            
        else:
            break

    return final_observation


run_nile()