import numpy as np
import mo_gymnasium
import examples.susquehanna_river_simulation



water_management_system = mo_gymnasium.make('susquehanna-v0')

def run_susquehanna():

    #reset
    obs, info = water_management_system.reset()


    all_rewards = [0.0,0.0,0.0,0.0,0.0,0.0]

    final_observation_list = list(obs)
    final_truncated = False
    final_terminated = False
    for t in range(2191):
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
            all_rewards=all_rewards+final_reward
            
            
            
        else:
            print(f'Final reward: {all_rewards}')
            break

    return final_observation

#actions take values from 0-1
run_susquehanna()