import numpy as np
from examples.susquehanna_river_simulation import create_susquehanna_river_env
import mo_gymnasium

import examples.susquehanna_river_simulation



water_management_system = mo_gymnasium.make('susquehanna-v0')



class TrainSusquehanna():
    '''Class created to train susquehanna with emodps and rbf policy'''

    def __init__(self, rbf, timesteps):
        
        self.timesteps = timesteps
        
         # variables from the header file
        self.input_min = []
        self.input_max = [self.timesteps, 120.0]
        self.output_max = [41.302169, 464.16667, 54.748458, 85412]
        self.rbf = rbf

        


    def apply_rbf_policy(self, rbf_input):

        # apply rbf
        normalized_output = self.rbf.apply_rbfs(rbf_input)

        return normalized_output 



    def run_episode(self, rbf_params):


        #reset
        final_observation, info = water_management_system.reset()

        #set the rbf parameters
        self.rbf.set_decision_vars(np.asarray(rbf_params))

        all_rewards = []

        #initial level
        final_terminated = False
        final_truncated = False
        for t in range(self.timesteps):
            if not final_terminated and not final_truncated:

                rbf_input = np.asarray(final_observation)

                action = self.apply_rbf_policy(rbf_input)

                (
                            final_observation,
                            final_reward,
                            final_terminated,
                            final_truncated,
                            final_info
                        ) = water_management_system.step(action)
                
                all_rewards.append(final_reward)
            else:
                break


        all_rewards = np.array(all_rewards)
        recreation = np.sum(all_rewards[:, 0])
        energy_revenue = np.sum(all_rewards[:,1])
        baltimore = np.sum(all_rewards[:,2])
        atomic = np.sum(all_rewards[:,3])
        chester = np.sum(all_rewards[:,4])
        environment = np.sum(all_rewards[:,5])

        return recreation, energy_revenue, baltimore, atomic, chester, environment


