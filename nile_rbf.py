import numpy as np
from examples.nile_river_simulation import create_nile_river_env
from rbf import rbf_functions
import mo_gymnasium

import examples.nile_river_simulation



water_management_system = mo_gymnasium.make('nile-v0')




class TrainNile():
    '''Class created to train nile with emodps and rbf policy'''

    def __init__(self, rbf, timesteps):
        
        self.timesteps = timesteps
        
         # variables from the header file
        self.input_min = [0,0,0, 0,0]
        self.input_max = [117500000000.00,6095000000.0,579900000.0,182700000000.0,11]
        
        self.output_max = [10000.0,15000.0,7000.0,7000.0]
            
        self.rbf = rbf
        self.time = 0

        


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

        final_truncated = False
        final_terminated = False
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
        ethiopia_power = np.sum(all_rewards[:, 0]) #total hydroproduction
        sudan_deficit = np.sum(all_rewards[:,1])
        egypt_deficit = np.sum(all_rewards[:,2])
        had_min_level = np.sum(all_rewards[:,3])



        return ethiopia_power, sudan_deficit, egypt_deficit, had_min_level

