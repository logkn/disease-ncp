import os
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing

class DiseaseProgressionSimulator:
    def __init__(self, states=5, sigma=0.5):
        self.states = states
        self.generate_transition_rates()
        self.max_chain_duration = 100 / np.min(self.qi) # 100 times the largest mean holding time
        self.sigma = sigma

    def generate_transition_rates(self):
        # Transition rates qi randomly drawn from [1, 5]
        self.qi = np.random.uniform(1, 5, size=self.states)

        # Transition probabilities qij drawn from [0, 1] and normalized
        self.qij = np.random.uniform(0, 1, size=(self.states, self.states))

        for i in range(self.states):
            # the sum of the row should be zero,
            # but the diagonal element qij[i,i] should be -qi[i]
            self.qij[i, i] = -self.qi[i]

            # row sum is the sum of the transition rates
            # not including the self-transition
            row_sum = np.sum(self.qij[i, :]) + self.qi[i]
            
            self.qij[i, :] *= self.qi[i] / row_sum

            self.qij[i, i] = -self.qi[i]

            assert np.isclose(np.sum(self.qij[i, :]), 0.0)
            assert np.isclose(self.qij[i, i], -self.qi[i])

    def generate_chain(self):
        # Initialize state chain and time array
        state_chain = []
        time_array = []
        current_state = np.random.choice(self.states)
        current_time = 0

        while current_time < self.max_chain_duration:
            state_chain.append(current_state)
            time_array.append(current_time)

            # Update the current time based on a random draw
            time_spent = np.random.exponential(1/self.qi[current_state])
            current_time += time_spent

            # Select next state based on transition probabilities, not including self-transition
            probabilities = self.qij[current_state, :].copy()
            probabilities[current_state] = 0
            probabilities /= self.qi[current_state]
            next_state = np.random.choice(self.states, p=probabilities)

            current_state = next_state

        return state_chain, time_array

    def generate_observations(self, total_observations):
        current_total_obs = 0
        chains = []

        while current_total_obs < total_observations:
            chain_states, chain_times = self.generate_chain()
            chain_total_duration = chain_times[-1]
            
            current_observation_time = 0
            chain_obs_times = []
            chain_obs_emissions = []
            chain_obs_states = []

            while True:
                # Sample observation time from exponential distribution
                obs_time = np.random.exponential(0.5/np.max(self.qi))
                current_observation_time += obs_time

                if current_observation_time > chain_total_duration:
                    break

                chain_obs_times.append(obs_time)

                # Find the index of the state at the observation time
                state_idx = np.searchsorted(chain_times, current_observation_time)

                state = chain_states[state_idx]

                chain_obs_states.append(state)

                # Append the state and observation time
                chain_obs_emissions.append(
                    np.random.normal(self.qi[state], self.sigma)
                )

                current_total_obs += 1

            chain_df = pd.DataFrame({
                'time': chain_obs_times,
                'emission': chain_obs_emissions,
                "state": chain_obs_states
            })

            chains.append(chain_df)

        return chains


    def generate_and_save(self, total_observations, save_dir, transition_matrix=None):
        # self.generate_transition_rates()

        if transition_matrix is not None:
            self.qij = transition_matrix
            self.qi = -np.diag(self.qij)

        # save_dir = os.path.join(save_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        save_dir = os.path.join(save_dir, f'sigma_{self.sigma}'.replace('.', '_'))

        # Create directory for chains if it doesn't exist
        chains_dir = os.path.join(save_dir, 'chains')
        os.makedirs(chains_dir, exist_ok=True)

        # Save transition matrix
        np.savetxt(os.path.join(save_dir, 'jump_matrix.csv'), self.qij, delimiter=',')

        for i, chain in enumerate(self.generate_observations(total_observations)):
            chain.to_csv(os.path.join(chains_dir, f'chain_{i}.csv'), index=False)

def generate_and_save_parallel(sigma, transition_matrix):
    save_dir = 'data/synthetic_data/'
    simulator = DiseaseProgressionSimulator(sigma=sigma)
    simulator.generate_and_save(1e5, save_dir, transition_matrix=transition_matrix)

if __name__ == '__main__':
    processes = []

    simulator = DiseaseProgressionSimulator(sigma=1)
    transition_matrix = simulator.qij

    for sigma in [1/4, 1/2, 1, 2]:
        process = multiprocessing.Process(target=generate_and_save_parallel, args=(sigma,transition_matrix))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
