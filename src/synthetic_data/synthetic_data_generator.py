import os
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing
import json


class DiseaseProgressionSimulator:
    """
    Simulator for disease progression using a Markov model.

    Attributes:
        n_states (int): Number of states in the Markov model.
        sigma (float): Standard deviation for the noise in observation.
        qi (np.array): Transition rates between states.
        qij (np.array): Transition probability matrix.
        max_chain_duration (float): Maximum duration of the Markov chain.
    """
    def __init__(self, states=5, sigma=0.5):
        
        self.n_states = states
        self.generate_transition_rates()
        self.max_chain_duration = 100 / np.min(
            self.qi
        )  # 100 times the largest mean holding time
        self.sigma = sigma

    def generate_transition_rates(self):
        # Transition rates qi randomly drawn from [1, 5]
        self.qi = np.random.uniform(1, self.n_states, size=self.n_states)

        # Transition probabilities qij drawn from [0, 1] and normalized
        self.qij = np.random.uniform(0, 1, size=(self.n_states, self.n_states))

        for i in range(self.n_states):
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
        current_state = np.random.choice(self.n_states)
        current_time = 0

        while current_time < self.max_chain_duration:
            state_chain.append(current_state)
            time_array.append(current_time)

            # Update the current time based on a random draw
            time_spent = np.random.exponential(1 / self.qi[current_state])
            current_time += time_spent

            # Select next state based on transition probabilities, not including self-transition
            probabilities = self.qij[current_state, :].copy()
            probabilities[current_state] = 0
            probabilities /= self.qi[current_state]
            next_state = np.random.choice(self.n_states, p=probabilities)

            current_state = next_state

        return state_chain, time_array

    def generate_observations(self, total_observations, n_vars=1):
        current_total_obs = 0
        chains = []

        state_to_center = [
            np.random.uniform(-1, 1, size=n_vars) for _ in range(self.n_states)
        ]

        while current_total_obs < total_observations:
            chain_states, chain_times = self.generate_chain()
            chain_total_duration = chain_times[-1]

            current_observation_time = 0
            chain_obs_times = []
            chain_obs_emissions = []
            chain_obs_states = []

            while True:
                # Sample observation time from exponential distribution
                obs_time = np.random.exponential(0.5 / np.max(self.qi))
                current_observation_time += obs_time

                if current_observation_time > chain_total_duration:
                    break

                chain_obs_times.append(obs_time)

                # Find the index of the state at the observation time
                state_idx = np.searchsorted(chain_times, current_observation_time)

                state = chain_states[state_idx]

                # scale state to [-1, 1]

                emission_center = state_to_center[state]

                chain_obs_states.append(emission_center[0])

                cov_matrix = (
                    np.eye(len(emission_center)) * self.sigma**2
                )  # Create a covariance matrix

                # Append the state and observation time
                chain_obs_emissions.append(
                    np.random.multivariate_normal(emission_center, cov_matrix)
                )

                current_total_obs += 1

            chain_df = pd.DataFrame(chain_obs_emissions, columns=[f"var_{i}" for i in range(n_vars)])
            chain_df["time"] = chain_obs_times
            chain_df["state"] = chain_obs_states

            chains.append(chain_df)

        return chains

    def generate_and_save(
        self, total_observations, save_dir, transition_matrix=None, idx=0, n_vars=128
    ):
        # self.generate_transition_rates()

        if transition_matrix is not None:
            self.n_states = len(transition_matrix)
            self.qij = transition_matrix
            self.qi = -np.diag(self.qij)
        else:
            self.generate_transition_rates()

        save_dir = os.path.join(
            save_dir, f'{idx}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )

        # save_dir = os.path.join(save_dir, f'sigma_{self.sigma}'.replace('.', '_'))

        # Create directory for chains if it doesn't exist
        chains_dir = os.path.join(save_dir, "chains")
        os.makedirs(chains_dir, exist_ok=True)

        # Save transition matrix
        stats = {
            "sigma": self.sigma,
            "states": self.n_states,
            "transition_matrix": self.qij.tolist(),
        }

        json_path = os.path.join(save_dir, "stats.json")
        json.dump(stats, open(json_path, "w"))

        for i, chain in enumerate(
            self.generate_observations(total_observations, n_vars)
        ):
            chain.to_csv(os.path.join(chains_dir, f"chain_{i}.csv"), index=False)


def generate_and_save_parallel(total_observations, sigma, transition_matrix, idx=0):
    save_dir = "data/synthetic_data/"
    simulator = DiseaseProgressionSimulator(sigma=sigma)
    simulator.generate_and_save(
        total_observations, save_dir, transition_matrix=transition_matrix, idx=idx
    )


if __name__ == "__main__":
    processes = []

    TOTAL_OBS = 1e5
    OBS_PER_PARAMSET = 5000
    n_vars = 1

    for idx, i in enumerate(range(int(TOTAL_OBS // OBS_PER_PARAMSET))):
        n_states = np.random.choice([5, 5, 10, 20, 50])
        sigma = np.random.choice([0.1, 0.15, 0.25, 0.5, 0.75, 1])
        simulator = DiseaseProgressionSimulator(sigma=sigma, states=n_states)
        transition_matrix = simulator.qij
        # process = multiprocessing.Process(target=generate_and_save_parallel, args=(OBS_PER_PARAMSET, sigma,transition_matrix, idx))
        # process.start()
        # processes.append(process)
        simulator.generate_and_save(
            OBS_PER_PARAMSET,
            "data/synthetic_data/one_dim/",
            transition_matrix=transition_matrix,
            idx=idx,
            n_vars=n_vars,
        )

    # for process in processes:
    #     process.join()
