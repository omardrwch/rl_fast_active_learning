import logging
import numpy as np
from rlberry.agents import Agent
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.agents.dynprog.utils import backward_induction_sd, backward_induction


logger = logging.getLogger(__name__)


class AdaptiveRFAgent(Agent):
    """
    rf_type : str, 'rf_express' or 'rf_ucrl'

    """

    name = "AdaptiveRF"

    def __init__(
        self,
        env,
        horizon=50,
        delta=0.1,
        rf_type="rf_express",
        clip_v=True,
        stage_dependent=True,
        **kwargs
    ):
        Agent.__init__(self, env, **kwargs)
        self.horizon = horizon
        self.delta = delta
        self.clip_v = clip_v
        self.stage_dependent = stage_dependent

        assert rf_type in ["rf_express", "rf_ucrl"]
        self.rf_type = rf_type

        self.vmax = np.inf
        if clip_v:
            self.vmax = self.horizon

        self.reset()

    def reset(self, **kwargs):
        del kwargs
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n

        # initial state (for stopping rule)
        self.initial_state = self.env.reset()

        # ep counter
        self.episode = 0

        # useful object to compute total number of visited states & entropy of visited states
        self.counter = DiscreteCounter(
            self.env.observation_space, self.env.action_space
        )

        if self.stage_dependent:
            shape_hsa = (H, S, A)
            shape_hsas = (H, S, A, S)
        else:
            shape_hsa = (S, A)
            shape_hsas = (S, A, S)

        # W_h^t(s, a), used to upper bound the error
        self.W_hsa = np.zeros((H, S, A))
        self.V_hs = np.zeros((H, S))  # auxiliary only

        # N_h^t(s, a) and N_h^t(s,a,s'), counting the number of visits
        self.N_hsa = np.zeros(shape_hsa)
        self.N_hsas = np.zeros(shape_hsas)

        self.bonus = np.ones(shape_hsa)
        self.P_hat = np.ones(shape_hsas) * 1.0 / S

        # initialize bonus
        if self.rf_type == "rf_express":
            self.bonus *= 15 * (self.horizon**2) * self._beta(0.0)
            self.name = "RF-Express"

        elif self.rf_type == "rf_ucrl":
            self.bonus *= self.horizon * np.sqrt(self._beta(0.0))
            self.name = "RF-UCRL"

    def _beta(self, n):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n
        delta = self.delta
        beta = np.log(3 * S * A * H / delta) + S * np.log(8 * np.exp(1) * (n + 1))
        return beta

    def get_error_upper_bound(self):
        w = self.W_hsa[0, self.initial_state, :].max()

        if self.rf_type == "rf_express":
            bound = 3 * np.exp(1) * np.sqrt(w) + w
        else:
            bound = w
        return bound

    def stopping_rule(self, epsilon):
        return self.get_error_upper_bound() < epsilon / 2

    def exploration_policy(self, state, hh):
        return self.W_hsa[hh, state, :].argmax()

    def eval(self, **kwargs):
        del kwargs
        return self.counter.get_n_visited_states()

    def fit(self, budget, **kwargs):
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self.run_episode()
            count += 1

    def _update(self, state, action, next_state, hh):
        if self.stage_dependent:
            self.N_hsa[hh, state, action] += 1
            self.N_hsas[hh, state, action, next_state] += 1

            n_hsa = self.N_hsa[hh, state, action]
            n_hsas = self.N_hsas[hh, state, action, :]
            self.P_hat[hh, state, action, :] = n_hsas / n_hsa

            if self.rf_type == "rf_express":
                self.bonus[hh, state, action] = (
                    15 * (self.horizon**2) * self._beta(n_hsa) / n_hsa
                )

            elif self.rf_type == "rf_ucrl":
                self.bonus[hh, state, action] = self.horizon * np.sqrt(
                    self._beta(n_hsa) / n_hsa
                )

        else:
            self.N_hsa[state, action] += 1
            self.N_hsas[state, action, next_state] += 1

            n_hsa = self.N_hsa[state, action]
            n_hsas = self.N_hsas[state, action, :]
            self.P_hat[state, action, :] = n_hsas / n_hsa

            if self.rf_type == "rf_express":
                self.bonus[state, action] = (
                    15 * (self.horizon**2) * self._beta(n_hsa) / n_hsa
                )

            elif self.rf_type == "rf_ucrl":
                self.bonus[state, action] = self.horizon * np.sqrt(
                    self._beta(n_hsa) / n_hsa
                )

    def run_episode(self):
        state = self.env.reset()
        for hh in range(self.horizon):
            action = self.exploration_policy(state, hh)
            next_s, _, done, _ = self.env.step(action)
            del done

            self.counter.update(state, action, next_s, 0.0)
            self._update(state, action, next_s, hh)

            state = next_s

        # update W
        N = np.maximum(self.N_hsa, 1)

        if self.rf_type == "rf_express":
            multiplier = 1.0 + 1.0 / self.horizon

        elif self.rf_type == "rf_ucrl":
            multiplier = 1.0
        else:
            raise ValueError()

        if self.stage_dependent:
            backward_induction_sd(
                self.W_hsa,
                self.V_hs,
                self.bonus,
                self.P_hat,
                gamma=multiplier,
                vmax=self.vmax,
            )
        else:
            self.W_hsa, _ = backward_induction(
                self.bonus, self.P_hat, self.horizon, gamma=multiplier, vmax=self.vmax
            )

        # write info
        if self.writer is not None:
            self.writer.add_scalar(
                "n_visited_states", self.counter.get_n_visited_states(), self.episode
            )
            self.writer.add_scalar("entropy", self.counter.get_entropy(), self.episode)
            self.writer.add_scalar(
                "error_bound", self.get_error_upper_bound(), self.episode
            )

        self.episode += 1


if __name__ == "__main__":
    from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom

    env = NRoom(nrooms=5, success_probability=0.95)

    params = {
        "rf_type": "rf_express",
        "horizon": 50,
        "delta": 0.1,
        "clip_v": False,
        "stage_dependent": True,
    }

    agent = AdaptiveRFAgent(env, **params)
    agent.fit(2000)
