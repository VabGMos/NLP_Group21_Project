import baycomp
import numpy as np
import scipy.stats as st


class BenavoliCorrelatedT:
    """Estimate the posterior difference (B-A) between each pair of systems."""

    def __init__(self, systems: list, system_names: list = None, runs=1):
        self._cts = []
        combinations = [(i, j) for i in range(len(systems))
                        for j in range(i + 1, len(systems))]

        for comb in combinations:
            sys_a = systems[comb[0]]
            sys_b = systems[comb[1]]
            self._cts.append(baycomp.CorrelatedTTest(
                np.array(sys_a),
                np.array(sys_b),
                rope=0.,
                runs=runs
            ))
            self._names = system_names

    @property
    def mean_diffs(self):
        """Posterior mean difference."""
        return [cts.mean for cts in self._cts]

    @property
    def stds(self):
        """Posterior standard deviation."""
        return [np.sqrt(cts.var) for cts in self._cts]

    @property
    def dfs(self):
        """Posterior degrees of freedom."""
        return [cts.df for cts in self._cts]

    def probs(self, rope_param: float):
        """Return probabilities that A wins, the systems tie, and B wins."""
        p_A = st.t.cdf(-rope_param, self.dfs,
                       loc=self.mean_diffs, scale=self.stds)
        p_tie = st.t.cdf(rope_param, self.dfs, loc=self.mean_diffs, scale=self.stds) - \
            st.t.cdf(-rope_param, self.dfs,
                     loc=self.mean_diffs, scale=self.stds)
        p_B = 1 - st.t.cdf(rope_param, self.dfs,
                           loc=self.mean_diffs, scale=self.stds)
        return p_A, p_tie, p_B
