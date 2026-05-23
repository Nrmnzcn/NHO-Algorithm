import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import cauchy, norm
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class NHO(Optimizer):

    def __init__(self, epoch: int=500, pop_size: int=50, p_elite: float=0.2, F0: float=0.6, CR0: float=0.9, alpha0: float=0.3, delta0: float=0.01, F_bounds: Tuple[float, float]=(0.1, 1.2), lambda_D: float=0.3, lambda_S: float=0.2, lambda_C: float=0.3, kD: float=1.0, kS: float=0.8, kC: float=1.2, kS2: float=0.5, b0: float=2.0, b1: float=4.0, b2: float=2.0, cD: float=1.0, cS: float=0.3, div_threshold: float=0.001, restart_frac: float=0.05, log_hormones: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 10 ** 9])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [3, 10 ** 6])
        self.p_elite = self.validator.check_float('p_elite', p_elite, (0.0, 1.0))
        self.F0 = self.validator.check_float('F0', F0, (0.0, 2.0))
        self.CR0 = self.validator.check_float('CR0', CR0, (0.0, 1.0))
        self.alpha0 = self.validator.check_float('alpha0', alpha0, (0.0, 2.0))
        self.delta0 = self.validator.check_float('delta0', delta0, (0.0, 1.0))
        self.F_min, self.F_max = F_bounds
        self.lambda_D = self.validator.check_float('lambda_D', lambda_D, (0.0, 1.0))
        self.lambda_S = self.validator.check_float('lambda_S', lambda_S, (0.0, 1.0))
        self.lambda_C = self.validator.check_float('lambda_C', lambda_C, (0.0, 1.0))
        self.kD = kD
        self.kS = kS
        self.kC = kC
        self.kS2 = kS2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.cD = cD
        self.cS = cS
        self.div_threshold = self.validator.check_float('div_threshold', div_threshold, (0.0, 1.0))
        self.restart_frac = self.validator.check_float('restart_frac', restart_frac, (0.0, 1.0))
        self.log_hormones = bool(log_hormones)
        self.set_parameters(['p_elite', 'F0', 'CR0', 'alpha0', 'delta0', 'F_min', 'F_max', 'lambda_D', 'lambda_S', 'lambda_C', 'kD', 'kS', 'kC', 'kS2', 'b0', 'b1', 'b2', 'cD', 'cS', 'div_threshold', 'restart_frac', 'log_hormones'])
        self.sort_flag = False
        self.DA = 0.5
        self.ST = 0.5
        self.CORT = 0.0
        self._prev_best_fit = None
        self.convergence_curve: List[float] = []
        self.best_solution_curve: List[np.ndarray] = []
        self.best_fitness_curve: List[float] = []
        self.DA_history: List[float] = []
        self.ST_history: List[float] = []
        self.CORT_history: List[float] = []
        self.succ_history: List[float] = []
        self.div_history: List[float] = []
        self.F_history: List[float] = []
        self.CR_history: List[float] = []
        self.alpha_history: List[float] = []
        self.delta_history: List[float] = []

    def _is_min(self) -> bool:
        return getattr(self.problem, 'minmax', 'min') == 'min'

    def _better(self, a: float, b: float) -> bool:
        return a < b if self._is_min() else a > b

    def _get_lb_ub(self):
        pb = self.problem
        if hasattr(pb, 'lb') and hasattr(pb, 'ub'):
            lb = np.asarray(pb.lb, float)
            ub = np.asarray(pb.ub, float)
        elif hasattr(pb, 'bounds'):
            lb = np.asarray(pb.bounds.lb, float)
            ub = np.asarray(pb.bounds.ub, float)
        else:
            lb = np.full(pb.n_dims, -1.0)
            ub = np.full(pb.n_dims, 1.0)
        return (lb, ub)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def initialize_variables(self):
        self.convergence_curve.clear()
        self.best_solution_curve.clear()
        self.best_fitness_curve.clear()
        self.DA_history.clear()
        self.ST_history.clear()
        self.CORT_history.clear()
        self.succ_history.clear()
        self.div_history.clear()
        self.F_history.clear()
        self.CR_history.clear()
        self.alpha_history.clear()
        self.delta_history.clear()
        self.DA, self.ST, self.CORT = (0.5, 0.5, 0.0)
        self._prev_best_fit = None

    def evolve(self, epoch: int) -> None:
        N, D = (self.pop_size, self.problem.n_dims)
        lb, ub = self._get_lb_ub()
        rng = self.generator
        X = np.stack([ag.solution for ag in self.pop], axis=0)
        f = np.array([ag.target.fitness for ag in self.pop], float)
        order = np.argsort(f) if self._is_min() else np.argsort(-f)
        K = max(1, int(np.ceil(self.p_elite * N)))
        elite_idx = order[:K]
        m = np.mean(X[elite_idx], axis=0)
        gbest = self.pop[order[0]]
        xbest = gbest.solution.copy()
        tau = (epoch + 1) / max(1, self.epoch)
        Ft = float(np.clip(self.F0 * np.exp(self.kD * self.DA - self.kS * self.ST), self.F_min, self.F_max))
        CRt = float(self._sigmoid(self.b0 + self.b1 * self.ST - self.b2 * self.DA))
        alphat = self.alpha0 * (1 - tau) * (self.cD * self.DA + self.cS * (1 - self.ST))
        deltat = self.delta0 * (1 - tau) * np.exp(self.kC * self.CORT - self.kS2 * self.ST)
        improved = 0
        for i in range(N):
            idx = list(range(N))
            idx.remove(i)
            r1, r2 = rng.choice(idx, size=2, replace=False)
            d = X[r1] - X[r2]
            v1 = X[i] + Ft * d + (m - X[i]) * (0.8 * (1 - tau))
            u = m - X[i]
            un = np.linalg.norm(u)
            o = d - np.dot(d, u / un) * (u / un) if un > 1e-12 else d
            v2 = X[i] + Ft * o
            gauss = rng.normal(0.0, 1.0, size=D) * deltat * (ub - lb)
            v3 = X[i] + alphat * (xbest - X[i]) + gauss

            def mix_fix(cand):
                U = rng.random(D)
                y = np.where(U < CRt, cand, X[i])
                y = np.asarray(y, float)
                y = np.nan_to_num(y, copy=False)
                return self.correct_solution(y)
            y1 = mix_fix(v1)
            y2 = mix_fix(v2)
            y3 = mix_fix(v3)
            a1 = self.generate_empty_agent(y1)
            a1.target = self.get_target(y1)
            a2 = self.generate_empty_agent(y2)
            a2.target = self.get_target(y2)
            a3 = self.generate_empty_agent(y3)
            a3.target = self.get_target(y3)
            best_cand = a1
            if self._better(a2.target.fitness, best_cand.target.fitness):
                best_cand = a2
            if self._better(a3.target.fitness, best_cand.target.fitness):
                best_cand = a3
            if self._better(best_cand.target.fitness, f[i]):
                self.pop[i] = best_cand
                X[i] = best_cand.solution.copy()
                f[i] = best_cand.target.fitness
                improved += 1
                if self._better(f[i], gbest.target.fitness):
                    gbest = best_cand
                    xbest = best_cand.solution.copy()
        span = ub - lb + 1e-12
        div = float(np.mean(np.std(X, axis=0) / span))
        if div < self.div_threshold:
            q = max(1, int(self.restart_frac * N))
            worst_idx = order[-q:]
            radius = 0.1 * span * (1 - tau)
            for j in worst_idx:
                newx = xbest + rng.uniform(-1, 1, size=D) * radius
                newx = self.correct_solution(newx)
                ag = self.generate_empty_agent(newx)
                ag.target = self.get_target(newx)
                if self._better(ag.target.fitness, f[j]):
                    self.pop[j] = ag
                    X[j] = ag.solution.copy()
                    f[j] = ag.target.fitness
        curr_best = self.pop[order[0]].target.fitness if order.size > 0 else gbest.target.fitness
        if self._prev_best_fit is None:
            rpe = 0.0
        else:
            num = self._prev_best_fit - curr_best
            den = abs(self._prev_best_fit) + 1e-12
            rpe = max(0.0, float(num / den))
        self._prev_best_fit = curr_best
        self.DA = (1 - self.lambda_D) * self.DA + self.lambda_D * rpe
        self.ST = (1 - self.lambda_S) * self.ST + self.lambda_S * (1 - rpe)
        self.CORT = (1 - self.lambda_C) * self.CORT + self.lambda_C * (1 - div)
        _, bests, _ = self.get_special_agents(self.pop, n_best=1, minmax=self.problem.minmax)
        b = bests[0]
        self.convergence_curve.append(b.target.fitness)
        self.best_fitness_curve.append(b.target.fitness)
        self.best_solution_curve.append(b.solution.copy())
        if self.log_hormones:
            self.DA_history.append(float(self.DA))
            self.ST_history.append(float(self.ST))
            self.CORT_history.append(float(self.CORT))
            self.succ_history.append(float(improved / max(1, N)))
            self.div_history.append(div)
            self.F_history.append(Ft)
            self.CR_history.append(CRt)
            self.alpha_history.append(alphat)
            self.delta_history.append(deltat)

    def get_logs(self) -> Dict[str, np.ndarray]:
        return {'DA': np.array(self.DA_history), 'ST': np.array(self.ST_history), 'CORT': np.array(self.CORT_history), 'success': np.array(self.succ_history), 'diversity': np.array(self.div_history), 'F': np.array(self.F_history), 'CR': np.array(self.CR_history), 'alpha': np.array(self.alpha_history), 'delta': np.array(self.delta_history), 'convergence': np.array(self.convergence_curve)}


class ACSA(Optimizer):

    def __init__(self, epoch: int=1000, pop_size: int=50, NH_rate: float=0.1, S: float=0.1, Ea_neural: float=1.8, Es_neural: float=0.002, Ea_hormonal: float=0.66, Es_hormonal: float=0.33, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 1000000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [3, 100000])
        self.NH_rate = self.validator.check_float('NH_rate', NH_rate, (0.0, 1.0))
        self.S = self.validator.check_float('S', S, (0.0, 1.0))
        self.Ea_neural = self.validator.check_float('Ea_neural', Ea_neural, (0.0, np.inf))
        self.Es_neural = self.validator.check_float('Es_neural', Es_neural, (0.0, np.inf))
        self.Ea_hormonal = self.validator.check_float('Ea_hormonal', Ea_hormonal, (0.0, np.inf))
        self.Es_hormonal = self.validator.check_float('Es_hormonal', Es_hormonal, (0.0, np.inf))
        self.set_parameters(['epoch', 'pop_size', 'NH_rate', 'S', 'Ea_neural', 'Es_neural', 'Ea_hormonal', 'Es_hormonal'])
        self.sort_flag = False
        self.convergence_curve: List[float] = []
        self.best_fitness_curve: List[float] = []
        self.best_solution_curve: List[np.ndarray] = []

    def initialize_variables(self):
        self.convergence_curve = []
        self.best_fitness_curve = []
        self.best_solution_curve = []

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return solution

    def _is_min(self) -> bool:
        return getattr(self.problem, 'minmax', 'min') == 'min'

    def _fitness_better(self, fa: float, fb: float) -> bool:
        return fa < fb if self._is_min() else fa > fb

    def _get_lb_ub(self) -> Tuple[np.ndarray, np.ndarray]:
        pb = self.problem
        if hasattr(pb, 'lb') and hasattr(pb, 'ub'):
            lb = np.asarray(pb.lb, dtype=float)
            ub = np.asarray(pb.ub, dtype=float)
        elif hasattr(pb, 'bounds') and hasattr(pb.bounds, 'lb') and hasattr(pb.bounds, 'ub'):
            lb = np.asarray(pb.bounds.lb, dtype=float)
            ub = np.asarray(pb.bounds.ub, dtype=float)
        else:
            lb = np.full(pb.n_dims, -1.0, dtype=float)
            ub = np.full(pb.n_dims, 1.0, dtype=float)
        return (lb, ub)

    def _stimulation(self, x: np.ndarray, is_neural: bool, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        Ea = self.Ea_neural if is_neural else self.Ea_hormonal
        Es = self.Es_neural if is_neural else self.Es_hormonal
        return (ub - lb) * self.S / (1.0 + np.exp(Ea * (-x + Es)))

    def evolve(self, epoch: int) -> None:
        N = self.pop_size
        D = self.problem.n_dims
        lb, ub = self._get_lb_ub()
        curr_positions = np.stack([ag.solution for ag in self.pop], axis=0)
        curr_fits = np.array([ag.target.fitness for ag in self.pop], dtype=float)
        if self._is_min():
            sorted_idx = np.argsort(curr_fits)[::-1]
        else:
            sorted_idx = np.argsort(curr_fits)
        neural_size = int(np.clip(int(round(N * self.NH_rate)), 1, max(1, N - 1)))
        neural_idx = sorted_idx[:neural_size]
        hormonal_idx = sorted_idx[neural_size:]
        pop_new: List[Agent] = []
        pop_mean = np.mean(curr_positions, axis=0)
        gbest_sol = self.g_best.solution
        for i in range(N):
            xi = curr_positions[i]
            if i in neural_idx:
                sf = self._stimulation(xi, is_neural=True, lb=lb, ub=ub)
                cand = self.generator.random(D) * sf
            elif self.generator.random() < 0.5:
                sf = self._stimulation(xi, is_neural=False, lb=lb, ub=ub)
                cand = pop_mean + self.generator.random(D) * sf
            else:
                cand = gbest_sol + self.generator.random(D) * (pop_mean - xi)
            cand = self.correct_solution(cand)
            agent_new = self.generate_empty_agent(cand)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(cand)
                self.pop[i] = self.get_better_agent(self.pop[i], agent_new, minmax=self.problem.minmax)
            else:
                pop_new.append(agent_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, bests, _ = self.get_special_agents(self.pop, n_best=1, minmax=self.problem.minmax)
        best = bests[0]
        self.convergence_curve.append(best.target.fitness)
        self.best_fitness_curve.append(best.target.fitness)
        self.best_solution_curve.append(best.solution.copy())


class SRA(Optimizer):

    def __init__(self, epoch: int=1000, pop_size: int=50, L: float=0.5, h: float=6.625e-34, restart_prob: float=0.03, levy_beta: float=1.5, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 1000000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [3, 100000])
        self.L = self.validator.check_float('L', L, (0.0, np.inf))
        self.h = self.validator.check_float('h', h, (0.0, np.inf))
        self.restart_prob = self.validator.check_float('restart_prob', restart_prob, (0.0, 1.0))
        self.levy_beta = self.validator.check_float('levy_beta', levy_beta, (1.0, 3.0))
        self.set_parameters(['epoch', 'pop_size', 'L', 'h', 'restart_prob', 'levy_beta'])
        self.sort_flag = False
        self.convergence_curve: List[float] = []
        self.best_fitness_curve: List[float] = []
        self.best_solution_curve: List[np.ndarray] = []
        self._psi: Optional[np.ndarray] = None

    def initialize_variables(self):
        self.convergence_curve = []
        self.best_fitness_curve = []
        self.best_solution_curve = []
        self._psi = None

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return solution

    def _is_min(self) -> bool:
        return getattr(self.problem, 'minmax', 'min') == 'min'

    def _fitness_better(self, fa: float, fb: float) -> bool:
        return fa < fb if self._is_min() else fa > fb

    def _get_lb_ub(self) -> Tuple[np.ndarray, np.ndarray]:
        pb = self.problem
        if hasattr(pb, 'lb') and hasattr(pb, 'ub'):
            lb = np.asarray(pb.lb, dtype=float)
            ub = np.asarray(pb.ub, dtype=float)
        elif hasattr(pb, 'bounds') and hasattr(pb.bounds, 'lb') and hasattr(pb.bounds, 'ub'):
            lb = np.asarray(pb.bounds.lb, dtype=float)
            ub = np.asarray(pb.bounds.ub, dtype=float)
        else:
            lb = np.full(pb.n_dims, -1.0, dtype=float)
            ub = np.full(pb.n_dims, 1.0, dtype=float)
        return (lb, ub)

    def _init_psi_if_needed(self):
        if self._psi is not None:
            return
        L = self.L
        sols = np.stack([ag.solution for ag in self.pop], axis=0)
        self._psi = np.sqrt(2.0 / L) * np.sin(sols) * np.exp(2.0)

    def _levy(self, dim: int, beta: float) -> np.ndarray:
        num = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2.0)
        den = np.math.gamma((1 + beta) / 2.0) * beta * 2 ** ((beta - 1) / 2.0)
        sigma = (num / den) ** (1.0 / beta)
        u = 0.01 * self.generator.normal(loc=0.0, scale=sigma, size=dim)
        v = self.generator.normal(loc=0.0, scale=1.0, size=dim)
        step = u / (np.abs(v) ** (1.0 / beta) + 1e-12)
        return step

    def evolve(self, epoch: int) -> None:
        self._init_psi_if_needed()
        N = self.pop_size
        D = self.problem.n_dims
        lb, ub = self._get_lb_ub()
        positions = np.stack([ag.solution for ag in self.pop], axis=0)
        fits = np.array([ag.target.fitness for ag in self.pop], dtype=float)
        psi = self._psi
        order = np.argsort(fits) if self._is_min() else np.argsort(-fits)
        best_idx = order[0]
        worst_idx = order[-1]
        Best_X = positions[best_idx].copy()
        Best_Cost = fits[best_idx]
        Worst_Cost = fits[worst_idx]
        Best_Psai = psi[best_idx].copy()
        Worst_Psai = psi[worst_idx].copy()
        t = epoch
        T = self.epoch
        b = 1.0 - t ** (1.0 / 5.0) / T ** (1.0 / 5.0)
        Threshold = (t / T) ** 3 if T > 0 else 1.0
        seq = np.arange(N)
        R = N - seq
        p = (R / N) ** 2
        pop_new: List[Agent] = []
        eps = 1e-12
        for i in range(N):
            xi = positions[i]
            if self.generator.random() < self.restart_prob:
                cand = self.generator.uniform(lb, ub, size=D)
            else:
                vc = self.generator.uniform(-b, b, size=D)
                Z = self._levy(D, self.levy_beta)
                pool = [idx for idx in range(N) if idx != i]
                id_1, id_2 = self.generator.choice(pool, size=2, replace=False)
                if np.abs(p[i]) < Threshold:
                    if self.generator.random() < 0.5:
                        prev = positions[i - 1]
                        cand = 1.0 * self.generator.random() + 2.0 * xi - prev
                    else:
                        cand = Best_X - 0.1 * Z + self.generator.random() * ((ub - lb) * self.generator.random(size=D) + lb)
                else:
                    term = self.h * (Best_Psai - Worst_Psai) + p[i] * (psi[id_1] - 2.0 * psi[i] + psi[id_2])
                    term2 = self.h * (Best_Psai - Worst_Psai) + p[i] * (psi[id_1] + 2.0 * psi[i] + psi[id_2])
                    scale1 = self.generator.random() * vc
                    scale2 = self.generator.random() * vc
                    pos_1 = Best_X + scale1 * (term / (psi[i] + eps))
                    pos_2 = xi + scale2 * (term2 / (psi[i] + eps))
                    mask = self.generator.random(D) < 0.5
                    cand = np.where(mask, pos_1, pos_2)
            cand = self.correct_solution(cand)
            agent_new = self.generate_empty_agent(cand)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(cand)
                if self._fitness_better(agent_new.target.fitness, fits[i]):
                    self.pop[i] = agent_new
                    positions[i] = cand
                    fits[i] = agent_new.target.fitness
                    if self._fitness_better(fits[i], Best_Cost):
                        Best_Cost = fits[i]
                        Best_X = positions[i].copy()
                    if self._fitness_better(Worst_Cost, fits[i]):
                        Worst_Cost = fits[i]
                psi[i] = np.sin(self.generator.random() * positions[i])
            else:
                pop_new.append(agent_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
            positions = np.stack([ag.solution for ag in self.pop], axis=0)
            fits = np.array([ag.target.fitness for ag in self.pop], dtype=float)
            psi = np.sin(self.generator.random((N, 1)) * positions)
            self._psi = psi
        else:
            self._psi = psi
        _, bests, _ = self.get_special_agents(self.pop, n_best=1, minmax=self.problem.minmax)
        best = bests[0]
        self.convergence_curve.append(best.target.fitness)
        self.best_fitness_curve.append(best.target.fitness)
        self.best_solution_curve.append(best.solution.copy())


class CHO(Optimizer):

    def __init__(self, epoch=500, pop_size=50, n_clusters=5, p_comb=0.5, p_comp=0.5, c=0.25, elite_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, float('inf')])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [2, float('inf')])
        self.n_clusters = self.validator.check_int('n_clusters', n_clusters, [1, self.pop_size])
        self.p_comb = self.validator.check_float('p_comb', p_comb, [0.0, 1.0])
        self.p_comp = self.validator.check_float('p_comp', p_comp, [0.0, 1.0])
        self.c = self.validator.check_float('c', c, [0.0, 1.0])
        self.elite_ratio = self.validator.check_float('elite_ratio', elite_ratio, [0.0, 1.0])
        self.set_parameters(['n_clusters', 'p_comb', 'p_comp', 'c', 'elite_ratio'])
        self.sort_flag = False

    def _partition_into_clusters(self, pop_sorted):
        n = len(pop_sorted)
        k = max(1, min(self.n_clusters, n))
        sizes = [n // k] * k
        for i in range(n % k):
            sizes[i] += 1
        clusters, start = ([], 0)
        for s in sizes:
            clusters.append(pop_sorted[start:start + s])
            start += s
        return clusters

    def _roulette_on_pool(self, pool):
        fits = np.array([ag.target.fitness for ag in pool], dtype=float)
        idx = self.get_index_roulette_wheel_selection(fits)
        return idx

    def _inverse_roulette_on_pool(self, pool):
        fits = np.array([ag.target.fitness for ag in pool], dtype=float)
        idx = self.get_index_roulette_wheel_selection(-fits)
        return idx

    def evolve(self, epoch: int) -> None:
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        clusters = self._partition_into_clusters(pop_sorted)
        elite_cut = max(1, int(np.ceil(self.elite_ratio * len(pop_sorted))))
        elite_pool = pop_sorted[:elite_cut]
        nonelite_pool = pop_sorted[elite_cut:] if elite_cut < len(pop_sorted) else []
        offspring = []
        n_comb = max(1, int(self.p_comb * self.pop_size))
        if len(elite_pool) >= 2:
            cluster_map = {}
            for ci, cl in enumerate(clusters):
                for ag in cl:
                    cluster_map[id(ag)] = ci
            for _ in range(n_comb):
                i1 = self._roulette_on_pool(elite_pool)
                p1 = elite_pool[i1]
                tries = 0
                while True:
                    i2 = self._roulette_on_pool(elite_pool)
                    if i2 != i1 and cluster_map.get(id(elite_pool[i2]), -1) != cluster_map.get(id(p1), -1):
                        break
                    tries += 1
                    if tries > 8:
                        if i2 == i1:
                            i2 = (i1 + 1) % len(elite_pool)
                        break
                p2 = elite_pool[i2]
                a = -self.c + self.generator.uniform() * (1.0 + 2.0 * self.c)
                x1, x2 = (p1.solution, p2.solution)
                y1 = a * x1 + (1 - a) * x2
                y2 = a * x2 + (1 - a) * x1
                y1 = self.correct_solution(y1)
                y2 = self.correct_solution(y2)
                off1 = self.generate_empty_agent(y1)
                off1.target = self.get_target(y1)
                off2 = self.generate_empty_agent(y2)
                off2.target = self.get_target(y2)
                offspring.extend([off1, off2])
        n_comp = max(1, int(self.p_comp * self.pop_size))
        if len(nonelite_pool) >= 2:
            for _ in range(n_comp):
                i1 = self._inverse_roulette_on_pool(nonelite_pool)
                i2 = self._inverse_roulette_on_pool(nonelite_pool)
                if i2 == i1:
                    i2 = (i1 + 1) % len(nonelite_pool)
                a1, a2 = (nonelite_pool[i1], nonelite_pool[i2])
                if self.compare_fitness(a1.target.fitness, a2.target.fitness, self.problem.minmax):
                    x1, x2 = (a1, a2)
                else:
                    x1, x2 = (a2, a1)
                d = self.generator.uniform(0.0, 1.0)
                f = self.generator.uniform(0.0, 0.25)
                pos_new = d * x1.solution + f * self.g_best.solution
                pos_new = self.correct_solution(pos_new)
                ag_new = self.generate_empty_agent(pos_new)
                ag_new.target = self.get_target(pos_new)
                if self.compare_fitness(ag_new.target.fitness, x2.target.fitness, self.problem.minmax):
                    offspring.append(ag_new)
                else:
                    offspring.append(x2.copy())
        pool_all = self.pop + offspring if offspring else self.pop
        self.pop = self.get_sorted_and_trimmed_population(pool_all, self.pop_size, self.problem.minmax)


class BPBO(Optimizer):

    def __init__(self, epoch: int=1000, pop_size: int=90, Pi: float=0.7, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 1000000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [3, 100000])
        self.Pi = self.validator.check_float('Pi', Pi, (0.0, 1.0))
        self.set_parameters(['epoch', 'pop_size', 'Pi'])
        self.sort_flag = False
        self.convergence_curve: List[float] = []
        self.best_fitness_curve: List[float] = []
        self.best_solution_curve: List[np.ndarray] = []

    def initialize_variables(self):
        self.convergence_curve = []
        self.best_fitness_curve = []
        self.best_solution_curve = []

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return solution

    def _is_min(self) -> bool:
        return getattr(self.problem, 'minmax', 'min') == 'min'

    def _fitness_better(self, fa: float, fb: float) -> bool:
        return fa < fb if self._is_min() else fa > fb

    def _get_lb_ub(self) -> Tuple[np.ndarray, np.ndarray]:
        pb = self.problem
        if hasattr(pb, 'lb') and hasattr(pb, 'ub'):
            lb = np.asarray(pb.lb, dtype=float)
            ub = np.asarray(pb.ub, dtype=float)
        elif hasattr(pb, 'bounds') and hasattr(pb.bounds, 'lb') and hasattr(pb.bounds, 'ub'):
            lb = np.asarray(pb.bounds.lb, dtype=float)
            ub = np.asarray(pb.bounds.ub, dtype=float)
        else:
            lb = np.full(pb.n_dims, -1.0, dtype=float)
            ub = np.full(pb.n_dims, 1.0, dtype=float)
        return (lb, ub)

    def evolve(self, epoch: int) -> None:
        N = self.pop_size
        D = self.problem.n_dims
        lb, ub = self._get_lb_ub()
        positions = np.stack([ag.solution for ag in self.pop], axis=0)
        fits = np.array([ag.target.fitness for ag in self.pop], dtype=float)
        order = np.argsort(fits) if self._is_min() else np.argsort(-fits)
        prey = self.pop[order[0]]
        worst = self.pop[order[-1]]
        mean_pos = np.mean(positions, axis=0)
        prey_pos = prey.solution.copy()
        prey_fit = prey.target.fitness
        worst_pos = worst.solution.copy()
        pop_new: List[Agent] = []
        for i in range(N):
            xi = positions[i]
            if self.generator.random() < self.Pi:
                r1 = self.generator.random()
                r2 = self.generator.random()
                if self.generator.random() < self.generator.random():
                    M00 = 1 if self.generator.random() < 0.5 else 2
                    step = self.generator.random(D) * (prey_pos - M00 * xi)
                    cand = xi + step
                elif self.generator.random() < self.generator.random():
                    M01 = 1 if self.generator.random() < 0.5 else 2
                    step = self.generator.random(D) * (prey_pos - M01 * mean_pos)
                    cand = mean_pos + step
                else:
                    M02 = 1 if self.generator.random() < 0.5 else 2
                    step = self.generator.random(D) * (xi - M02 * worst_pos)
                    cand = xi + step
            else:
                cand = xi + self.generator.random() * self.generator.uniform(lb, ub, size=D)
            cand = self.correct_solution(cand)
            agent_new = self.generate_empty_agent(cand)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(cand)
                if self._fitness_better(agent_new.target.fitness, fits[i]):
                    self.pop[i] = agent_new
                    positions[i] = cand
                    fits[i] = agent_new.target.fitness
                    if self._fitness_better(fits[i], prey_fit):
                        prey_pos = positions[i].copy()
                        prey_fit = fits[i]
            else:
                pop_new.append(agent_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, bests, _ = self.get_special_agents(self.pop, n_best=1, minmax=self.problem.minmax)
        best = bests[0]
        self.convergence_curve.append(best.target.fitness)
        self.best_fitness_curve.append(best.target.fitness)
        self.best_solution_curve.append(best.solution.copy())


# MealPy kütüphanesinden: OriginalDE
class DE(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, wf: float=0.1, cr: float=0.9, strategy: int=0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.wf = self.validator.check_float('wf', wf, (-3.0, 3.0))
        self.cr = self.validator.check_float('cr', cr, (0, 1.0))
        self.strategy = self.validator.check_int('strategy', strategy, [0, 5])
        self.set_parameters(['epoch', 'pop_size', 'wf', 'cr', 'strategy'])
        self.sort_flag = False

    def mutation__(self, current_pos, new_pos):
        condition = self.generator.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)
        return self.correct_solution(pos_new)

    def evolve(self, epoch):
        pop = []
        if self.strategy == 0:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx_list[0]].solution + self.wf * (self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        elif self.strategy == 1:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.g_best.solution + self.wf * (self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        elif self.strategy == 2:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
                pos_new = self.g_best.solution + self.wf * (self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution) + self.wf * (self.pop[idx_list[2]].solution - self.pop[idx_list[3]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        elif self.strategy == 3:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 5, replace=False)
                pos_new = self.pop[idx_list[0]].solution + self.wf * (self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution) + self.wf * (self.pop[idx_list[3]].solution - self.pop[idx_list[4]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        elif self.strategy == 4:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx].solution + self.wf * (self.g_best.solution - self.pop[idx].solution) + self.wf * (self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        else:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx].solution + self.wf * (self.pop[idx_list[0]].solution - self.pop[idx].solution) + self.wf * (self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)


# MealPy kütüphanesinden: BaseGA
class GA(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, pc: float=0.95, pm: float=0.025, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.pc = self.validator.check_float('pc', pc, (0, 1.0))
        self.pm = self.validator.check_float('pm', pm, (0, 1.0))
        self.set_parameters(['epoch', 'pop_size', 'pc', 'pm'])
        self.sort_flag = False
        self.selection = 'tournament'
        self.k_way = 0.2
        self.crossover = 'uniform'
        self.mutation = 'flip'
        self.mutation_multipoints = True
        if 'selection' in kwargs:
            self.selection = self.validator.check_str('selection', kwargs['selection'], ['tournament', 'random', 'roulette'])
        if 'k_way' in kwargs:
            self.k_way = self.validator.check_float('k_way', kwargs['k_way'], (0, 1.0))
        if 'crossover' in kwargs:
            self.crossover = self.validator.check_str('crossover', kwargs['crossover'], ['one_point', 'multi_points', 'uniform', 'arithmetic'])
        if 'mutation_multipoints' in kwargs:
            self.mutation_multipoints = self.validator.check_bool('mutation_multipoints', kwargs['mutation_multipoints'])
        if self.mutation_multipoints:
            if 'mutation' in kwargs:
                self.mutation = self.validator.check_str('mutation', kwargs['mutation'], ['flip', 'swap'])
        elif 'mutation' in kwargs:
            self.mutation = self.validator.check_str('mutation', kwargs['mutation'], ['flip', 'swap', 'scramble', 'inversion'])

    def selection_process__(self, list_fitness):
        if self.selection == 'roulette':
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
            while id_c2 == id_c1:
                id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == 'random':
            id_c1, id_c2 = self.generator.choice(range(self.pop_size), 2, replace=False)
        else:
            id_c1, id_c2 = self.get_index_kway_tournament_selection(self.pop, k_way=self.k_way, output=2)
        return (self.pop[id_c1].solution, self.pop[id_c2].solution)

    def selection_process_00__(self, pop_selected):
        if self.selection == 'roulette':
            list_fitness = np.array([agent.target.fitness for agent in pop_selected])
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
            while id_c2 == id_c1:
                id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == 'random':
            id_c1, id_c2 = self.generator.choice(range(len(pop_selected)), 2, replace=False)
        else:
            id_c1, id_c2 = self.get_index_kway_tournament_selection(pop_selected, k_way=self.k_way, output=2)
        return (pop_selected[id_c1].solution, pop_selected[id_c2].solution)

    def selection_process_01__(self, pop_dad, pop_mom):
        if self.selection == 'roulette':
            list_fit_dad = np.array([agent.target.fitness for agent in pop_dad])
            list_fit_mom = np.array([agent.target.fitness for agent in pop_mom])
            id_c1 = self.get_index_roulette_wheel_selection(list_fit_dad)
            id_c2 = self.get_index_roulette_wheel_selection(list_fit_mom)
        elif self.selection == 'random':
            id_c1 = self.generator.choice(range(len(pop_dad)))
            id_c2 = self.generator.choice(range(len(pop_mom)))
        else:
            id_c1 = self.get_index_kway_tournament_selection(pop_dad, k_way=self.k_way, output=1)[0]
            id_c2 = self.get_index_kway_tournament_selection(pop_mom, k_way=self.k_way, output=1)[0]
        return (pop_dad[id_c1].solution, pop_mom[id_c2].solution)

    def crossover_process__(self, dad, mom):
        if self.crossover == 'arithmetic':
            w1, w2 = self.crossover_arithmetic(dad, mom)
        elif self.crossover == 'one_point':
            cut = self.generator.integers(1, self.problem.n_dims - 1)
            w1 = np.concatenate([dad[:cut], mom[cut:]])
            w2 = np.concatenate([mom[:cut], dad[cut:]])
        elif self.crossover == 'multi_points':
            idxs = self.generator.choice(range(1, self.problem.n_dims - 1), 2, replace=False)
            cut1, cut2 = (np.min(idxs), np.max(idxs))
            w1 = np.concatenate([dad[:cut1], mom[cut1:cut2], dad[cut2:]])
            w2 = np.concatenate([mom[:cut1], dad[cut1:cut2], mom[cut2:]])
        else:
            flip = self.generator.integers(0, 2, self.problem.n_dims)
            w1 = dad * flip + mom * (1 - flip)
            w2 = mom * flip + dad * (1 - flip)
        return (w1, w2)

    def mutation_process__(self, child):
        if self.mutation_multipoints:
            if self.mutation == 'swap':
                for idx in range(self.problem.n_dims):
                    idx_swap = self.generator.choice(list(set(range(0, self.problem.n_dims)) - {idx}))
                    child[idx], child[idx_swap] = (child[idx_swap], child[idx])
                    return child
            else:
                mutation_child = self.problem.generate_solution()
                flag_child = self.generator.uniform(0, 1, self.problem.n_dims) < self.pm
                return np.where(flag_child, mutation_child, child)
        elif self.mutation == 'swap':
            idx1, idx2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
            child[idx1], child[idx2] = (child[idx2], child[idx1])
            return child
        elif self.mutation == 'inversion':
            cut1, cut2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
            temp = child[cut1:cut2]
            temp = temp[::-1]
            child[cut1:cut2] = temp
            return child
        elif self.mutation == 'scramble':
            cut1, cut2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
            temp = child[cut1:cut2]
            self.generator.shuffle(temp)
            child[cut1:cut2] = temp
            return child
        else:
            idx = self.generator.integers(0, self.problem.n_dims)
            child[idx] = self.generator.uniform(self.problem.lb[idx], self.problem.ub[idx])
            return child

    def survivor_process__(self, pop, pop_child):
        pop_new = []
        for idx in range(0, self.pop_size):
            id_child = self.get_index_kway_tournament_selection(pop, k_way=0.1, output=1, reverse=True)[0]
            pop_new.append(self.get_better_agent(pop_child[idx], pop[id_child], self.problem.minmax))
        return pop_new

    def evolve(self, epoch):
        list_fitness = np.array([agent.target.fitness for agent in self.pop])
        pop_new = []
        for i in range(0, int(self.pop_size / 2)):
            child1, child2 = self.selection_process__(list_fitness)
            if self.generator.random() < self.pc:
                child1, child2 = self.crossover_process__(child1, child2)
            child1 = self.mutation_process__(child1)
            child2 = self.mutation_process__(child2)
            child1 = self.correct_solution(child1)
            child2 = self.correct_solution(child2)
            agent1 = self.generate_empty_agent(child1)
            agent2 = self.generate_empty_agent(child2)
            pop_new.append(agent1)
            pop_new.append(agent2)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-2].target = self.get_target(child1)
                pop_new[-1].target = self.get_target(child2)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = self.survivor_process__(self.pop, pop_new)


# MealPy kütüphanesinden: OriginalHGSO
class HGSO(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, n_clusters: int=2, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [10, 10000])
        self.n_clusters = self.validator.check_int('n_clusters', n_clusters, [2, int(self.pop_size / 5)])
        self.set_parameters(['epoch', 'pop_size', 'n_clusters'])
        self.n_elements = int(self.pop_size / self.n_clusters)
        self.sort_flag = False
        self.T0 = 298.15
        self.K = 1.0
        self.beta = 1.0
        self.alpha = 1
        self.epsilon = 0.05
        self.l1 = 0.05
        self.l2 = 100.0
        self.l3 = 0.01

    def initialize_variables(self):
        self.H_j = self.l1 * self.generator.uniform()
        self.P_ij = self.l2 * self.generator.uniform()
        self.C_j = self.l3 * self.generator.uniform()
        self.pop_group, self.p_best = (None, None)

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_group = self.generate_group_population(self.pop, self.n_clusters, self.n_elements)
        self.p_best = self.get_best_solution_in_team__(self.pop_group)

    def flatten_group__(self, group):
        pop = []
        for idx in range(0, self.n_clusters):
            pop += group[idx]
        return pop

    def get_best_solution_in_team__(self, group=None):
        list_best = []
        for idx in range(len(group)):
            best_agent = self.get_best_agent(group[idx], self.problem.minmax)
            list_best.append(best_agent)
        return list_best

    def evolve(self, epoch):
        for idx in range(self.n_clusters):
            pop_new = []
            for jdx in range(self.n_elements):
                F = -1.0 if self.generator.uniform() < 0.5 else 1.0
                self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
                S_ij = self.K * self.H_j * self.P_ij
                gama = self.beta * np.exp(-((self.p_best[idx].target.fitness + self.epsilon) / (self.pop_group[idx][jdx].target.fitness + self.epsilon)))
                pos_new = self.pop_group[idx][jdx].solution + F * self.generator.uniform() * gama * (self.p_best[idx].solution - self.pop_group[idx][jdx].solution) + F * self.generator.uniform() * self.alpha * (S_ij * self.g_best.solution - self.pop_group[idx][jdx].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            pop_new = self.update_target_for_population(pop_new)
            self.pop_group[idx] = pop_new
        self.pop = self.flatten_group__(self.pop_group)
        self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
        S_ij = self.K * self.H_j * self.P_ij
        N_w = int(self.pop_size * (self.generator.uniform(0, 0.1) + 0.1))
        sorted_id_pos = np.argsort([x.target.fitness for x in self.pop])
        pop_new = []
        pop_idx = []
        for item in range(N_w):
            id = sorted_id_pos[item]
            pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_idx.append(id)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx, id_selected in enumerate(pop_idx):
            self.pop[id_selected] = pop_new[idx].copy()
        self.pop_group = self.generate_group_population(self.pop, self.n_clusters, self.n_elements)
        self.p_best = self.get_best_solution_in_team__(self.pop_group)


# MealPy kütüphanesinden: OriginalACOR
class ACO(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, sample_count: int=25, intent_factor: float=0.5, zeta: float=1.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.sample_count = self.validator.check_int('sample_count', sample_count, [2, 10000])
        self.intent_factor = self.validator.check_float('intent_factor', intent_factor, (0, 1.0))
        self.zeta = self.validator.check_float('zeta', zeta, (0, 5))
        self.set_parameters(['epoch', 'pop_size', 'sample_count', 'intent_factor', 'zeta'])
        self.sort_flag = True

    def evolve(self, epoch):
        pop_rank = np.array([idx for idx in range(1, self.pop_size + 1)])
        qn = self.intent_factor * self.pop_size
        matrix_w = 1 / (np.sqrt(2 * np.pi) * qn) * np.exp(-0.5 * ((pop_rank - 1) / qn) ** 2)
        matrix_p = matrix_w / np.sum(matrix_w)
        matrix_pos = np.array([agent.solution for agent in self.pop])
        matrix_sigma = []
        for idx in range(0, self.pop_size):
            matrix_i = np.repeat(self.pop[idx].solution.reshape((1, -1)), self.pop_size, axis=0)
            D = np.sum(np.abs(matrix_pos - matrix_i), axis=0)
            temp = self.zeta * D / (self.pop_size - 1)
            matrix_sigma.append(temp)
        matrix_sigma = np.array(matrix_sigma)
        pop_new = []
        for idx in range(0, self.sample_count):
            child = np.zeros(self.problem.n_dims)
            for jdx in range(0, self.problem.n_dims):
                rdx = self.get_index_roulette_wheel_selection(matrix_p)
                child[jdx] = self.pop[rdx].solution[jdx] + self.generator.normal() * matrix_sigma[rdx, jdx]
            pos_new = self.correct_solution(child)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)


# MealPy kütüphanesinden: OriginalGWO
class GWO(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.set_parameters(['epoch', 'pop_size'])
        self.sort_flag = False

    def evolve(self, epoch):
        a = 2 - 2.0 * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


# MealPy kütüphanesinden: OriginalHHO
class HHO(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.set_parameters(['epoch', 'pop_size'])
        self.sort_flag = False

    def evolve(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            E0 = 2 * self.generator.uniform() - 1
            E = 2 * E0 * (1.0 - epoch * 1.0 / self.epoch)
            J = 2 * (1 - self.generator.uniform())
            if np.abs(E) >= 1:
                if self.generator.random() >= 0.5:
                    X_rand = self.pop[self.generator.integers(0, self.pop_size)].solution.copy()
                    pos_new = X_rand - self.generator.uniform() * np.abs(X_rand - 2 * self.generator.uniform() * self.pop[idx].solution)
                else:
                    X_m = np.mean([x.solution for x in self.pop])
                    pos_new = self.g_best.solution - X_m - self.generator.uniform() * (self.problem.lb + self.generator.uniform() * (self.problem.ub - self.problem.lb))
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
            elif self.generator.random() >= 0.5:
                delta_X = self.g_best.solution - self.pop[idx].solution
                if np.abs(E) >= 0.5:
                    pos_new = delta_X - E * np.abs(J * self.g_best.solution - self.pop[idx].solution)
                else:
                    pos_new = self.g_best.solution - E * np.abs(delta_X)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
            else:
                LF_D = self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)
                if np.abs(E) >= 0.5:
                    Y = self.g_best.solution - E * np.abs(J * self.g_best.solution - self.pop[idx].solution)
                else:
                    X_m = np.mean([x.solution for x in self.pop])
                    Y = self.g_best.solution - E * np.abs(J * self.g_best.solution - X_m)
                pos_Y = self.correct_solution(Y)
                target_Y = self.get_target(pos_Y)
                Z = Y + self.generator.uniform(self.problem.lb, self.problem.ub) * LF_D
                pos_Z = self.correct_solution(Z)
                target_Z = self.get_target(pos_Z)
                if self.compare_target(target_Y, self.pop[idx].target, self.problem.minmax):
                    agent = self.generate_empty_agent(pos_Y)
                    agent.target = target_Y
                    pop_new.append(agent)
                    continue
                if self.compare_target(target_Z, self.pop[idx].target, self.problem.minmax):
                    agent = self.generate_empty_agent(pos_Z)
                    agent.target = target_Z
                    pop_new.append(agent)
                    continue
                pop_new.append(self.pop[idx].copy())
        if self.mode not in self.AVAILABLE_MODES:
            for idx, agent in enumerate(pop_new):
                pop_new[idx].target = self.get_target(agent.solution)
        else:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


# MealPy kütüphanesinden: OriginalSSO
class SSO(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.set_parameters(['epoch', 'pop_size'])
        self.sort_flag = True

    def evolve(self, epoch):
        c1 = 2 * np.exp(-(4 * epoch / self.epoch) ** 2)
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx < self.pop_size / 2:
                c2_list = self.generator.random(self.problem.n_dims)
                c3_list = self.generator.random(self.problem.n_dims)
                pos_new_1 = self.g_best.solution + c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
                pos_new_2 = self.g_best.solution - c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
                pos_new = np.where(c3_list < 0.5, pos_new_1, pos_new_2)
            else:
                pos_new = (self.pop[idx].solution + self.pop[idx - 1].solution) / 2
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


# MealPy kütüphanesinden: L_SHADE
class L_SHADE(Optimizer):

    def __init__(self, epoch: int=750, pop_size: int=100, miu_f: float=0.5, miu_cr: float=0.5, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.miu_f = self.validator.check_float('miu_f', miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float('miu_cr', miu_cr, (0, 1.0))
        self.set_parameters(['epoch', 'pop_size', 'miu_f', 'miu_cr'])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_miu_f = self.miu_f * np.ones(self.pop_size)
        self.dyn_miu_cr = self.miu_cr * np.ones(self.pop_size)
        self.dyn_pop_archive = list()
        self.dyn_pop_size = self.pop_size
        self.k_counter = 0
        self.n_min = int(self.pop_size / 5)

    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = np.sum(list_weights * list_objects ** 2)
        down = np.sum(list_weights * list_objects)
        return up / down if down != 0 else 0.5

    def evolve(self, epoch):
        list_f = list()
        list_cr = list()
        list_f_index = list()
        list_cr_index = list()
        list_f_new = np.ones(self.pop_size)
        list_cr_new = np.ones(self.pop_size)
        pop_old = [agent.copy() for agent in self.pop]
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            idx_rand = self.generator.integers(0, self.pop_size)
            cr = self.generator.normal(self.dyn_miu_cr[idx_rand], 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f[idx_rand], 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            list_cr_new[idx] = cr
            list_f_new[idx] = f
            p = self.generator.uniform(0.15, 0.2)
            top = int(np.ceil(self.dyn_pop_size * p))
            x_best = pop_sorted[self.generator.integers(0, top)]
            r1_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            new_pop = self.pop + self.dyn_pop_archive
            r2_idx = self.generator.choice(list(set(range(0, len(new_pop))) - {idx, r1_idx}))
            x_r1 = self.pop[r1_idx].solution
            x_r2 = new_pop[r2_idx].solution
            x_new = self.pop[idx].solution + f * (x_best.solution - self.pop[idx].solution) + f * (x_r1 - x_r2)
            pos_new = np.where(self.generator.random(self.problem.n_dims) < cr, x_new, self.pop[idx].solution)
            j_rand = self.generator.integers(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].target = self.get_target(pos_new)
        pop = self.update_target_for_population(pop)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop[idx].target, self.pop[idx].target, self.problem.minmax):
                list_cr.append(list_cr_new[idx])
                list_f.append(list_f_new[idx])
                list_f_index.append(idx)
                list_cr_index.append(idx)
                self.pop[idx] = pop[idx].copy()
                self.dyn_pop_archive.append(self.pop[idx].copy())
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = self.generator.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, agent in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(agent.copy())
            self.dyn_pop_archive = archive_pop_new
        if len(list_f) != 0 and len(list_cr) != 0:
            list_fit_old = np.ones(len(list_cr_index))
            list_fit_new = np.ones(len(list_cr_index))
            idx_increase = 0
            for idx in range(0, self.dyn_pop_size):
                if idx in list_cr_index:
                    list_fit_old[idx_increase] = pop_old[idx].target.fitness
                    list_fit_new[idx_increase] = self.pop[idx].target.fitness
                    idx_increase += 1
            total_fit = np.sum(np.abs(list_fit_new - list_fit_old))
            list_weights = 0 if total_fit == 0 else np.abs(list_fit_new - list_fit_old) / total_fit
            self.dyn_miu_cr[self.k_counter] = np.sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean(np.array(list_f), list_weights)
            self.k_counter += 1
            if self.k_counter >= self.dyn_pop_size:
                self.k_counter = 0
        self.dyn_pop_size = round(self.pop_size + epoch * ((self.n_min - self.pop_size) / self.epoch))


# MealPy kütüphanesinden: CMA_ES
class CMA_ES(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.set_parameters(['epoch', 'pop_size'])
        self.sort_flag = True

    def generate_empty_agent(self, solution: np.ndarray=None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        step = self.generator.multivariate_normal(np.zeros(self.problem.n_dims), np.eye(self.problem.n_dims))
        return Agent(solution=solution, step=step)

    def before_main_loop(self):
        self.mu = int(np.round(self.pop_size / 2))
        self.ps = np.zeros(self.problem.n_dims)
        self.C = np.eye(self.problem.n_dims)
        self.pc = np.zeros(self.problem.n_dims)
        self.w = np.log(self.pop_size + 0.5) - np.log(np.arange(1, self.pop_size + 1))
        self.w = self.w / np.sum(self.w)
        self.mu_eff = 1.0 / np.sum(self.w ** 2)
        sigma0 = 0.1 * (self.problem.ub - self.problem.lb)
        self.cs = (self.mu_eff + 2) / (self.problem.n_dims + self.mu_eff + 5)
        self.ds = 1 + self.cs + 2 * np.max(np.sqrt((self.mu_eff - 1.0) / (self.problem.n_dims + 1)) - 1, 0)
        self.ENN = np.sqrt(self.problem.n_dims) * (1 - 1.0 / (4 * self.problem.n_dims) + 1.0 / (21 * self.problem.n_dims ** 2))
        self.cc = (4 + self.mu_eff / self.problem.n_dims) / (4 + self.problem.n_dims + 2 * self.mu_eff / self.problem.n_dims)
        self.c1 = 2.0 / ((self.problem.n_dims + 1.3) ** 2 + self.mu_eff)
        alpha_mu = 2
        self.cmu = min(1 - self.c1, alpha_mu * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.problem.n_dims + 2) ** 2 + alpha_mu * self.mu_eff / 2))
        self.hth = (1.4 + 2 / (self.problem.n_dims + 1)) * self.ENN
        self.sigma = sigma0
        self.x_mean = np.mean([agent.solution for agent in self.pop[:self.mu]], axis=0)

    def update_step__(self, pop, cc):
        for idx in range(0, self.pop_size):
            pop[idx].step = self.generator.multivariate_normal(np.zeros(self.problem.n_dims), cc)
        return pop

    def evolve(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.x_mean + self.sigma * self.pop[idx].step
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_population(pop_new, self.problem.minmax)
        self.pop = self.update_step__(self.pop, self.C)
        self.x_step = np.zeros(self.problem.n_dims)
        for idx in range(0, self.mu):
            self.x_step += self.w[idx] * self.pop[idx].step
        self.x_mean = self.x_mean + self.sigma * self.x_step
        t11 = np.dot(self.x_step, np.linalg.inv(np.linalg.cholesky(self.C).T))
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * t11
        self.sigma = self.sigma * np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.ENN - 1)) ** 0.3
        if np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * epoch)) < self.hth:
            hs = 1
        else:
            hs = 0
        delta = (1 - hs) * self.cc * (2 - self.cc)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * self.x_step
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc) + delta * self.C
        for idx in range(0, self.mu):
            self.C = self.C + self.cmu * self.w[idx] * np.outer(self.pop[idx].step, self.pop[idx].step)
        E, V = np.linalg.eig(self.C)
        E = np.diag(E)
        if np.any(np.diag(E) < 0):
            E[E < 0] = 0
            self.C = V * E / V


# MealPy kütüphanesinden: OriginalIMODE
class IMODE(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, memory_size: int=5, archive_size: int=20, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.memory_size = self.validator.check_int('memory_size', memory_size, [2, 100])
        self.archive_size = self.validator.check_int('archive_size', archive_size, [5, 100])
        self.set_parameters(['epoch', 'pop_size', 'memory_size', 'archive_size'])
        self.sort_flag = True
        self.is_parallelizable = False

    def initialize_variables(self):
        self.operator_probs = np.ones(3) / 3
        self.memory_pos = 0
        self.memory_f = np.full(self.memory_size, 0.5)
        self.memory_cr = np.full(self.memory_size, 0.5)
        self.archive_size = max(self.archive_size, self.pop_size)

    def before_main_loop(self):
        self.archive = self.pop.copy()

    def _generate_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        mem_indices = self.generator.integers(0, self.memory_size, self.pop_size)
        mu_f = self.memory_f[mem_indices]
        mu_cr = self.memory_cr[mem_indices]
        cr = self.generator.normal(mu_cr, 0.1)
        cr[mu_cr == -1] = 0
        cr = np.clip(cr, 0, 1)
        f = mu_f + 0.1 * np.tan(np.pi * (self.generator.random(self.pop_size) - 0.5))
        negative_mask = f <= 0
        while np.any(negative_mask):
            f[negative_mask] = mu_f[negative_mask] + 0.1 * np.tan(np.pi * (self.generator.random(np.sum(negative_mask)) - 0.5))
            negative_mask = f <= 0
        f = np.clip(f, 0, 1)
        return (f, cr)

    def _select_operator_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rand_vals = self.generator.random(self.pop_size)
        prob_cumsum = np.cumsum(self.operator_probs)
        op1_mask = rand_vals <= prob_cumsum[0]
        op2_mask = (rand_vals > prob_cumsum[0]) & (rand_vals <= prob_cumsum[1])
        op3_mask = rand_vals > prob_cumsum[1]
        return (op1_mask, op2_mask, op3_mask)

    def _generate_random_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        combined_pop = self.pop + self.archive
        total_size = len(combined_pop)
        r1, r2, r3 = (np.zeros(self.pop_size, dtype=int), np.zeros(self.pop_size, dtype=int), np.zeros(self.pop_size, dtype=int))
        for idx in range(0, self.pop_size):
            x1, x3 = self.generator.choice(list(set(range(self.pop_size)) - {idx}), size=2, replace=False)
            x2 = self.generator.choice(list(set(range(total_size)) - {idx, x1, x3}))
            r1[idx] = x1
            r2[idx] = x2
            r3[idx] = x3
        return (r1, r2, r3, combined_pop)

    def _mutation(self, f: np.ndarray) -> np.ndarray:
        op1_mask, op2_mask, op3_mask = self._select_operator_indices()
        r1, r2, r3, combined_pop = self._generate_random_indices()
        matrix_pos = np.array([agent.solution for agent in self.pop])
        matrix_combined = np.array([agent.solution for agent in combined_pop])
        matrix_mutant = np.zeros_like(matrix_pos)
        if np.any(op1_mask):
            p_best_size = max(int(0.25 * self.pop_size), 1)
            pbest_indices = self.generator.integers(0, p_best_size, self.pop_size)
            matrix_pbest = matrix_pos[pbest_indices]
            matrix_mutant[op1_mask] = matrix_pos[op1_mask] + f[op1_mask, np.newaxis] * (matrix_pbest[op1_mask] - matrix_pos[op1_mask] + matrix_pos[r1[op1_mask]] - matrix_combined[r2[op1_mask]])
        if np.any(op2_mask):
            p_best_size = max(int(0.25 * self.pop_size), 1)
            pbest_indices = self.generator.integers(0, p_best_size, self.pop_size)
            matrix_pbest = matrix_pos[pbest_indices]
            matrix_mutant[op2_mask] = matrix_pos[op2_mask] + f[op2_mask, np.newaxis] * (matrix_pbest[op2_mask] - matrix_pos[op2_mask] + matrix_pos[r1[op2_mask]] - matrix_pos[r3[op2_mask]])
        if np.any(op3_mask):
            p_best_size = max(int(0.5 * self.pop_size), 2)
            pbest_indices = self.generator.integers(0, p_best_size, self.pop_size)
            matrix_pbest = matrix_pos[pbest_indices]
            matrix_mutant[op3_mask] = f[op3_mask, np.newaxis] * matrix_pos[r1[op3_mask]] + f[op3_mask, np.newaxis] * (matrix_pbest[op3_mask] - matrix_pos[r3[op3_mask]])
        return matrix_mutant

    def _handle_boundaries(self, vectors: np.ndarray) -> np.ndarray:
        strategy = self.generator.integers(1, 4)
        result = []
        if strategy == 1:
            for idx in range(0, len(vectors)):
                res = np.select([vectors[idx] < self.problem.lb, vectors[idx] > self.problem.ub], [(vectors[idx] + self.problem.ub) / 2, (vectors[idx] + self.problem.lb) / 2], default=vectors[idx])
                result.append(res)
        elif strategy == 2:
            for idx in range(0, len(vectors)):
                res = vectors[idx]
                flag1 = res < self.problem.lb
                res[flag1] = np.clip(2 * self.problem.lb[flag1] - res[flag1], self.problem.lb[flag1], self.problem.ub[flag1])
                flag2 = res > self.problem.ub
                res[flag2] = np.clip(2 * self.problem.ub[flag2] - res[flag2], self.problem.lb[flag2], self.problem.ub[flag2])
                result.append(res)
        else:
            for idx in range(0, len(vectors)):
                res = vectors[idx]
                mask_lower = res < self.problem.lb
                mask_upper = res > self.problem.ub
                res[mask_lower | mask_upper] = self.generator.uniform(self.problem.lb[mask_lower | mask_upper], self.problem.ub[mask_lower | mask_upper])
                result.append(res)
        results = np.clip(result, self.problem.lb, self.problem.ub)
        return results

    def _crossover(self, mutant: np.ndarray, cr: np.ndarray) -> np.ndarray:
        matrix_pos = np.array([agent.solution for agent in self.pop])
        if self.generator.random() < 0.4:
            cross_mask = self.generator.random((self.pop_size, self.problem.n_dims)) <= cr[:, np.newaxis]
            for idx in range(self.pop_size):
                if not np.any(cross_mask[idx]):
                    cross_mask[idx, self.generator.integers(0, self.problem.n_dims)] = True
            trial = matrix_pos.copy()
            trial[cross_mask] = mutant[cross_mask]
        else:
            trial = matrix_pos.copy()
            start_points = self.generator.integers(0, self.problem.n_dims, self.pop_size)
            for idx in range(self.pop_size):
                jdx = start_points[idx]
                while self.generator.random() < cr[idx] and jdx < self.problem.n_dims:
                    trial[idx, jdx] = mutant[idx, jdx]
                    jdx += 1
        return trial

    def _update_archive(self, improved_pop=None):
        if len(improved_pop) == 0:
            return
        if len(self.archive) == 0:
            self.archive = improved_pop
        else:
            self.archive = self.archive + improved_pop
        if len(self.archive) > 1:
            self.archive = list(set(self.archive))
            if len(self.archive) > self.archive_size:
                remove_count = len(self.archive) - self.archive_size
                remove_indices = self.generator.choice(len(self.archive), remove_count, replace=False)
                keep_indices = list(set(range(len(self.archive))) - set(remove_indices))
                self.archive = [self.archive[idx] for idx in keep_indices]

    def evolve(self, epoch):
        f_values, cr_values = self._generate_parameters()
        self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
        cr_values = np.sort(cr_values)
        matrix_mutant = self._mutation(f_values)
        matrix_mutant = self._handle_boundaries(matrix_mutant)
        matrix_child = self._crossover(matrix_mutant, cr_values)
        improvement_mask = np.zeros(self.pop_size, dtype=bool)
        improvements = np.zeros(self.pop_size)
        pop_new = []
        for idx in range(len(matrix_child)):
            pos_new = self.correct_solution(matrix_child[idx])
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target):
                improvement_mask[idx] = True
            improvements[idx] = np.abs(self.pop[idx].target.fitness - agent.target.fitness)
            pop_new.append(agent)
        op1_mask, op2_mask, op3_mask = self._select_operator_indices()
        fits = np.array([agent.target.fitness for agent in self.pop])
        fits_child = np.array([agent.target.fitness for agent in pop_new])
        relative_improvements = np.maximum(0, (fits - fits_child) / np.abs(fits))
        if np.any(improvement_mask):
            self._update_archive([pop_new[idx] for idx in range(self.pop_size) if improvement_mask[idx]])
        if np.any(improvement_mask):
            successful_f = f_values[improvement_mask]
            successful_cr = cr_values[improvement_mask]
            successful_improvements = improvements[improvement_mask]
            if len(successful_f) > 0:
                weights = successful_improvements / np.sum(successful_improvements)
                self.memory_f[self.memory_pos] = np.sum(weights * successful_f ** 2) / np.sum(weights * successful_f)
                if np.max(successful_cr) == 0:
                    self.memory_cr[self.memory_pos] = -1
                else:
                    self.memory_cr[self.memory_pos] = np.sum(weights * successful_cr ** 2) / np.sum(weights * successful_cr)
                self.memory_pos = (self.memory_pos + 1) % self.memory_size
            else:
                self.memory_f[self.memory_pos] = 0.5
                self.memory_cr[self.memory_pos] = 0.5
        op1_improvement = np.mean(relative_improvements[op1_mask]) if np.any(op1_mask) else 0
        op2_improvement = np.mean(relative_improvements[op2_mask]) if np.any(op2_mask) else 0
        op3_improvement = np.mean(relative_improvements[op3_mask]) if np.any(op3_mask) else 0
        total_improvement = op1_improvement + op2_improvement + op3_improvement
        if total_improvement > 0:
            self.operator_probs = np.array([op1_improvement, op2_improvement, op3_improvement])
            self.operator_probs = np.clip(self.operator_probs / total_improvement, 0.1, 0.9)
            self.operator_probs = self.operator_probs / np.sum(self.operator_probs)
        else:
            self.operator_probs = np.ones(3) / 3
        self.pop = [a if flag else b for a, b, flag in zip(pop_new, self.pop, improvement_mask)]


# MealPy kütüphanesinden: OriginalLSHADEcnEpSin
class LSHADEcnEpSin(Optimizer):

    def __init__(self, epoch: int=10000, pop_size: int=100, miu_f: float=0.5, miu_cr: float=0.5, freq: float=0.5, memory_size: int=5, ps: float=0.5, pc: float=0.4, pop_size_min: int=10, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [5, 10000])
        self.miu_f = self.validator.check_float('miu_f', miu_f, (0.1, 1.0))
        self.miu_cr = self.validator.check_float('miu_cr', miu_cr, (0.1, 1.0))
        self.freq = self.validator.check_float('freq', freq, (0.1, 2.0))
        self.memory_size = self.validator.check_int('memory_size', memory_size, [1, 100])
        self.ps = self.validator.check_float('ps', ps, (0.1, 1.0))
        self.pc = self.validator.check_float('pc', pc, (0.1, 1.0))
        self.pop_size_min = self.validator.check_int('pop_size_min', pop_size_min, [4, 1000])
        self.set_parameters(['epoch', 'pop_size', 'miu_f', 'miu_cr', 'freq', 'memory_size', 'ps', 'pc', 'pop_size_min'])
        self.sort_flag = False

    def initialize_variables(self):
        self.NP_init = self.pop_size if self.pop_size else 18 * self.problem.n_dims
        self.NP_min = self.pop_size_min
        self.NP = self.NP_init
        self.H = self.memory_size
        self.M_F = np.full(self.H, self.miu_f)
        self.M_CR = np.full(self.H, self.miu_cr)
        self.M_freq = np.full(self.H, self.freq)
        self.memory_index = 0
        self.LP = 10
        self.freq_fixed = self.freq
        self.epsilon = 0.01
        self.ns1_history = []
        self.ns2_history = []
        self.nf1_history = []
        self.nf2_history = []

    def before_main_loop(self):
        self.archive = self.pop.copy()

    def update_sinusoidal_probabilities(self, epoch):
        if epoch <= self.LP:
            return (0.5, 0.5)
        start_idx = max(0, epoch - self.LP)
        S1 = S2 = self.epsilon
        if len(self.ns1_history) > start_idx:
            ns1_sum = sum(self.ns1_history[start_idx:epoch])
            nf1_sum = sum(self.nf1_history[start_idx:epoch])
            S1 = (ns1_sum + self.epsilon) / (ns1_sum + nf1_sum + 2 * self.epsilon)
        if len(self.ns2_history) > start_idx:
            ns2_sum = sum(self.ns2_history[start_idx:epoch])
            nf2_sum = sum(self.nf2_history[start_idx:epoch])
            S2 = (ns2_sum + self.epsilon) / (ns2_sum + nf2_sum + 2 * self.epsilon)
        total_S = S1 + S2
        p1 = S1 / total_S
        p2 = S2 / total_S
        return (p1, p2)

    def sinusoidal_adaptation(self, epoch, max_epoch, config_type, freq=None):
        if config_type == 1:
            F = 0.5 * np.sin(2 * np.pi * self.freq_fixed * (max_epoch - epoch) / max_epoch) + 0.5
        else:
            if freq is None:
                freq = self.freq_fixed
            F = 0.5 * np.sin(2 * np.pi * freq * epoch / max_epoch) + 0.5
        return max(0.1, min(1.0, F))

    def current_to_pbest_mutation(self, idx, F, p=0.1):
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        p_size = max(1, int(p * self.NP))
        pbest_idx = self.generator.choice(range(p_size))
        r1 = self.generator.choice(list(set(range(self.NP)) - {idx}))
        pop_combined = self.pop + self.archive
        r2 = self.generator.choice(list(set(range(len(pop_combined))) - {idx, r1}))
        pos_new = self.pop[idx].solution + F * (pop_sorted[pbest_idx].solution - self.pop[idx].solution) + F * (self.pop[r1].solution - pop_combined[r2].solution)
        pos_new = self.correct_solution(pos_new)
        return pos_new

    def binomial_crossover(self, target, mutant, CR=None):
        if CR is None:
            r_idx = self.generator.integers(0, self.H)
            CR = norm.rvs(loc=self.M_CR[r_idx], scale=0.1)
            CR = np.clip(CR, 0, 1)
        trial = np.where(self.generator.uniform(0, 1, self.problem.n_dims) <= CR, mutant, target)
        j_rand = self.generator.integers(0, self.problem.n_dims)
        trial[j_rand] = mutant[j_rand]
        return trial

    def covariance_matrix_crossover(self, target, mutant):
        list_pos = np.array([agent.solution for agent in self.pop])
        dist = np.linalg.norm(list_pos - self.g_best.solution, axis=1)
        dist_indices = np.argsort(dist)
        neighborhood_size = max(2, int(self.ps * self.NP))
        neighborhood_indices = dist_indices[:neighborhood_size]
        neighborhood = list_pos[neighborhood_indices]
        if neighborhood.shape[0] > 1:
            cov_matrix = np.cov(neighborhood.T)
            cov_matrix += np.eye(self.problem.n_dims) * 1e-08
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                B = eigenvectors
                B_T = B.T
                target_prime = B_T @ target
                mutant_prime = B_T @ mutant
                r_idx = self.generator.integers(0, self.H)
                CR = norm.rvs(loc=self.M_CR[r_idx], scale=0.1)
                CR = np.clip(CR, 0, 1)
                trial_prime = np.where(self.generator.uniform(0, 1, self.problem.n_dims) <= CR, mutant_prime, target_prime)
                j_rand = self.generator.integers(0, self.problem.n_dims)
                trial_prime[j_rand] = mutant_prime[j_rand]
                trial = B @ trial_prime
            except np.linalg.LinAlgError:
                trial = self.binomial_crossover(target, mutant)
        else:
            trial = self.binomial_crossover(target, mutant)
        return trial

    def weighted_lehmer_mean(self, S_values, delta_f):
        if len(S_values) == 0:
            return 0.5
        S_values = np.array(S_values)
        delta_f = np.array(delta_f)
        delta_f = np.maximum(delta_f, 1e-10)
        weights = delta_f / np.sum(delta_f)
        numerator = np.sum(weights * S_values ** 2)
        denominator = np.sum(weights * S_values)
        if denominator == 0:
            return 0.5
        return numerator / denominator

    def linear_population_reduction(self, epoch, max_epoch):
        new_NP = int(self.NP_min + (self.NP_init - self.NP_min) * (max_epoch - epoch) / max_epoch)
        new_NP = max(self.NP_min, new_NP)
        if new_NP < self.NP:
            _, indices = self.get_sorted_population(self.pop, self.problem.minmax, return_index=True)
            tt = indices[:new_NP]
            self.generator.shuffle(tt)
            self.pop = [self.pop[idx] for idx in tt]
        self.NP = new_NP

    def evolve(self, epoch):
        S_F = []
        S_CR = []
        delta_f = []
        ns1_current = ns2_current = 0
        nf1_current = nf2_current = 0
        for idx in range(0, self.NP):
            if epoch <= self.epoch // 2:
                p1, p2 = self.update_sinusoidal_probabilities(epoch)
                if self.generator.random() < p1:
                    F = self.sinusoidal_adaptation(epoch, self.epoch, config_type=1)
                    config_used = 1
                else:
                    r_idx = self.generator.integers(0, self.H)
                    freq = cauchy.rvs(loc=self.M_freq[r_idx], scale=0.1)
                    freq = np.clip(freq, 0.1, 2.0)
                    F = self.sinusoidal_adaptation(epoch, self.epoch, config_type=2, freq=freq)
                    config_used = 2
            else:
                r_idx = self.generator.integers(0, self.H)
                F = cauchy.rvs(loc=self.M_F[r_idx], scale=0.1)
                F = np.clip(F, 0.1, 1.0)
                config_used = 0
            r_idx = self.generator.integers(0, self.H)
            CR = norm.rvs(loc=self.M_CR[r_idx], scale=0.1)
            CR = np.clip(CR, 0, 1)
            pos_new = self.current_to_pbest_mutation(idx, F)
            if self.generator.random() < self.pc:
                pos_new = self.covariance_matrix_crossover(self.pop[idx].solution, pos_new)
            else:
                pos_new = self.binomial_crossover(self.pop[idx].solution, pos_new, CR)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                delta = abs(self.pop[idx].target.fitness - agent.target.fitness)
                S_F.append(F)
                S_CR.append(CR)
                delta_f.append(delta)
                if epoch <= self.epoch // 2:
                    if config_used == 1:
                        ns1_current += 1
                    elif config_used == 2:
                        ns2_current += 1
                self.archive = self.archive + [self.pop[idx].copy()]
                self.pop[idx] = agent
            elif epoch <= self.epoch // 2:
                if config_used == 1:
                    nf1_current += 1
                elif config_used == 2:
                    nf2_current += 1
        self.ns1_history.append(ns1_current)
        self.ns2_history.append(ns2_current)
        self.nf1_history.append(nf1_current)
        self.nf2_history.append(nf2_current)
        if len(S_F) > 0:
            if len(S_F) > 0:
                self.M_F[self.memory_index] = self.weighted_lehmer_mean(S_F, delta_f)
            if len(S_CR) > 0:
                self.M_CR[self.memory_index] = self.weighted_lehmer_mean(S_CR, delta_f)
            self.memory_index = (self.memory_index + 1) % self.H
        self.linear_population_reduction(epoch, self.epoch)
        if len(self.archive) > self.NP:
            remove_count = len(self.archive) - self.NP
            remove_indices = self.generator.choice(len(self.archive), remove_count, replace=False)
            keep_indices = list(set(range(len(self.archive))) - set(remove_indices))
            self.archive = [self.archive[idx] for idx in keep_indices]

# ------------------------------------------------------------------------------------------------------- # 


class _NHO_AblationBase(NHO):

    use_dopamine = True
    use_serotonin = True
    use_cortisol = True
    use_homeostasis = True

    def initialize_variables(self):
        super().initialize_variables()
        if not self.use_dopamine:
            self.DA = 0.0
        if not self.use_serotonin:
            self.ST = 0.0
        if not self.use_cortisol:
            self.CORT = 0.0

    def evolve(self, epoch: int) -> None:
        N, D = (self.pop_size, self.problem.n_dims)
        lb, ub = self._get_lb_ub()
        rng = self.generator
        X = np.stack([ag.solution for ag in self.pop], axis=0)
        f = np.array([ag.target.fitness for ag in self.pop], float)
        order = np.argsort(f) if self._is_min() else np.argsort(-f)
        K = max(1, int(np.ceil(self.p_elite * N)))
        elite_idx = order[:K]
        m = np.mean(X[elite_idx], axis=0)
        gbest = self.pop[order[0]]
        xbest = gbest.solution.copy()
        tau = (epoch + 1) / max(1, self.epoch)

        DA = self.DA if self.use_dopamine else 0.0
        ST = self.ST if self.use_serotonin else 0.0
        CORT = self.CORT if self.use_cortisol else 0.0

        if not self.use_homeostasis:
            DA = 0.5
            ST = 0.5
            CORT = 0.0

        Ft = float(np.clip(self.F0 * np.exp(self.kD * DA - self.kS * ST), self.F_min, self.F_max))
        CRt = float(self._sigmoid(self.b0 + self.b1 * ST - self.b2 * DA))
        alphat = self.alpha0 * (1 - tau) * (self.cD * DA + self.cS * (1 - ST))
        deltat = self.delta0 * (1 - tau) * np.exp(self.kC * CORT - self.kS2 * ST)
        improved = 0

        for i in range(N):
            idx = list(range(N))
            idx.remove(i)
            r1, r2 = rng.choice(idx, size=2, replace=False)
            d = X[r1] - X[r2]
            v1 = X[i] + Ft * d + (m - X[i]) * (0.8 * (1 - tau))
            u = m - X[i]
            un = np.linalg.norm(u)
            o = d - np.dot(d, u / un) * (u / un) if un > 1e-12 else d
            v2 = X[i] + Ft * o
            gauss = rng.normal(0.0, 1.0, size=D) * deltat * (ub - lb)
            v3 = X[i] + alphat * (xbest - X[i]) + gauss

            def mix_fix(cand):
                U = rng.random(D)
                y = np.where(U < CRt, cand, X[i])
                y = np.asarray(y, float)
                y = np.nan_to_num(y, copy=False)
                return self.correct_solution(y)

            y1 = mix_fix(v1)
            y2 = mix_fix(v2)
            y3 = mix_fix(v3)
            a1 = self.generate_empty_agent(y1)
            a1.target = self.get_target(y1)
            a2 = self.generate_empty_agent(y2)
            a2.target = self.get_target(y2)
            a3 = self.generate_empty_agent(y3)
            a3.target = self.get_target(y3)
            best_cand = a1
            if self._better(a2.target.fitness, best_cand.target.fitness):
                best_cand = a2
            if self._better(a3.target.fitness, best_cand.target.fitness):
                best_cand = a3
            if self._better(best_cand.target.fitness, f[i]):
                self.pop[i] = best_cand
                X[i] = best_cand.solution.copy()
                f[i] = best_cand.target.fitness
                improved += 1
                if self._better(f[i], gbest.target.fitness):
                    gbest = best_cand
                    xbest = best_cand.solution.copy()

        span = ub - lb + 1e-12
        div = float(np.mean(np.std(X, axis=0) / span))
        if div < self.div_threshold:
            q = max(1, int(self.restart_frac * N))
            worst_idx = order[-q:]
            radius = 0.1 * span * (1 - tau)
            for j in worst_idx:
                newx = xbest + rng.uniform(-1, 1, size=D) * radius
                newx = self.correct_solution(newx)
                ag = self.generate_empty_agent(newx)
                ag.target = self.get_target(newx)
                if self._better(ag.target.fitness, f[j]):
                    self.pop[j] = ag
                    X[j] = ag.solution.copy()
                    f[j] = ag.target.fitness

        curr_best = self.pop[order[0]].target.fitness if order.size > 0 else gbest.target.fitness
        if self._prev_best_fit is None:
            rpe = 0.0
        else:
            num = self._prev_best_fit - curr_best
            den = abs(self._prev_best_fit) + 1e-12
            rpe = max(0.0, float(num / den))
        self._prev_best_fit = curr_best

        if self.use_homeostasis:
            if self.use_dopamine:
                self.DA = (1 - self.lambda_D) * self.DA + self.lambda_D * rpe
            else:
                self.DA = 0.0

            if self.use_serotonin:
                self.ST = (1 - self.lambda_S) * self.ST + self.lambda_S * (1 - rpe)
            else:
                self.ST = 0.0

            if self.use_cortisol:
                self.CORT = (1 - self.lambda_C) * self.CORT + self.lambda_C * (1 - div)
            else:
                self.CORT = 0.0
        else:
            self.DA = 0.5
            self.ST = 0.5
            self.CORT = 0.0

        _, bests, _ = self.get_special_agents(self.pop, n_best=1, minmax=self.problem.minmax)
        b = bests[0]
        self.convergence_curve.append(b.target.fitness)
        self.best_fitness_curve.append(b.target.fitness)
        self.best_solution_curve.append(b.solution.copy())
        if self.log_hormones:
            self.DA_history.append(float(self.DA))
            self.ST_history.append(float(self.ST))
            self.CORT_history.append(float(self.CORT))
            self.succ_history.append(float(improved / max(1, N)))
            self.div_history.append(div)
            self.F_history.append(Ft)
            self.CR_history.append(CRt)
            self.alpha_history.append(alphat)
            self.delta_history.append(deltat)


class NHO_NoDopamine(_NHO_AblationBase):
    use_dopamine = False
    use_serotonin = True
    use_cortisol = True
    use_homeostasis = True


class NHO_NoSerotonin(_NHO_AblationBase):
    use_dopamine = True
    use_serotonin = False
    use_cortisol = True
    use_homeostasis = True


class NHO_NoCortisol(_NHO_AblationBase):
    use_dopamine = True
    use_serotonin = True
    use_cortisol = False
    use_homeostasis = True


class NHO_NoHomeostasis(_NHO_AblationBase):
    use_dopamine = True
    use_serotonin = True
    use_cortisol = True
    use_homeostasis = False

