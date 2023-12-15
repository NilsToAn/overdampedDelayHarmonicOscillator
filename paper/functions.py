import pickle
from timeit import default_timer as timer
import numpy as np
import scipy
from typing import Callable, Optional
from scipy.special import factorial
import json
from pathlib import Path
import datetime

from algorithm import get_prop_v1_1, get_prop_v2_1, create_R_v3, get_dyn_v2


#  Nummerical


def get_p_x4_short_time(x: np.ndarray, tau: float) -> np.ndarray:
    """Probabilty function derived with
    small delay approximation for cubic time delay
    potential.

    Parameters
    ----------
    x : np.ndarray
        The position(s) to evaluate the function
    tau : float
        Time delay

    Returns
    -------
    np.ndarray
        The values of the function
    """
    if tau > 0:
        res = (
            1
            / (1 + 3 * tau * x**2) ** 2
            * np.exp(
                -((3 * tau * x**2 - np.log(3 * tau * x**2 + 1)) / (9 * tau**2))
            )
        )
    else:
        res = np.exp(-(x**4) / 2)
    return res / res.sum()


def get_lin_var(tau, a, b, s):
    D = 1 / 2 * s**2
    if b > a:
        return (
            D
            * (
                1
                + np.sin(np.sqrt(b**2 - a**2) * tau) / np.sqrt(1 - a**2 / b**2)
            )
            * 1
            / (a + b * np.cos(np.sqrt(b**2 - a**2) * tau))
        )
    elif a == b:
        return D * (1 + b * tau) * 1 / (2 * b)
    elif b < a:
        return (
            D
            * (
                1
                + np.sinh(np.sqrt(a**2 - b**2) * tau) / np.sqrt(a**2 / b**2 - 1)
            )
            * 1
            / (a + b * np.cosh(np.sqrt(a**2 - b**2) * tau))
        )


get_lin_var = np.vectorize(get_lin_var, excluded=(1, 2, 3))


def get_lin_var_short_time(tau, a, b, s):
    D = 1 / 2 * s**2
    return D / (a + b) * (1 + b * tau)


# Theory
def l_two_time(a: float, b: float, tau: float, t: float, max_p: int = 40):
    i = np.arange(0, max_p, 1)
    return np.sum(
        (-b) ** i
        / factorial(i)
        * (t - i * tau) ** i
        * np.exp(-a * (t - i * tau))
        * np.heaviside(t - i * tau, 1)
    )


l_two_time = np.vectorize(l_two_time)


def get_theo_var_l_two_time(ts, tau, D, a=1, b=1):
    if tau > 0:
        dt = ts[1] - ts[0]
        max_p = np.max(ts) / tau
        l_data = l_two_time(a, b, tau, ts, max_p)
        var = np.zeros(len(ts))
        var[1:] = (2 * D * np.cumsum(l_data**2) * dt)[:-1]
        return var
    else:
        k = a + b
        return D / k * (1 - np.exp(-2 * k * ts))


def get_eq_times(tau, D, eq_perc, a, b):
    database_path = Path.cwd() / "database/eq_times.json"
    if database_path.is_file():
        known_eq_times = json.load(open(database_path))
    else:
        known_eq_times = []
    eq_time = [
        o["eq_time"]
        for o in known_eq_times
        if o["params"] == {"tau": tau, "D": D, "eq_perc": eq_perc, "a": a, "b": b}
    ]
    if len(eq_time):
        return eq_time[0]

    s = np.sqrt(2 * D)
    if tau < 1:
        test_ts = np.linspace(0, 5, 16000)
    else:
        test_ts = np.linspace(0, 40, 16000)

    arg_min = np.argmin(
        (
            get_theo_var_l_two_time(test_ts, tau, D, a, b)
            - eq_perc * get_lin_var(tau, a, b, s)
        )
        ** 2
    )
    if (arg_min == len(test_ts)) or (arg_min == 0):
        print("no quliibrium reached")
        exact_eqtime = -1
    else:
        exact_eqtime = test_ts[arg_min]
    known_eq_times.append(
        {
            "params": {"tau": tau, "D": D, "eq_perc": eq_perc, "a": a, "b": b},
            "eq_time": exact_eqtime,
        }
    )
    with open("database/eq_times.json", "w") as file:
        json.dump(known_eq_times, file)
    return exact_eqtime


# Simulation
def simulate_traj_g(
    N_p: int,
    N_loop: int,
    N_t: int,
    ntau: int,
    s: float,
    dt: float,
    x_0: float,
    force: Callable,
    filter: Optional[list[float]] = None,
):
    pos = np.empty((N_loop, N_p, N_t + ntau), dtype=float)
    vel = s * np.random.randn(N_loop, N_p, N_t + ntau - 1) * 1 / np.sqrt(dt)
    pos[:, :, : ntau + 1] = x_0
    vel[:, :, :ntau] = 0
    for i in range(ntau + 1, N_t + ntau):
        vel[:, :, i - 1] += force(pos[:, :, i - 1], pos[:, :, i - ntau - 1])
        pos[:, :, i] = pos[:, :, i - 1] + vel[:, :, i - 1] * dt
        if filter is not None:
            pos[(pos[:, :, i] < filter[0]) | (pos[:, :, i] > filter[1]), i] = np.nan
    return pos


#  General functions
def get_var_hist(hists, x_s):
    if isinstance(hists, list):
        hists = np.stack(hists)
    if hists.ndim == 2:
        p = hists / np.sum(hists, axis=1)[:, None]
        return (
            np.sum(p * x_s[None, :] ** 2, axis=1)
            - np.sum(p * x_s[None, :], axis=1) ** 2
        )
    if hists.ndim == 1:
        p = hists / np.sum(hists)
        return np.sum(p * x_s**2) - np.sum(p * x_s) ** 2
    else:
        assert "Wrong number of dim in hists"


def linear_force_2(x_0, x_t):
    res_array = np.empty(np.broadcast_shapes(x_0.shape, x_t.shape))
    res_array[:] = -x_t
    return res_array


def no_delay_linear_force_2(x_0, x_t):
    res_array = np.empty(np.broadcast_shapes(x_0.shape, x_t.shape))
    res_array[:] = -x_0
    return res_array


def general_force(x_0, x_t):
    res_array = np.empty(np.broadcast_shapes(x_0.shape, x_t.shape))
    res_array[:] = -x_0 - x_t
    return res_array


def no_delay_general_force(x_0, x_t):
    res_array = np.empty(np.broadcast_shapes(x_0.shape, x_t.shape))
    res_array[:] = -x_0 - x_0
    return res_array


def cubic_force_2(x_0, x_t):
    res_array = np.empty(np.broadcast_shapes(x_0.shape, x_t.shape))
    res_array[:] = -(x_t**3)
    return res_array


def no_delay_cubic_force_2(x_0, x_t):
    res_array = np.empty(np.broadcast_shapes(x_0.shape, x_t.shape))
    res_array[:] = -(x_0**3)
    return res_array


def cusp_force_2(x_0, x_t, thresh=1e-7):
    b = np.where(x_t < -thresh, -(x_t + 1), x_t)
    b = np.where(x_t > thresh, -(x_t - 1), b)
    b = np.where(np.abs(x_t) < thresh, 0, b)

    res_array = np.empty(np.broadcast_shapes(x_0.shape, x_t.shape))
    res_array[:] = b
    return res_array


def no_delay_cusp_force_2(x_0, x_t, thresh=1e-7):
    b = np.where(x_0 < -thresh, -(x_0 + 1), x_t)
    b = np.where(x_0 > thresh, -(x_0 - 1), b)
    b = np.where(np.abs(x_0) < thresh, 0, b)

    res_array = np.empty(np.broadcast_shapes(x_0.shape, x_t.shape))
    res_array[:] = b
    return res_array


forces_dict_2: dict[str, Callable] = {
    "linear": linear_force_2,
    "cubic": cubic_force_2,
    "general": general_force,
    "cusp_force": cusp_force_2,
    "no_delay_general": no_delay_general_force,
    "no_delay_linear": no_delay_linear_force_2,
    "no_delay_cubic": no_delay_cubic_force_2,
    "no_delay_cusp_force": no_delay_cusp_force_2,
}


# Classes
class StorageManager:
    timing: float = -1.0

    def run(self, **kargs):
        myname = self.__class__.__name__
        self.data_dir = Path.cwd() / "database"
        self.data_dir.mkdir(exist_ok=True)
        self.overview_file = self.data_dir / f"{myname}.json"
        if (self.overview_file).is_file():
            from_file = json.load(open(self.overview_file, "r"))
            infos = [o for o in from_file if o["params"] == kargs]
            if len(infos) == 1:
                filename = infos[0]["filename"]
                self.timing = infos[0]["time"]
                return pickle.load(open(self.data_dir / filename, "rb"))
        else:
            from_file = []

        start = timer()
        result = self.main(**kargs)
        end = timer()

        current_time = datetime.datetime.now()
        # Create a filename that includes the current time, string, and number
        filename_base = f"{current_time.strftime('%Y-%m-%d')}_{myname}_"
        exsisting_files = list(self.data_dir.glob(filename_base + "*"))
        if len(exsisting_files) > 0:
            highst_idx = max([int(o.stem.split("_")[-1]) for o in exsisting_files])
        else:
            highst_idx = -1
        filename = filename_base + f"{highst_idx+1}.pkl"

        with open(self.data_dir / filename, "wb") as file:
            pickle.dump(result, file)

        timing = end - start
        from_file.append({"params": kargs, "filename": filename, "time": timing})
        with open(self.overview_file, "w") as file:
            json.dump(from_file, file)
        self.time = timing
        return result

    def main(self, *args, **kargs):
        print(
            "ERROR, every class which uses \
              StorageManager should have a run method"
        )

    def get_timing(self):
        return self.timing


class SimulationManager(StorageManager):
    def main(
        self,
        N_p: int,
        N_loop: int,
        N_t: int,
        N_x: int,
        ntau: int,
        s: float,
        dt: float,
        x_0: float,
        force: str,
        hist_sigma: float,
        norm_sigma: Optional[bool] = False,
        filter: Optional[list[float]] = None,
    ) -> dict:
        pos = simulate_traj_g(
            N_p,
            N_loop,
            N_t,
            ntau,
            s,
            dt,
            x_0,
            force=forces_dict_2[force],
            filter=filter,
        )
        pos_filtered = pos
        sim_var = np.nanvar(pos_filtered, axis=1)
        if norm_sigma:
            sb = hist_sigma
        else:
            sb = hist_sigma * np.sqrt(np.max(sim_var))

        dx = 2 * sb / (N_x - 1)
        x_s = np.arange(-sb, sb + 1e-6, dx)
        bins = np.arange(-sb - dx / 2, sb + dx / 2 + 1e-6, dx)

        sim_hists = np.swapaxes(
            np.apply_along_axis(lambda a: np.histogram(a, bins)[0], 1, pos), 1, 2
        )
        sim_hist_var = np.apply_along_axis(get_var_hist, -1, sim_hists, x_s=x_s)
        hist_sum = np.sum(sim_hists, axis=2)
        return {
            "x_s": x_s,
            "sim_var": sim_var,
            "sim_hist_var": sim_hist_var,
            "hist_sum": hist_sum,
        }


class SolverManager(StorageManager):
    def main(self, N_t, N_x, sb, ntau, s, dt, x_0, force, version):
        dx = 2 * sb / (N_x - 1)
        x_s = np.arange(-sb, sb + 1e-6, dx)
        i_zero = np.argmin((x_s - x_0) ** 2)
        D = s**2 / 2

        force_func = (
            forces_dict_2[force] if ntau > 0 else forces_dict_2["no_delay_" + force]
        )
        if version == 1:
            prop = get_prop_v1_1(x_s, force_func, D, dt)
        elif version == 2:
            prop = get_prop_v2_1(x_s, force_func, D, dt)

        R, _, end_states = create_R_v3(ntau, prop)
        _, hists = get_dyn_v2(R, i_zero, N_t, N_x, ntau, end_states)
        num_var = get_var_hist(hists, x_s)
        return {"num_var": num_var}


class EigenvectorManager(StorageManager):
    def main(self, N_x, sb, ntau, s, dt, force, version):
        dx = 2 * sb / (N_x - 1)
        x_s = np.arange(-sb, sb + 1e-6, dx)
        D = s**2 / 2

        force_func = (
            forces_dict_2[force] if ntau > 0 else forces_dict_2["no_delay_" + force]
        )
        if version == 1:
            prop = get_prop_v1_1(x_s, force_func, D, dt)
        elif version == 2:
            prop = get_prop_v2_1(x_s, force_func, D, dt)

        R, _, end_states = create_R_v3(ntau, prop)
        evals, evect = scipy.sparse.linalg.eigs(R, k=1, which="LR")

        main_evect = np.real(evect[:, 0])
        if main_evect.sum() < 0:
            main_evect *= -1
        eig_var = get_var_hist(main_evect[end_states].sum(axis=1), x_s)

        return {"eig_var": eig_var}


class SimulateRateManager(StorageManager):
    def main(
        self,
        N_p: int,
        N_loop: int,
        N_t: int,
        ntau: int,
        s: float,
        dt: float,
        x_0: float,
        border: float,
        force: str,
        absorbing: bool = False,
    ) -> dict:
        if absorbing:
            filter = [-np.inf, border]
        else:
            filter = None
        pos = simulate_traj_g(
            N_p,
            N_loop,
            N_t,
            ntau,
            s,
            dt,
            x_0,
            force=forces_dict_2[force],
            filter=filter,
        )
        if absorbing:
            sim_sur = np.mean(~np.isnan(pos), axis=1)
        else:
            sim_sur = np.sum(pos < border, axis=1) / np.sum(~np.isnan(pos), axis=1)

        return {
            "sim_sur": sim_sur,
        }


class SolverRateManager(StorageManager):
    def main(self, N_t, N_x, sbs, ntau, s, dt, x_0, force, border, absorbing, version):
        dx = (sbs[1] - sbs[0]) / (N_x - 1)
        x_s = sbs[0] + np.arange(N_x) * dx
        i_zero = np.argmin((x_s - x_0) ** 2)
        D = s**2 / 2

        force_func = (
            forces_dict_2[force] if ntau > 0 else forces_dict_2["no_delay_" + force]
        )
        if version == 1:
            prop = get_prop_v1_1(x_s, force_func, D, dt)
        elif version == 2:
            prop = get_prop_v2_1(x_s, force_func, D, dt)

        R, _, end_states = create_R_v3(ntau, prop)
        _, hists = get_dyn_v2(R, i_zero, N_t, N_x, ntau, end_states)
        if absorbing:
            num_sur = np.sum(hists, axis=1) / np.sum(hists[0])
        else:
            num_sur = np.sum(hists[:, x_s < border], axis=1) / np.sum(hists, axis=1)

        return {"num_sur": num_sur}


class EigenvectorRateManager(StorageManager):
    def main(self, N_x, sbs, ntau, s, dt, force, version):
        dx = (sbs[1] - sbs[0]) / (N_x - 1)
        x_s = sbs[0] + np.arange(N_x) * dx
        D = s**2 / 2

        force_func = (
            forces_dict_2[force] if ntau > 0 else forces_dict_2["no_delay_" + force]
        )
        if version == 1:
            prop = get_prop_v1_1(x_s, force_func, D, dt)
        elif version == 2:
            prop = get_prop_v2_1(x_s, force_func, D, dt)

        R, _, end_states = create_R_v3(ntau, prop)
        evals, evect = scipy.sparse.linalg.eigs(R, k=1, which="LR")

        main_evect = np.real(evect[:, 0])
        if main_evect.sum() < 0:
            main_evect *= -1

        return {"eig_rate": -np.log(evals[0]) / dt}
