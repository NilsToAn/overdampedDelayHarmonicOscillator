import pickle
from timeit import default_timer as timer
import numpy as np
import scipy
from typing import Callable, Optional
from scipy.special import factorial
import json
from pathlib import Path
import datetime

from algorithm import get_prop, create_R, get_dyn


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
    r"""Returns the steady state variance for a linear
    time delayed stachstic system.
    `\dot{x}(t) = -a x(t) -b x(t - \tau) + s \xi`

    Parameters
    ----------
    tau : float
        Time delay
    a : float
        paramter for the no timedelay coupling
    b : float
        paramter for the timedelay coupling
    s : float
        strength of the noise

    Returns
    -------
    float
        The steady state variance
    """
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


def get_lin_var_small_delay(tau, a, b, s):
    r"""Returns the small delay approximation of the steady
    state variance for a linear time delayed stachstic system.
    `\dot{x}(t) = -a x(t) -b x(t - \tau) + s \xi`

    Parameters
    ----------
    tau : float
        Time delay
    a : float
        paramter for the no timedelay coupling
    b : float
        paramter for the timedelay coupling
    s : float
        strength of the noise

    Returns
    -------
    float
        The steady state variance
    """
    D = 1 / 2 * s**2
    return D / (a + b) * (1 + b * tau)


# Theory
def l_two_time(a: float, b: float, tau: float, t: float, max_p: int = 40):
    r"""Calculates the kernel function `\lambda(t)` used for the solution
    of `\dot{x}(t) = -a x(t) -b x(t - \tau) + s \xi`.

    Parameters
    ----------
    a : float
        non delayed coupling
    b : float
        delayed coupling
    tau : float
        time delay
    t : float
        time point
    max_p : int, optional
        highest order to be calculated, by default 40

    Returns
    -------
    float
        value of the kernel function
    """
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
    r"""Get analytical time dependend variance for the linear
    time-delayed stochstaic differential equation
    `\dot{x}(t) = -a x(t) -b x(t - \tau) + \sqrt{2 D} \xi` by integrating
    the kernel function.



    Parameters
    ----------
    ts : np.ndarray
        The timesteps, which are used for integration and at which the variance is evaluated
    tau : float
        Time delay
    D : float
        Noise strength
    a : int, optional
        non time delay coupling, by default 1
    b : int, optional
        time delay coupling, by default 1

    Returns
    -------
    np.ndarray
        variance at the timepoints ts
    """
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


def get_eq_times(tau: float, D: float, eq_perc: float, a: float, b: float):
    r"""Get the times the linear system descripted by
    `\dot{x}(t) = -a x(t) -b x(t - \tau) + \sqrt{2D} \xi` takes to
    reach (eq_perc)% of its it steady state variance. If already
    calculated load from disk.

    Parameters
    ----------
    tau : float
        time delay
    D : float
        Strength of noise
    eq_perc : float
        percentage of final variance
    a : float
        non time delay coupling
    b : float
        time delay coupling

    Returns
    -------
    float
        the equilibration time
    """
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
    r"""Simulate trajectories for a given time delayed stachstic
    system.

    Parameters
    ----------
    N_p : int
        Number of particles
    N_loop : int
        Number of runs
    N_t : int
        Number interation steps
    ntau : int
        time delay in units of dt (N_t in the paper)
    s : float
        strength of the noise
    dt : float
        temporal resolution
    x_0 : float
        inital condition
    force : Callable
        force function with two arguments x(t) and x(t - tau)
    filter : Optional[list[float]], optional
        All particle moving out of devined filter are set to nan, by default None

    Returns
    -------
    np.ndarray
        all trajectories
    """
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
def get_var_hist(hists: np.ndarray, x_s: np.ndarray) -> float:
    r"""Calculate the variance for a given histogram.

    Parameters
    ----------
    hists : np.ndarray
        the histogram data
    x_s : np.ndarray
        the corresponding x values

    Returns
    -------
    float
        the variance
    """
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
    return np.nan


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
        """Runs the main function of the child object and returns the result.
        If the same function was executed with the same parameters, it just loads
        the result form the hard drive.

        Returns
        -------
        dict
            the result of the function
        """
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
        """Gives the time in seconds, it took to calculate the last result.

        Returns
        -------
        float
            time
        """
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
        """Simulate time delay stochastic difertial equation.
        If the same settings were run before load for hard drive (StorageManager).

        Parameters
        ----------
        N_p : int
            Number of particles per run.
        N_loop : int
            Number of runs (used to calc mean and std)
        N_t : int
            Number of timesteps
        N_x : int
            Number of bins when calculating the histograms
        ntau : int
            time delay in units of dt (N_t in the paper)
        s : float
            strength of noise
        dt : float
            temporal discretisation
        x_0 : float
            initial condition
        force : str
            name of the force, which is part of `forces_dict_2`
        hist_sigma : float
            boundaries for the hist in units of the the calculated variance
            or if norm_sigma as fixed value.
        norm_sigma : Optional[bool], optional
            if true boundaries do not scale with variance, by default False
        filter : Optional[list[float]], optional
            particle crossing boundaries of filter are set to np.nan, by default None

        Returns
        -------
        dict
            all results in the form:
        {
            "x_s": x_s,
            "sim_var": sim_var,
            "sim_hist_var": sim_hist_var,
            "hist_sum": hist_sum,
        }
        """
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
    def main(
        self,
        N_t: int,
        N_x: int,
        sb: float,
        ntau: int,
        s: float,
        dt: float,
        x_0: float,
        force: str,
    ) -> dict:
        r"""Use the dynamical algorithm to solve the dynamic.
        If the same settings were run before load for hard drive (StorageManager).

        Parameters
        ----------
        N_t : int
            Number of total timesteps
        N_x : int
            Number of spatial steps
        sb : float
            Distance from the border to the 0.
        ntau : int
            time delay in units of dt (in the paper N_t)
        s : float
            strength of noise
        dt : float
            temporal discretisation
        x_0 : float
            initial condition
        force : str
            name of the force, which is part of `forces_dict_2`

        Returns
        -------
        dict
            containing the timedependend variance {"num_var": num_var}
        """
        dx = 2 * sb / (N_x - 1)
        x_s = np.arange(-sb, sb + 1e-6, dx)
        i_zero = int(np.argmin((x_s - x_0) ** 2))
        D = s**2 / 2

        force_func = (
            forces_dict_2[force] if ntau > 0 else forces_dict_2["no_delay_" + force]
        )

        prop = get_prop(x_s, force_func, D, dt)

        R, _, end_states = create_R(ntau, prop)
        _, hists = get_dyn(R, i_zero, N_t, N_x, ntau, end_states)
        num_var = get_var_hist(hists, x_s)
        return {"num_var": num_var}


class EigenvectorManager(StorageManager):
    def main(
        self, N_x: int, sb: float, ntau: int, s: float, dt: float, force: str
    ) -> dict:
        r"""use the steady state version of the algorithm to calculate
        the stable state configuration, which fullfills the eigenvalue equation.
        If the same settings were run before load for hard drive (StorageManager).

        Parameters
        ----------
        N_x : int
            Number spatial points
        sb : float
            distance of the boundary to 0.
        ntau : int
            time delay in units of dt (N_t in the paper)
        s : float
            strength of the noise
        dt : float
            temporal discretisation
        force : str
            name of the force, which is part of `forces_dict_2`

        Returns
        -------
        dict
           steady state variance {"eig_var": eig_var}
        """
        dx = 2 * sb / (N_x - 1)
        x_s = np.arange(-sb, sb + 1e-6, dx)
        D = s**2 / 2

        force_func = (
            forces_dict_2[force] if ntau > 0 else forces_dict_2["no_delay_" + force]
        )

        prop = get_prop(x_s, force_func, D, dt)

        R, _, end_states = create_R(ntau, prop)
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
        """Use a simulation to determin the rate in a potential either crossing the border
        or getting absorbed at the border.
        If the same settings were run before load for hard drive (StorageManager).

        Parameters
        ----------
        N_p : int
            Number of particles
        N_loop : int
            Number of runs
        N_t : int
            Number of timesteps
        ntau : int
            Time delay in units of dt (N_t in the paper)
        s : float
            Strength of the noise
        dt : float
            Temporal discretisation
        x_0 : float
            Inital condition
        border : float
            position of the border
        force : str
            name of the force, which is part of `forces_dict_2`
        absorbing : bool, optional
            if true all particles are removed if appraoching the border, by default False

        Returns
        -------
        dict
            The timedepend probabilty to stay in the inital well {"sim_sur": sim_sur}
        """
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
    def main(
        self,
        N_t: int,
        N_x: int,
        sbs: list,
        ntau: int,
        s: float,
        dt: float,
        x_0: float,
        force: str,
        border: float,
        absorbing: bool,
    ) -> dict:
        """Use the algorithm to determin the rate in a potential either crossing the border
        or getting absorbed at the border.
        If the same settings were run before load for hard drive (StorageManager).


        Parameters
        ----------
        N_t : int
            Number of timesteps
        N_x : int
            Number of space points
        sbs : list
            sbs[0] is left boundary and sbs[1] right boundary
        ntau : int
            time delay in units of dt
        s : float
            strength of noise
        dt : float
            temporal resolution
        x_0 : float
            inital condition
        force : str
            name of the force, which is part of `forces_dict_2`
        border : float
            The position of the border
        absorbing : bool
            If absorbing all particles are absorbed at the border

        Returns
        -------
        dict
            The timedepend probabilty to stay in the inital well {"num_sur": num_sur}
        """
        dx = (sbs[1] - sbs[0]) / (N_x - 1)
        x_s = sbs[0] + np.arange(N_x) * dx
        i_zero = int(np.argmin((x_s - x_0) ** 2))
        D = s**2 / 2

        force_func = (
            forces_dict_2[force] if ntau > 0 else forces_dict_2["no_delay_" + force]
        )
        prop = get_prop(x_s, force_func, D, dt)

        R, _, end_states = create_R(ntau, prop)
        _, hists = get_dyn(R, i_zero, N_t, N_x, ntau, end_states)
        if absorbing:
            num_sur = np.sum(hists, axis=1) / np.sum(hists[0])
        else:
            num_sur = np.sum(hists[:, x_s < border], axis=1) / np.sum(hists, axis=1)

        return {"num_sur": num_sur}


class EigenvectorRateManager(StorageManager):
    def main(
        self, N_x: int, sbs: list, ntau: int, s: float, dt: float, force: str
    ) -> dict:
        """Use the steady state method to determin the rates only possible for
        absorbing case.

        Parameters
        ----------
        N_x : int
            Number of space points
        sbs : list
            sbs[0] is left boundary and sbs[1] right boundary
        ntau : int
            time delay in units of dt
        s : float
            strength of noise
        dt : float
            temporal resolution
        force : str
            name of the force, which is part of `forces_dict_2`

        Returns
        -------
        dict
            The steady state rate {"eig_rate": eig_rate}
        """
        dx = (sbs[1] - sbs[0]) / (N_x - 1)
        x_s = sbs[0] + np.arange(N_x) * dx
        D = s**2 / 2

        force_func = (
            forces_dict_2[force] if ntau > 0 else forces_dict_2["no_delay_" + force]
        )
        prop = get_prop(x_s, force_func, D, dt)

        R, _, end_states = create_R(ntau, prop)
        evals, evect = scipy.sparse.linalg.eigs(R, k=1, which="LR")

        main_evect = np.real(evect[:, 0])
        if main_evect.sum() < 0:
            main_evect *= -1

        return {"eig_rate": -np.log(evals[0]) / dt}
