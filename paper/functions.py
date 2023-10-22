import pickle
from timeit import default_timer as timer
import numpy as np
from scipy.linalg import expm

# from tqdm.notebook import tqdm
from tqdm import tqdm
from scipy.sparse import csr_array, coo_array
from scipy.special import factorial
import json
from pathlib import Path
import datetime


#  Nummerical (new)


def get_prop_abs(x_s, force, D, dt, dx, N_border=None):
    # x(t-tau), x(t), res
    N_x = len(x_s)
    R_abs = np.zeros((N_x, N_x, N_x))

    F = force(x_s)
    lp = D / dx**2 * np.exp((F * dx / D) / 2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(F * dx / D) / 2)  # r_i+1->i

    R_abs[:, np.arange(0, N_x), np.arange(0, N_x)] = -(
        lp[:, None] + ln[:, None]
    )  # -(r_i->i+1 + r_i->i-1) ????
    R_abs[:, np.arange(0, N_x - 1), np.arange(1, N_x)] = ln[:, None]
    R_abs[:, np.arange(1, N_x), np.arange(0, N_x - 1)] = lp[:, None]
    prop_abs = expm(
        R_abs * dt,
    )
    return prop_abs


def get_prop_abs_v2(x_s, force, D, dt, dx, N_border=None, side="lr"):
    # x(t-tau), x(t), res
    N_x = len(x_s)
    half_x_s = np.arange(x_s[0], x_s[-1] + dx / 4, dx / 2)

    R_abs = np.zeros((len(half_x_s), N_x, N_x))

    F = force(half_x_s)
    lp = D / dx**2 * np.exp((F * dx / D) / 2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(F * dx / D) / 2)  # r_i+1->i
    if side == "r":
        if N_border is None:
            N_border = N_x
        R_abs[:, 0, 0] = -lp
        R_abs[:, np.arange(1, N_border), np.arange(1, N_border)] = -(
            lp[:, None] + ln[:, None]
        )  # -(r_i->i+1 + r_i->i-1) ????
        R_abs[:, np.arange(0, N_border - 1), np.arange(1, N_border)] = ln[:, None]
        R_abs[:, np.arange(1, N_border), np.arange(0, N_border - 1)] = lp[:, None]
    elif side == "l":
        if N_border is None:
            N_border = 0
        R_abs[:, -1, -1] = -ln
        R_abs[:, np.arange(N_border, N_x - 1), np.arange(N_border, N_x - 1)] = -(
            lp[:, None] + ln[:, None]
        )  # -(r_i->i+1 + r_i->i-1) ????
        R_abs[:, np.arange(N_border, N_x - 1), np.arange(N_border + 1, N_x)] = ln[
            :, None
        ]
        R_abs[:, np.arange(N_border + 1, N_x), np.arange(N_border, N_x - 1)] = lp[
            :, None
        ]
    elif side == "lr":
        if N_border is not None:
            print("for lr N_border is ignored")
        R_abs[:, np.arange(0, N_x), np.arange(0, N_x)] = -(
            lp[:, None] + ln[:, None]
        )  # -(r_i->i+1 + r_i->i-1) ????
        R_abs[:, np.arange(0, N_x - 1), np.arange(1, N_x)] = ln[:, None]
        R_abs[:, np.arange(1, N_x), np.arange(0, N_x - 1)] = lp[:, None]
    prop_abs = expm(
        R_abs * dt,
    )
    if np.any(np.isnan(prop_abs)):
        print("CAREFUL: nan in prop, maybe because of to high values in potential")
    return prop_abs


def create_R(N_x, ntau, prop):
    all_states = np.arange(0, N_x ** (ntau + 1))

    # staetes # 1 * x(t-tau), N_x*x(t-tau1+1dt), ... , N_x**ntau * x(t)

    lm = all_states % N_x  # mean t-tau state
    lt = all_states // N_x ** (ntau)  # t state
    all_next_states = (all_states // N_x)[:, None] + (
        (N_x**ntau) * (np.arange(0, N_x))
    )[None, :]

    R = coo_array(
        (
            prop[lm, :, lt].flatten(),
            (all_next_states.flatten(), all_states.repeat(N_x)),
        ),
        shape=(N_x ** (ntau + 1), N_x ** (ntau + 1)),
    )

    R = csr_array(R)
    end_states = np.stack(
        [all_states[all_states // N_x**ntau == i] for i in range(N_x)]
    )
    return R, all_states, end_states


def create_R_v1(N_x, ntau, prop):
    all_states = np.arange(0, N_x ** (ntau + 1))

    # staetes # 1 * x(t-tau), N_x*x(t-tau1+1dt), ... , N_x**ntau * x(t)

    lm = all_states % N_x + (all_states // N_x) % N_x  # mean t-tau state
    lt = all_states // N_x ** (ntau)  # t state
    all_next_states = (all_states // N_x)[:, None] + (
        (N_x**ntau) * (np.arange(0, N_x))
    )[None, :]

    R = coo_array(
        (
            prop[lm, :, lt].flatten(),
            (all_next_states.flatten(), all_states.repeat(N_x)),
        ),
        shape=(N_x ** (ntau + 1), N_x ** (ntau + 1)),
    )

    R = csr_array(R)
    end_states = np.stack(
        [all_states[all_states // N_x**ntau == i] for i in range(N_x)]
    )
    return R, all_states, end_states


def get_dyn_v2(R, i_zero, N_t, N_x, ntau, end_states):
    initial_state = np.sum(i_zero * (N_x ** np.arange(0, ntau + 1)))
    p = np.zeros(N_x ** (ntau + 1), dtype=float)
    p[initial_state] = 1.0
    one_time_p = np.empty((N_t, N_x))
    one_time_p[0] = np.sum(p[end_states], axis=1)
    for i in tqdm(range(1, N_t), leave=False):
        p = R @ p
        one_time_p[i] = np.sum(p[end_states], axis=1)

    return p, one_time_p


def get_non_delayed_prop(x_s, force, D, dt, dx, N_border=None, Fp=None, Fl=None):
    N_x = len(x_s)
    mx_s = np.arange(x_s[0] - dx / 2, x_s[-1] + dx / 2 + dx / 4, dx)
    if Fp is None or Fl is None:
        F = force(mx_s)
    if Fp is None:
        Fp = F
    if Fl is None:
        Fl = F
    R_abs = np.zeros((N_x, N_x))

    lp = D / dx**2 * np.exp((Fp * dx / D) / 2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(Fl * dx / D) / 2)  # r_i+1->i

    R_abs[np.arange(0, N_x), np.arange(0, N_x)] = -(
        lp[1:] + ln[:-1]
    )  # -(r_i->i+1 + r_i->i-1) ????
    R_abs[np.arange(0, N_x - 1), np.arange(1, N_x)] = ln[1:-1]
    R_abs[np.arange(1, N_x), np.arange(0, N_x - 1)] = lp[1:-1]
    prop_abs = expm(
        R_abs * dt,
    )
    return prop_abs


def get_non_delayed_dyn(R, i_zero, N_t, N_x):
    p = np.zeros((N_t, N_x), dtype=float)
    p[0, i_zero] = 1.0
    if R.ndim == 2:
        for i in tqdm(range(1, N_t), leave=False):
            p[i] = R @ p[i - 1]
        return p
    else:
        for i in tqdm(range(1, N_t), leave=False):
            p[i] = R[i - 1] @ p[i - 1]
        return p


# Analytical

# OLD !?
# def get_p_x4_short_time(x,k,tau,s):
#     ga = s*(1+3*tau*k*x**2)
#     p_not_notmed = np.exp(-x**2/(3*tau*s**2)+(1-18*tau**2*k*s**2)/(9*tau**2*k*s**2)*np.log(1+3*tau*k*x**2))
#     return p_not_notmed/np.sum(p_not_notmed)

# def get_p_x4_short_time(x,tau): # WRONG !!!
#     res = 1 / (1 + 3 * tau * x**2)**2 * np.exp( - ( (1/(3*tau*x**2 + 1) + np.log(3*tau*x**2+1)) / (9*tau**2) ))
#     return res/res.sum()


def get_p_x4_short_time(x, tau):
    res = (
        1
        / (1 + 3 * tau * x**2) ** 2
        * np.exp(-((3 * tau * x**2 - np.log(3 * tau * x**2 + 1)) / (9 * tau**2)))
    )
    return res / res.sum()


def get_x2_var(tau, k, s):
    return s**2 / (2 * k) * (1 + np.sin(k * tau)) / np.cos(k * tau)


def get_x2_var_short_time(tau, k, s):
    return s**2 / (2 * k) * (1 + k * tau)


# Theory


# WRONG !!!
def summand(t, l1, l2, tau, k=1):
    return (
        (-k) ** (l1 + l2)
        / (factorial(l1) * factorial(l2))
        * (
            1 / (l1 + 1) * (t - l1 * tau) ** (l1 + 1) * (t - l2 * tau) ** l2
            + 1 / (l2 + 1) * (t - l2 * tau) ** (l2 + 1) * (t - l1 * tau) ** l1
        )
    )


# WRONG !!!!!
def get_theo_var(t, tau, D, k=1):
    l1 = np.arange(0, t / tau)
    l2 = np.arange(0, t / tau)
    return 2 * D * (np.sum(summand(t, l1[:, None], l2[None, :], tau, k=k)))


get_theo_var = np.vectorize(get_theo_var, excluded=(1, 2))


def l(k, tau, t, max_p=40):  # noqa: E743 (linting ignore short name)
    i = np.arange(0, max_p, 1)
    return np.sum(
        (-k) ** i / factorial(i) * (t - i * tau) ** i * np.heaviside(t - i * tau, 1)
    )


l = np.vectorize(  # noqa: E741 (linting ignore short name)
    l,
)


def get_theo_var_l(ts, tau, D, k=1):
    if tau > 0:
        dt = ts[1] - ts[0]
        max_p = np.max(ts) / tau
        l_data = l(k, tau, ts, max_p)
        var = np.zeros(len(ts))
        var[1:] = (2 * D * np.cumsum(l_data**2) * dt)[:-1]
        return var
    else:
        return D / k * (1 - np.exp(-2 * k * ts))


def get_eq_times(tau, D, eq_perc):
    database_path = Path.cwd() / "database/eq_times.json"
    if database_path.is_file():
        known_eq_times = json.load(open(database_path))
    else:
        known_eq_times = []
    eq_time = [
        o["eq_time"]
        for o in known_eq_times
        if o["params"] == {"tau": tau, "D": D, "eq_perc": eq_perc}
    ]
    if len(eq_time):
        return eq_time[0]

    s = np.sqrt(2 * D)
    if tau < 1:
        test_ts = np.linspace(0, 5, 16000)
    else:
        test_ts = np.linspace(0, 40, 16000)

    arg_min = np.argmin(
        (get_theo_var_l(test_ts, tau, D) - eq_perc * get_x2_var(tau, 1, s)) ** 2
    )
    if (arg_min == len(test_ts)) or (arg_min == 0):
        print("no quliibrium reached")
        exact_eqtime = -1
    else:
        exact_eqtime = test_ts[arg_min]
    known_eq_times.append(
        {"params": {"tau": tau, "D": D, "eq_perc": eq_perc}, "eq_time": exact_eqtime}
    )
    with open("database/eq_times.json", "w") as file:
        json.dump(known_eq_times, file)
    return exact_eqtime


# Simulation
def simulate_traj(N_p, N_loop, N_t, ntau, s, dt, border, force):
    pos = np.empty((N_loop, N_p, N_t))
    vel = s * np.random.randn(N_loop, N_p, N_t - 1) * 1 / np.sqrt(dt)

    pos[:, :, : ntau + 1] = -border
    vel[:, :, :ntau] = 0

    for i in tqdm(range(ntau + 1, N_t), leave=False):
        vel[:, :, i - 1] += force(pos[:, :, i - ntau - 1])
        pos[:, :, i] = pos[:, :, i - 1] + vel[:, :, i - 1] * dt
    return pos


#  General functions
def get_var_hist(hists, x_s):
    if isinstance(hists, list):
        hists = np.stack(hists)
    if hists.ndim == 2:
        p = hists / np.sum(hists, axis=1)[:, None]
        return np.sum(p * x_s[None, :] ** 2 - (p * x_s[None, :]) ** 2, axis=1)
    if hists.ndim == 1:
        p = hists / np.sum(hists)
        return np.sum(p * x_s**2 - (p * x_s) ** 2)
    else:
        assert "Wrong number of dim in hists"


def get_steady_mean(data, i=None, max_err=1, thresh=0.1, min_states=5):
    if i is None:
        i = len(data) - 1
    ref_var = data[i]
    test_points = data[i::-1]
    mask = np.cumprod(np.abs(test_points - ref_var) < ref_var * thresh).astype(bool)
    if np.sum(mask) < min_states:
        return False
    steady_points = test_points[mask]
    relerr = (
        np.std(steady_points) / np.sqrt(len(steady_points)) / np.mean(steady_points)
    )
    if relerr < max_err:
        return np.mean(steady_points), np.std(steady_points) / np.sqrt(
            len(steady_points)
        )
    return False


# Forces
def linear_force(x):
    return -x


forces_dict = {"linear": linear_force}


# Classes
class StorageManager:
    def run(self, **kargs):
        myname = self.__class__.__name__
        self.data_dir = Path.cwd() / "database"
        self.overview_file = self.data_dir / f"{myname}.json"
        if (self.overview_file).is_file():
            from_file = json.load(open(self.overview_file, "r"))
            filenames = [o["filename"] for o in from_file if o["params"] == kargs]
            if len(filenames) == 1:
                return pickle.load(open(self.data_dir / filenames[0], "rb"))
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
        filename = filename_base + f"{highst_idx+1}.txt"

        with open(self.data_dir / filename, "wb") as file:
            pickle.dump(result, file)

        from_file.append({"params": kargs, "filename": filename, "time": end - start})
        with open(self.overview_file, "w") as file:
            json.dump(from_file, file)
        return result

    def main(self, *args, **kargs):
        print(
            "ERROR, every class which uses \
              StorageManager should have a run method"
        )


class SimulationManager(StorageManager):
    def main(self, N_p, N_loop, N_t, N_x, ntau, s, dt, x_0, force, hist_sigma):
        pos = simulate_traj(
            N_p, N_loop, N_t, ntau, s, dt, x_0, force=forces_dict[force]
        )
        sim_var = np.var(pos, axis=1)

        sb = hist_sigma * np.sqrt(np.max(sim_var))
        dx = 2 * sb / (N_x - 1)

        x_s = np.arange(-sb, sb + 1e-6, dx)
        bins = np.arange(-sb - dx / 2, sb + dx / 2 + 1e-6, dx)

        sim_hists = np.swapaxes(
            np.apply_along_axis(lambda a: np.histogram(a, bins)[0], 1, pos), 1, 2
        )
        sim_hist_var = np.apply_along_axis(get_var_hist, -1, sim_hists, x_s=x_s)
        return {"x_s": x_s, "sim_var": sim_var, "sim_hist_var": sim_hist_var}


class SolverManager(StorageManager):
    def main(self, N_t, N_x, sb, ntau, s, dt, x_0, force):
        dx = 2 * sb / (N_x - 1)
        x_s = np.arange(-sb, sb + 1e-6, dx)
        i_zero = np.argmin((x_s - x_0) ** 2)
        D = s**2 / 2

        if ntau > 0:
            # v1
            # prop = get_prop_abs(x_s, force,D,ldt,dx)
            # R, _, end_states = create_R(N_x, ntau, prop)

            # v2
            prop = get_prop_abs_v2(x_s, forces_dict[force], D, dt, dx)
            R, _, end_states = create_R_v1(N_x, ntau, prop)

            _, hists = get_dyn_v2(R, i_zero, N_t, N_x, ntau, end_states)
        else:
            R = get_non_delayed_prop(x_s, forces_dict[force], D, dt, dx)
            hists = get_non_delayed_dyn(R, i_zero, N_t, N_x)

        num_var = get_var_hist(hists, x_s)
        return {"num_var": num_var}
