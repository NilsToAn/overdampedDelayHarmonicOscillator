# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
from functions import (
    SimulationManager,
    EigenvectorManager,
    SolverManager,
    create_R_v3,
    get_ana_hist_var,
    get_prop_v2_1,
    forces_dict_2,
    get_quantil_hist,
    get_theo_var_l_two_time,
    get_var_hist,
    get_lin_var,
    get_dyn_v2,
    simulate_traj_g,
)

# %%
N_p = 4_000
N_loop = 40
N_t = 5_000
N_x = 21

ntau = 3
s = 1
dt = 0.001
x_0 = 0
force = "linear"
hist_sigma = 4

time = np.arange(0, N_t) * dt
# %%
my_sim_manager = SimulationManager()
my_eig_manager = EigenvectorManager()
res = my_sim_manager.run(
    N_p=N_p,
    N_loop=N_loop,
    N_t=N_t,
    N_x=N_x,
    ntau=ntau,
    s=s,
    dt=dt,
    x_0=x_0,
    force=force,
    hist_sigma=hist_sigma,
)
sim_var_err = (
    np.mean(res["sim_var"], axis=0),
    np.std(res["sim_var"], axis=0) / np.sqrt(N_loop),
)
sim_var_hist_err = (
    np.mean(res["sim_hist_var"], axis=0),
    np.std(res["sim_hist_var"], axis=0) / np.sqrt(N_loop),
)
x_s = res["x_s"]
sb = -x_s[0]
# %%
my_num_manager = SolverManager()

res = my_num_manager.run(
    N_t=N_t, N_x=N_x, sb=sb, ntau=0, s=s, dt=dt, x_0=0, force=force
)

num_var = res["num_var"]
num_var.shape
# %%
num_res = my_eig_manager.run(N_x=N_x, sb=sb, ntau=ntau, s=s, dt=dt, force=force)
# %%
ana_hist_var = get_ana_hist_var(N_x, x_s[1] - x_s[0], 0.5)

# %%
plt.errorbar(time, *sim_var_err)
plt.errorbar(time, *sim_var_hist_err)
plt.errorbar(time, num_var)

plt.xlim(*plt.xlim())
plt.hlines(0.5, *plt.xlim())
plt.hlines(ana_hist_var, *plt.xlim(), color="C1", ls="--")
plt.hlines(num_res["eig_var"], *plt.xlim(), color="C1")

plt.ylim(0.45, 0.55)

# %%
num_res["eig_var"] / 0.5

# %%
ana_hist_var

# %%
np.array([1, 0, 1]) | np.array([1, 0, 1])
# %%
{"a": (1, 0)} == {"a": [1, 0]}
# %%
a = np.array([-2, 0.1, 2])
b = np.where(a < 0.2, a**2, a)
b = np.where(a > 0.2, a**2, b)
b = np.where(np.abs(a) < 0.2, 0, b)
b
# %%
np.broadcast_shapes((25, 25, 1, 1), (1, 1, 25, 25))


# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
from functions import (
    SimulationManager,
    EigenvectorManager,
    SolverManager,
    create_R_v3,
    get_ana_hist_var,
    get_prop_v2_1,
    forces_dict_2,
    get_quantil_hist,
    get_theo_var_l_two_time,
    get_var_hist,
    get_lin_var,
    get_dyn_v2,
    simulate_traj_g,
)

ntau = 2
s = 1
D = 1 / 2 * s**2
dt = 0.1
N_x = 79
force = "general"
if force == "linear":
    a = 0.0
    b = 1.0
elif force == "general":
    a = 1.0
    b = 1.0
N_t = 60
N_p = 2000
N_loop = 40
x_0 = 0

sim_N_t = 600
sim_dt = 0.01
sim_ntau = 20

time = np.arange(0, N_t) * dt
sim_time = np.arange(0, sim_N_t) * sim_dt


truth = get_lin_var(dt * ntau, a=a, b=b, s=1)
x_s = np.linspace(-4 * np.sqrt(truth), 4 * np.sqrt(truth), N_x)
i_zero = np.argmin(x_s**2)

prop = get_prop_v2_1(x_s, forces_dict_2[force], D, dt)
R, _, end_states = create_R_v3(ntau, prop)

evals, evect = scipy.sparse.linalg.eigs(R, k=1, which="LR")
main_evect = np.real(evect[:, 0])
if main_evect.sum() < 0:
    main_evect *= -1
eig_var = get_var_hist(main_evect[end_states].sum(axis=1), x_s)
print(eig_var, truth, eig_var / truth)

_, hists = get_dyn_v2(R, i_zero, N_t, N_x, ntau, end_states)
num_var = get_var_hist(hists, x_s)

pos = simulate_traj_g(
    N_p, N_loop, sim_N_t, sim_ntau, s, sim_dt, x_0, force=forces_dict_2[force]
)
pos_filtered = pos
sim_var = np.nanvar(pos_filtered, axis=1)
mean_sim_var = np.mean(sim_var, axis=0)

sim_q = np.quantile(pos_filtered, 0.842, axis=1) ** 2
mean_sim_q = np.mean(sim_q, axis=0)

# %%
plt.plot(time, num_var)
plt.plot(sim_time, mean_sim_var[sim_ntau:])
high_res_t = np.linspace(time[0], time[-1], 1000)
plt.plot(high_res_t, get_theo_var_l_two_time(high_res_t, dt * ntau, D, a=a, b=b))
plt.hlines(truth, *plt.xlim())

# %%
plt.plot(time, get_quantil_hist(hists, x_s))
plt.plot(time, num_var)
plt.plot(sim_time, mean_sim_q[sim_ntau:])
# %%
get_quantil_hist(hists, x_s)

# %%
num_var[1]
q = 0.84

p = hists / np.sum(hists, axis=1)[:, None]
q_dis = np.cumsum(p, axis=1) - q
cross = np.where((q_dis[:, :-1] * q_dis[:, 1:]) < 0)

x1 = x_s[cross[1]]
x2 = x_s[cross[1] + 1]
y1 = q_dis[cross[0], cross[1]]
y2 = q_dis[cross[0], cross[1] + 1]
x1.shape, y1.shape
mytry = (-y2 * (x2 - x1) / (y2 - y1) + x2 + (x_s[1] - x_s[0]) / 2) ** 2

# %%
import numpy as np

a = np.array([-np.inf, 1, np.nan, 2, 3, 4, np.inf])
np.quantile(a[~np.isnan(a)], 0.7)
# %%
from pathlib import Path

for path in Path("database").glob("*.txt"):
    path.rename(str(path).replace(".txt", ".pkl"))

# %%
import numpy as np

tests = np.random.randint(1, 7, (100000, 100))
rolls = np.argmax(tests == 1, axis=1) + 1
np.mean(rolls)
