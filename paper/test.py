# %%
import numpy as np
import matplotlib.pyplot as plt
from functions import (
    SimulationManager,
    EigenvectorManager,
    SolverManager,
    get_ana_hist_var,
)

# %%
N_p = 4_000
N_loop = 40
N_t = 5_000
N_x = 21

ntau = 0
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
