# %%
import numpy as np
import matplotlib.pyplot as plt
from functions import (
    create_R_v3,
    get_non_delayed_prop,
    get_prop_v3,
    get_prop_abs_v2,
    get_prop_v2_1,
)

# %%
D = 1 / 2
N_x = 25
sb = 3
x_s = np.linspace(-sb, sb, N_x)
dx = x_s[1] - x_s[0]
dt = 0.25 / 4  # 3


def f(x_0, x_t):
    return -x_0  # -x_t


f = np.vectorize(f)


# %%
# dim as , x^t_f,x^t_i, p_or_m, x_i ,t


# %%
R = get_prop_v2_1(x_s, f, D, dt)
Real_R = create_R_v3(3, R)
# %%
plt.imshow(R[0, 0])

# %%
plt.plot(R[12, 12, :, 12])
plt.plot(R[0, 0, :, 12])
plt.plot(R[24, 24, :, 12])

# %%
prop_comp = get_prop_abs_v2(x_s, lambda x: f(0, x), D, dt, dx)
non_delay = get_non_delayed_prop(x_s, lambda x: f(x, 0), D, dt, dx)
prop_comp.shape, non_delay.shape
# %%
plt.plot(R[12, 12, :, 4], color="C0", alpha=0.5)
plt.plot(R[12, 12, :, 12], color="C1", alpha=0.5)
plt.plot(R[12, 12, :, 20], color="C2", alpha=0.5)

plt.plot(non_delay[:, 4], color="C0", ls="--")
plt.plot(non_delay[:, 12], color="C1", ls="--")
plt.plot(non_delay[:, 20], color="C2", ls="--")

# plt.plot(prop_comp[24, :, 4], color="C0", ls="--")
# plt.plot(prop_comp[24, :, 12], color="C1", ls="--")
# plt.plot(prop_comp[24, :, 20], color="C2", ls="--")

plt.vlines([4, 12, 20], *plt.ylim())
# %%
# %%
plt.plot(R[12, 12, :, 12], color="C0", alpha=0.5)
plt.plot(R[0, 0, :, 12], color="C1", alpha=0.5)
plt.plot(R[24, 24, :, 12], color="C2", alpha=0.5)

plt.plot(prop_comp[24, :, 12], color="C0", ls="--")
plt.plot(prop_comp[0, :, 12], color="C1", ls="--")
plt.plot(prop_comp[48, :, 12], color="C2", ls="--")

plt.xlim(8, 16)

# %%
