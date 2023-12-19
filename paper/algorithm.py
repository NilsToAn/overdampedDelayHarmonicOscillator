from scipy.linalg import expm
from scipy.sparse import csr_array, coo_array
import numpy as np

from typing import Callable


def get_prop(
    x_s: np.ndarray, f: Callable[..., np.ndarray], D: float, dt: float
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    x_s : np.ndarray
        The spatial discritisation
    f : Callable
        The function f(x_0, x_t) vectorized
    D : float
        Diffusitivity, s**2/2
    dt : float
        temporal spacing

    Returns
    -------
    np.ndarray
        x^t_f,x^t_i, x_f, x_i
    """
    N_x = len(x_s)
    dx = x_s[1] - x_s[0]
    # dim as , x^t_f,x^t_i, x_f, x_i

    p_or_m = np.array([-1.0, 1.0])[None, None, :, None]
    x_step = p_or_m * dx
    x_i = x_s[None, None, None, :]
    x_t_i = x_s[None, :, None, None]
    x_t_f = x_s[:, None, None, None]
    crazy_tensor = f(x_i + x_step / 2, 1 / 2 * (x_t_f + x_t_i))
    U_raw = D / dx**2 * np.exp(p_or_m * crazy_tensor * dx / (2 * D))
    U_full = np.zeros([N_x] * 4, dtype=float)
    U_full[:, :, np.arange(0, N_x, 1), np.arange(0, N_x, 1)] = -(
        U_raw[:, :, 0, :] + U_raw[:, :, 1, :]
    )
    U_full[:, :, np.arange(0, N_x - 1, 1), np.arange(1, N_x, 1)] = U_raw[:, :, 0, 1:]
    U_full[:, :, np.arange(1, N_x, 1), np.arange(0, N_x - 1, 1)] = U_raw[:, :, 1, :-1]
    prob = expm(U_full * dt)
    return prob


def create_R(ntau: int, prop: np.ndarray):
    """create full 2d transition matrix

    Parameters
    ----------
    ntau : int
        tau_n
    prop : np.ndarray
        small 4D version of matrix

    Returns
    -------
    scipy.sparse.csr_array
        The full transition matrix
    np.ndarray
        All states
    np.ndarray
        All end states
    """
    N_x = prop.shape[-1]
    all_states = np.arange(0, N_x ** (ntau + 1), dtype=int)

    # staetes  1 * x(t-tau), N_x*x(t-tau1+1dt), ... , N_x**ntau * x(t)
    l_t_f = (all_states // N_x) % N_x
    l_t_i = all_states % N_x
    l_i = all_states // N_x ** (ntau)  # t state

    all_next_states = (all_states // N_x)[:, None] + (
        (N_x**ntau) * (np.arange(0, N_x))
    )[None, :]

    R = coo_array(
        (
            prop[l_t_f, l_t_i, :, l_i].flatten(),
            (all_next_states.flatten(), all_states.repeat(N_x)),
        ),
        shape=(N_x ** (ntau + 1), N_x ** (ntau + 1)),
    )

    R = csr_array(R)
    end_states = np.stack(
        [all_states[all_states // N_x**ntau == i] for i in range(N_x)]
    )
    return R, all_states, end_states


def get_dyn(
    R: np.ndarray, i_zero: int, N_t: int, N_x: int, ntau: int, end_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interativly solving the equation p = R @ p,
    with p initialized as initial state and saving
    one dimensional probability for every step.

    Parameters
    ----------
    R : np.ndarray
        The 2d rate matrix
    i_zero : int
        Initial condition
    N_t : int
        Number of time steps
    N_x : int
        number of spatial steps
    ntau : int
        number of temporal discretisation steps
    end_states : np.ndarray
        array conating all endstates

    Returns
    -------
    p: np.ndarray
        the final complet statevector
    one_time_p: np.ndarray
        The one time probaility function for every timestep
    """
    initial_state = np.sum(i_zero * (N_x ** np.arange(0, ntau + 1)))
    p = np.zeros(N_x ** (ntau + 1), dtype=float)
    p[initial_state] = 1.0
    one_time_p = np.empty((N_t, N_x))
    one_time_p[0] = np.sum(p[end_states], axis=1)
    for i in range(1, N_t):
        p = R @ p
        one_time_p[i] = np.sum(p[end_states], axis=1)

    return p, one_time_p
