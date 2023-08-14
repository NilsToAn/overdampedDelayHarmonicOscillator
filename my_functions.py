import torch
import numpy as np
from scipy.special import factorial
tensor = torch.tensor

# only for bullerjahn function
import matplotlib.pyplot as plt
from scipy.special import factorial, lambertw, erf


# Changed initial conditions (Right?)
# vel[:n_tau+1] = ....
# pos[:n_tau+1] = ....
# To 
# vel[:n_tau] = ....
# pos[:n_tau] = ....

def get_pos(
        num_particle,
        max_t,
        dt,
        k,
        tau,
        D,
        F=np.vectorize(lambda t: 0)):
    """

    Parameters
    ----------
    num_particle:int
    max_t:float
    dt:float
    k:float
    tau:float
    F:function

    Returns
    -------
    pos: torch.tensor
    
    Example
    -------
    get_pos(
        num_particle=10000,
        max_t=5,
        dt=1e-3,
        k=10,
        tau=0,
        D = 1e-6,
        F=np.vectorize(lambda t: 0))
    """
    N = int(max_t / dt)
    n_tau = int(tau / dt)
    print(n_tau)
    dims = (N, num_particle, 1)

    pos = torch.empty(dims, dtype=torch.float) # dtype=torch.float
    random_force = np.sqrt(2*D)* torch.normal(0, 1, dims)

    # without other forces
    vel = random_force * np.sqrt(1 / dt)

    # Set inital values
    vel[:n_tau] = torch.zeros((n_tau, num_particle, 1))
    pos[:n_tau] = torch.zeros((n_tau, num_particle, 1))
    
    #print(tau)
    #print('-----')
    #print(pos[:100].mean(1))
    
    for i in range(n_tau + 1, N):
        pos[i] = pos[i - 1] + vel[i - 1] * dt
        vel[i] += -k * pos[i - n_tau] + F(i * dt - tau)
    

    #print(pos[:100].mean(1))
    #print('-----')
    
    return pos

def get_pos_mirror(
        num_particle,
        max_t,
        dt,
        k,
        tau,
        D,
        x_m,
        F=np.vectorize(lambda t: 0)):
    """

    Parameters
    ----------
    num_particle:int
    max_t:float
    dt:float
    k:float
    tau:float
    F:function

    Returns
    -------
    pos: torch.tensor
    
    Example
    -------
    get_pos_mirror(
        num_particle=10000,
        max_t=5,
        dt=1e-3,
        k=10,
        tau=0,
        D = 1e-6,
        x_m=1.5e-3,
        F=np.vectorize(lambda t: 0))
    """
    N = int(max_t / dt)
    n_tau = int(tau / dt)
    dims = (N, num_particle, 1)

    pos = torch.empty(dims, dtype=torch.float) #dtype=torch.float16
    random_force = np.sqrt(2*D)* torch.normal(0, 1, dims)

    # without other forces
    vel = random_force * np.sqrt(1 / dt)

    # Set inital values
    vel[:n_tau] = torch.zeros((n_tau, num_particle, 1))
    pos[:n_tau] = torch.ones((n_tau, num_particle, 1))*-x_m
    
    #print(tau)
    #print('-----')
    #print(pos[:100].mean(1))
    
    for i in range(n_tau + 1, N):
        pos[i] = pos[i - 1] + vel[i - 1] * dt
        vel[i] += -k * (pos[i - n_tau]-pos[i - n_tau].sign()*x_m) + F(i * dt - tau)
    

    #print(pos[:100].mean(1))
    #print('-----')
    
    return pos



def get_mean_std(tau,
                 k,
                 num_particle,
                 max_t,
                 dt,
                 D,
                 F=lambda i: 0,
                 mirrored = False):
    """

    Parameters
    ----------
    tau:float
    k:float
    F:function
    num_particle:int
    max_t:float
    dt:float

    Returns
    -------
    mean, u_mean, std, u_std: torch.tensor,torch.tensor,torch.tensor,torch.tensor
    
    Example
    -------
    get_mean_std(tau,
                 k,
                 num_particle=10000,
                 max_t=5,
                 dt=1e-3,
                 D=1e-6,
                 F=lambda i: 0,
                 mirrored = False)
    """
    F = np.vectorize(F)
    if mirrored:
        pos = get_pos_mirror(num_particle=num_particle, max_t=max_t, tau=tau, F=F, dt=dt, k=k, D=D)
    else:    
        pos = get_pos(num_particle=num_particle, max_t=max_t, tau=tau, F=F, dt=dt, k=k, D=D)
    indx = np.linspace(0, len(pos), 10, dtype=int)
    means = torch.cat([pos[:, i:j].mean(axis=-2) for i, j in zip(indx[:-1], indx[1:])], dim=1)
    mean = means.mean(axis=1)
    u_mean = means.std(axis=1)
    stds = torch.cat([pos[:, i:j].std(axis=-2) for i, j in zip(indx[:-1], indx[1:])], dim=1)
    std = stds.mean(axis=1)
    u_std = stds.std(axis=1) / np.sqrt(stds.shape[1])
    return mean, u_mean, std, u_std


class FokkerPlankCalculator:
    def __init__(self):
        def l(k, tau, t, max_p=40):
            i = np.arange(0, max_p, 1)
            return np.sum((-k) ** i / factorial(i) * (t - i * tau) ** i * np.heaviside(t - i * tau, 1))
        self.l = np.vectorize(l, )

    def get_v_n(self,l_data, ts, max_t, s=1):
        dt = max_t / len(l_data)

        def int2(t, dt):
            nt = int(t / dt)
            return np.sum(l_data[:nt] ** 2) * dt

        int2 = np.vectorize(int2)
        return s ** 2 * int2(ts, dt)

    def get_G_o(self,l_data, l_data_, ts, F_data, max_t):
        dt = max_t / len(F_data)

        def int3(t, dt):
            nt = int(t / dt)
            return np.sum(l_data[nt::-1] * F_data[:nt + 1]) * dt

        int3 = np.vectorize(int3)
        return l_data_ * np.diff(1 / l_data * int3(ts, dt)) / dt
    
    
    
    def get_G(self,l_data, l_data_, ts, F_data, max_t):
        dt = max_t / len(l_data_)
        w = self.get_w(l_data,l_data_, max_t)
        mu = self.get_M(l_data, F_data, ts, max_t)
        
        def int3(t, dt):
            nt = int(t / dt)
            return np.sum(F_data[:nt] * np.diff(l_data[nt::-1])/dt) * dt

        int3 = np.vectorize(int3)
        print(len(int3(ts, dt)[:-1]))
        return - w * mu[:-1] + F_data[:-1] + int3(ts, dt)[:-1]

    def get_M_old(self,l_data, l_data_, G_data, ts_, max_t):
        dt = max_t / len(l_data)

        def int4(t, dt):
            nt = int(t / dt)
            return np.sum(G_data[:nt] / l_data_[:nt]) * dt

        int4 = np.vectorize(int4)
        return l_data_ * int4(ts_, dt)
    
    def get_M(self,l_data, F_data, ts, max_t):
        dt = max_t / len(l_data)

        def int5(t, dt):
            nt = int(t / dt)
            return np.sum(F_data[:nt + 1] * l_data[nt::-1]) * dt

        int5 = np.vectorize(int5)
        return int5(ts, dt)
    
    def get_w(self,l_data,l_data_,max_t):
        dt = max_t/len(l_data_)
        return - np.diff(l_data)/l_data_/dt
    
    # def get_D(self,l_data, l_data_, ts, max_t, D):
    #     dt = max_t / len(l_data)
    #     def int6(t, dt):
    #         nt = int(t / dt)
    #         return np.sum(l_data[:nt]**2) * dt
    #     int6 = np.vectorize(int6)
    #     return D*l_data_**2*np.diff(1/l_data**2 * int6(ts, dt))/dt
    
    def get_D(self,l_data, l_data_, ts, max_t, D):
        dt = max_t / len(l_data)
        def int6(t, dt):
            nt = int(t / dt)
            return np.sum(l_data[:nt]**2) * dt
        int6 = np.vectorize(int6)
        return D*(-2*np.diff(l_data)/dt/l_data_ * int6(ts,dt)[:-1] + l_data_**2)
    
        

    def get_fp_mean_std(
            self,
            tau,
            k,
            max_t,
            s,
            get_F=lambda i: 0):
        """

        Parameters
        ----------
        tau
        k
        max_t
        s
        get_F

        Returns
        -------
        ts, M_data, ts_, v_new, l_data, F_data
        """
        
        ts_all = np.linspace(0, max_t, 6999)
        ts = ts_all[::2]
        ts_ = ts_all[1::2]


        if tau > 0:
            max_p = int(max_t / tau) + 1
            l_data_all = self.l(k, tau, ts_all, max_p=max_p)
        else:
            l_data_all = np.exp(-k*ts_all)
        l_data = l_data_all[::2]
        l_data_ = l_data_all[1::2]

        # Varianz
        # D_data = get_D(l_data,l_data_,ts, max_t = max_t, s = s)
        # v_old = get_v(l_data, l_data_, ts_, D_data, max_t)
        v_new = self.get_v_n(l_data, ts, max_t, s=s)

        # Mean
        F_data = get_F(ts)
        #G_data = self.get_G(l_data, l_data_, ts, F_data, max_t)
        M_data = self.get_M(l_data, F_data, ts, max_t)

        return ts, M_data, ts_, v_new, l_data, F_data


    # def get_D(l_data, l_data_, ts, max_t, s):
    #     dt = max_t/len(l_data)
    #     def int1(t,dt):
    #         nt = int(t/dt)
    #         return 1/l_data[nt-1]**2*np.sum(l_data[:nt]**2)*dt
    #     int1 = np.vectorize(int1)
    #     diff1 = np.diff(int1(ts,dt))/dt
    #     return s/2*l_data_**2*diff1

    # def get_v(l_data, l_data_, ts_, D_data, max_t):
    #     dt = max_t/len(D_data)
    #     def int2(t,dt):
    #         nt = int(t/dt)
    #         return 2*np.sum(D_data[:nt]/l_data_[:nt]**2)*dt
    #     int2 = np.vectorize(int2)
    #     return l_data_**2*int2(ts_,dt)
    
    
def plot_bullerjahn_analysis(max_t, dt, tau, k, D,b, sim_time_s, sim_res,sim_time, sim_res_num, forces, save_run, use_eq_D = False):
    my_fpc = FokkerPlankCalculator()
    s = np.sqrt(2*D)

    ts_all = np.arange(0, max_t+dt, dt)
    ts = ts_all[::2]
    ts_ = ts_all[1::2]


    if tau > 0:
        max_p = int(max_t / tau) + 1
        print(max_p)
        l_data_all = my_fpc.l(k, tau, ts_all, max_p=max_p)
    else:
        l_data_all = np.exp(-k*ts_all)
    l_data = l_data_all[::2]
    l_data_ = l_data_all[1::2]

    w = my_fpc.get_w(l_data, l_data_, max_t)
    D_t_raw = my_fpc.get_D(l_data, l_data_, ts, max_t,D)
    if use_eq_D:
        D_t = D
    else:
        D_t = D_t_raw #D #D_t_raw #D#np.mean(D_t_raw)

    fig, axs = plt.subplots(1,2, figsize=(8, 4))
    axs[0].plot(ts_,w, max_t)
    axs[0].set_ylim(-30,30)
    axs[0].set_title(r"$\omega_\tau(t)$")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(r"$\omega_\tau(t)$")


    axs[1].plot(ts_,D_t_raw)
    axs[1].set_ylim(-1e-5,1e-5)
    axs[1].set_title(r"$D_\tau(t)$")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel(r"$D_\tau(t)$")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_run:
        plt.savefig(pic_path/f'omega_diff.pdf')

    bullerjahn = {}
    for prot in sim_res:
        my_F = np.vectorize(forces[prot])  #lambda i: 0 
        F_data = my_F(ts)
        F_data_ = my_F(ts_)
        G = my_fpc.get_G(l_data,l_data_,ts, F_data, max_t)

        mu = my_fpc.get_M(l_data_,F_data_, ts_, max_t)

        var = my_fpc.get_v_n(l_data_, ts_, max_t, s = s)

        d_prefactor = 4 # Bullerjahn 2, in approx 4

        # TODO: überprüfen !!!!! vorzeichen G ? -> Müsste so stimme
        j = -(w*b-G-d_prefactor*D_t*(b-mu)/var) * 1 / np.sqrt(2*np.pi*var) * np.exp(-(b-mu)**2/(2*var))

        S = 1/2*(1+erf( (b-mu)/np.sqrt(2*var)))

        kap = j/S

        bullerjahn[prot] = kap


        fig, axs = plt.subplots(2,3, figsize=(12, 8))
        axs[0,0].plot(ts_,G)
        axs[0,0].set_title(r"$G$")
        axs[0,0].set_ylim(0.9*np.sort(G)[11], 1.1*np.sort(G)[-11])
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel("G")

        axs[0,1].plot(ts_,mu)
        axs[0,1].set_title(r"$\mu$")
        axs[0,1].set_xlabel("Time")
        axs[0,1].set_ylabel("$\mu$")

        axs[0,2].plot(ts_,var)
        axs[0,2].set_title(r"$Var$")
        axs[0,2].set_xlabel(r"Time")
        axs[0,2].set_ylabel(r"$\sigma^2$")

        #axs[1,2].plot(ts_,(1-S))
        #axs[1,2].set_title(r"$(1-S)$")
        #axs[1,2].plot(sim_time, sim_res[prot])

        axs[1,0].plot(ts_,j)
        axs[1,0].set_ylim(np.sort(j)[4]*0.9, 1.1*np.sort(j)[-4])
        axs[1,0].set_title(r"$J$")
        axs[1,0].set_xlabel(r"Time")
        axs[1,0].set_ylabel(r"$j$")

        axs[1,1].plot(ts_,S)
        axs[1,1].plot(sim_time-tau, sim_res_num[prot])
        axs[1,1].set_title(r"$S$")
        axs[1,1].set_xlabel(r"Time")
        axs[1,1].set_ylabel(r"$S$")


        axs[1,2].plot(sim_time_s-tau, sim_res[prot], label='Simulation')

        min_y = min(np.sort(kap)[100]*0.9, sim_res[prot].min()*0.9)
        max_y = max(np.sort(kap)[-100]*1.1, sim_res[prot].max()*1.1)
        axs[1,2].plot(ts_, kap, label='Bullerjahn')
        
        axs[1,2].set_ylim(min_y,max_y)

        axs[1,2].set_title(r"$\kappa$")
        axs[1,2].legend()
        axs[1,2].set_xlabel(r"Time")
        axs[1,2].set_ylabel(r"$\kappa$")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout()
        if save_run:
            plt.savefig(pic_path/f'escape_rate_{prot}.pdf')
    return bullerjahn    


def get_approx_freq(tau,k):
    return 1/tau*(np.pi/3-1/3*1/(tau*k)+np.sqrt((1/3*1/(k*tau)+np.pi/6)**2-1/3*np.pi/(k*tau)+2/3))


def get_zero(eq,min_x,max_x):
    alpha = np.linspace(min_x,max_x,1000)
    eq_data = eq(alpha)
    return alpha[np.where(eq_data[1:]*eq_data[:-1] < 0)][0]


def damped_harmonic(phi,alpha,w,t, A = 1):
    return A*np.exp(-alpha*t)*np.cos(w*t+phi)


def get_c1(t1,tau,k):
    k1 = np.arange(0, int(t1/tau)+1)
    return np.sum((-k)**k1*1/factorial(k1)*(t1-k1*tau)**k1)
get_c1 = np.vectorize(get_c1)


def get_c2(t1,tau,k):
    k1 = np.arange(1, int(t1/tau)+1)
    return np.sum((-k)**k1*1/factorial(k1-1)*(t1-k1*tau)**(k1-1))
get_c2 = np.vectorize(get_c2)


def get_c3(t1,tau,k):
    k1 = np.arange(2, int(t1/tau)+1)
    return np.sum((-k)**k1*1/(2*factorial(k1-2))*(t1-k1*tau)**(k1-2))
get_c3 = np.vectorize(get_c3)


def get_c4(t1,tau,k):
    k1 = np.arange(3, int(t1/tau)+1)
    return np.sum((-k)**k1*1/(6*factorial(k1-3))*(t1-k1*tau)**(k1-3))
get_c4 = np.vectorize(get_c4)


def get_ham_damp_par(t1,tau,k):
    c1,c2,c3,c4 = get_c1(t1,tau,k), get_c2(t1,tau,k), get_c3(t1,tau,k), get_c4(t1,tau,k)

    def eq_alpha(a):
        return -4*a**3*c1+4*a**2*c2+(4*c3-c2**2/c1)*a-c3*c2/c1+3*c4

    def get_w(alpha):
        return np.sqrt(3 * alpha ** 2 - 2 * alpha * c2 / c1 - 2 * c3 / c1)

    def get_phi(a,w):
        return np.arctan(-(a*c1+c2)/(w*c1))-w*t1

    def get_A(phi,a,w):
        return c1*np.exp(a*t1)/np.cos(w*t1+phi)

    alp = get_zero(eq_alpha, -5,5)
    w = get_w(alp)
    phi = (get_phi(alp,w)%(np.pi))
    A = get_A(phi, alp,w)
    return [t1,alp,w,phi,A]


# Deterministic
def damped_harmonic_oszillator(
        x0,
        max_t,
        dt,
        k,
        gamma,
        m,
        F=np.vectorize(lambda t: 0)):
    """

    Parameters
    ----------
    num_particle:int
    max_t:float
    dt:float
    k:float
    tau:float
    F:function

    Returns
    -------
    pos: torch.tensor
    """
    N = int(max_t / dt)
    
    dims = (N, 1)

    pos = torch.empty(dims, dtype=torch.float) #dtype=torch.float16
    vel = torch.empty(dims, dtype=torch.float)
    acc = torch.empty(dims, dtype=torch.float)

   

    # Set inital values
    vel[0] = torch.zeros((1, 1))
    pos[0] = torch.full((1, 1),x0)
    acc[0] = torch.zeros((1, 1))

    for i in range(1, N):
        vel[i] = vel[i - 1] + acc[i - 1] * dt
        pos[i] = pos[i - 1] + vel[i - 1] * dt
        acc[i] = 1/m * (-k*pos[i] - gamma*vel[i] + F(i * dt))

    return pos


def time_delayed_harmonic(
        x0,
        max_t,
        dt,
        k,
        tau,
        gamma,
        F=np.vectorize(lambda t: 0)):
    """

    Parameters
    ----------
    num_particle:int
    max_t:float
    dt:float
    k:float
    tau:float
    F:function

    Returns
    -------
    pos: torch.tensor
    
    Example
    -------
    time_delayed_harmonic(
        x0 = 0,
        max_t=5000,
        dt=5,
        k=0.005,
        tau=0,
        gamma = 1,
        F=np.vectorize(lambda t: 0))
    """
    N = int(max_t / dt)
    n_tau = int(tau / dt)

    dims = (N, 1)

    pos = torch.empty(dims, dtype=torch.float)# dtype=torch.float16
    vel = torch.empty(dims, dtype=torch.float)

    # Set inital values
    vel[:n_tau] = torch.zeros((n_tau, 1))
    pos[:n_tau] = torch.full((n_tau, 1), x0)

    for i in range(n_tau + 1, N):
        pos[i] = pos[i - 1] + vel[i - 1] * dt
        vel[i] = 1/gamma*(-k * pos[i - n_tau] + F(i * dt - tau))

    return pos[n_tau:]



def get_pos_with_border(
    border,
    **args):
    """

    Parameters
    ----------
    border:float
    **args: dict


    Returns
    -------
    pos: torch.tensor
    
    Example
    -------
    get_pos_with_border(
        border = 8e-4,
        **args)
    """
    
    pos = get_pos(**args)
    for i in range(pos.shape[0]):
        one_timestep = pos[i]
        escaped = (one_timestep > border)
        pos[i:,escaped] = float('nan')
    return pos