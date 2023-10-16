import numpy as np
from scipy.linalg import expm
from tqdm.notebook import tqdm
from scipy.sparse import csr_array, coo_array
from scipy.special import factorial

#### Nummerical (new)

def get_prop_abs(x_s, force, D,dt,dx, N_border=None):
    # x(t-tau), x(t), res
    N_x = len(x_s)
    R_abs = np.zeros(( N_x,N_x, N_x))
    
    
    F = force(x_s)
    lp = D / dx**2 * np.exp((F*dx/D)/2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(F*dx/D)/2)  # r_i+1->i
    
    R_abs[:,np.arange(0,N_x),np.arange(0,N_x)] = -(lp[:,None]+ln[:,None]) # -(r_i->i+1 + r_i->i-1) ????
    R_abs[:,np.arange(0,N_x-1),np.arange(1,N_x)] = ln[:,None]
    R_abs[:,np.arange(1,N_x),np.arange(0,N_x-1)] = lp[:,None]
    prop_abs = expm(R_abs*dt, )
    return prop_abs


def get_prop_abs_v2(x_s, force, D,dt,dx, N_border=None, side = 'lr'):
    # x(t-tau), x(t), res
    N_x = len(x_s)
    half_x_s = np.arange(x_s[0],x_s[-1]+dx/4,dx/2)
    
    
    
    R_abs = np.zeros(( len(half_x_s),N_x, N_x))
    
    F = force(half_x_s)
    lp = D / dx**2 * np.exp((F*dx/D)/2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(F*dx/D)/2)  # r_i+1->i
    if side == 'r':
        if N_border is None:
            N_border = N_x
        R_abs[:,0,0] = -lp
        R_abs[:,np.arange(1,N_border),np.arange(1,N_border)] = -(lp[:,None]+ln[:,None]) # -(r_i->i+1 + r_i->i-1) ????
        R_abs[:,np.arange(0,N_border-1),np.arange(1,N_border)] = ln[:,None]
        R_abs[:,np.arange(1,N_border),np.arange(0,N_border-1)] = lp[:,None]
    elif side == 'l':
        if N_border is None:
            N_border = 0
        R_abs[:,-1,-1] = -ln
        R_abs[:,np.arange(N_border,N_x-1),np.arange(N_border,N_x-1)] = -(lp[:,None]+ln[:,None]) # -(r_i->i+1 + r_i->i-1) ????
        R_abs[:,np.arange(N_border,N_x-1),np.arange(N_border+1,N_x)] = ln[:,None]
        R_abs[:,np.arange(N_border+1,N_x),np.arange(N_border,N_x-1)] = lp[:,None]
    elif side == 'lr':
        if N_border is not None:
            print('for lr N_border is ignored')
        R_abs[:,np.arange(0,N_x),np.arange(0,N_x)] = -(lp[:,None]+ln[:,None]) # -(r_i->i+1 + r_i->i-1) ????
        R_abs[:,np.arange(0,N_x-1),np.arange(1,N_x)] = ln[:,None]
        R_abs[:,np.arange(1,N_x),np.arange(0,N_x-1)] = lp[:,None]
    prop_abs = expm(R_abs*dt, )
    if np.any(np.isnan(prop_abs)):
        print('CAREFUL: nan in prop, maybe because of to high values in potential')
    return prop_abs


def create_R(N_x, ntau, prop):
    all_states = np.arange(0, N_x**(ntau+1))

    # staetes # 1 * x(t-tau), N_x*x(t-tau1+1dt), ... , N_x**ntau * x(t)

    lm = all_states%N_x #mean t-tau state
    lt = all_states//N_x**(ntau)             # t state
    all_next_states = (all_states//N_x)[:,None] + ((N_x**ntau)*(np.arange(0,N_x)))[None,:]

    R = coo_array( (prop[lm,:,lt].flatten(),
                    ( all_next_states.flatten(), all_states.repeat(N_x))) 
                  , shape = (N_x**(ntau+1),N_x**(ntau+1)))
    
    R = csr_array(R)
    end_states = np.stack([all_states[all_states//N_x**ntau == i] for i in range(N_x)])
    return R, all_states, end_states


def create_R_v1(N_x, ntau, prop):
    all_states = np.arange(0, N_x**(ntau+1))

    # staetes # 1 * x(t-tau), N_x*x(t-tau1+1dt), ... , N_x**ntau * x(t)

    lm = all_states%N_x + (all_states//N_x)%N_x  #mean t-tau state
    lt = all_states//N_x**(ntau)             # t state
    all_next_states = (all_states//N_x)[:,None] + ((N_x**ntau)*(np.arange(0,N_x)))[None,:]

    R = coo_array( (prop[lm,:,lt].flatten(),
                    ( all_next_states.flatten(), all_states.repeat(N_x))) 
                  , shape = (N_x**(ntau+1),N_x**(ntau+1)))
    
    R = csr_array(R)
    end_states = np.stack([all_states[all_states//N_x**ntau == i] for i in range(N_x)])
    return R, all_states, end_states


def get_dyn_v2(R, i_zero, N_t,N_x, ntau, end_states):
    initial_state = np.sum(i_zero*(N_x**np.arange(0,ntau+1)))
    p = np.zeros(N_x**(ntau+1), dtype = float)
    p[initial_state] = 1.
    one_time_p = np.empty((N_t,N_x))
    one_time_p[0] = np.sum(p[end_states], axis = 1)
    for i in tqdm(range(1,N_t), leave=False):
        p = R@p
        one_time_p[i] = np.sum(p[end_states], axis = 1)
    
    return p, one_time_p

def get_non_delayed_prop(x_s, force, D,dt,dx, N_border=None, Fp = None, Fl = None):
    N_x = len(x_s)
    mx_s = np.arange(x_s[0]-dx/2,x_s[-1]+dx/2+dx/4,dx)
    if (Fp is None or Fl is None):
        F = force(mx_s)
    if Fp is None:
        Fp = F
    if Fl is None:
        Fl = F
    R_abs = np.zeros(( N_x, N_x))
    
    lp = D / dx**2 * np.exp((Fp*dx/D)/2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(Fl*dx/D)/2)  # r_i+1->i
    
    R_abs[np.arange(0,N_x),np.arange(0,N_x)] = -(lp[1:]+ln[:-1]) # -(r_i->i+1 + r_i->i-1) ????
    R_abs[np.arange(0,N_x-1),np.arange(1,N_x)] = ln[1:-1]
    R_abs[np.arange(1,N_x),np.arange(0,N_x-1)] = lp[1:-1]
    prop_abs = expm(R_abs*dt, )
    return prop_abs

def get_non_delayed_dyn(R, i_zero, N_t,N_x):
    p = np.zeros((N_t,N_x), dtype = float)
    p[0,i_zero] = 1.
    if R.ndim == 2:
        for i in tqdm(range(1,N_t), leave=False):
            p[i] = R@p[i-1]    
        return p
    else:
        for i in tqdm(range(1,N_t), leave=False):
            p[i] = R[i-1]@p[i-1]    
        return p
        
    

#### Analytical

# OLD !?
# def get_p_x4_short_time(x,k,tau,s):
#     ga = s*(1+3*tau*k*x**2)
#     p_not_notmed = np.exp(-x**2/(3*tau*s**2)+(1-18*tau**2*k*s**2)/(9*tau**2*k*s**2)*np.log(1+3*tau*k*x**2))
#     return p_not_notmed/np.sum(p_not_notmed)

def get_p_x4_short_time(x,tau):
    res = 1 / (1 + 3 * tau * x**2)**2 * np.exp( - ( (1/(3*tau*x**2 + 1) + np.log(3*tau*x**2+1)) / (9*tau**2) ))
    return res/res.sum()


def get_x2_var(tau,k,s):
    return s**2/(2*k)*(1+np.sin(k*tau))/np.cos(k*tau)

def get_x2_var_short_time(tau,k,s):
    return s**2/(2*k)*(1+k*tau)


# Theory

# WRONG !!!
def summand(t,l1,l2, tau,k = 1): 
    return (-k)**(l1+l2) / (factorial(l1)*factorial(l2)) * \
        (1/(l1 + 1) * (t-l1*tau)**(l1+1) * (t-l2*tau)**l2 + \
         1/(l2 + 1) * (t-l2*tau)**(l2+1) * (t-l1*tau)**l1)

# WRONG !!!!!
def get_theo_var(t, tau, D, k = 1):
    l1 = np.arange(0,t/tau)
    l2 = np.arange(0,t/tau)
    return 2*D*(np.sum(summand(t, l1[:,None], l2[None,:], tau, k = k)))
get_theo_var = np.vectorize(get_theo_var, excluded=(1,2))

def l(k, tau, t, max_p=40):
        i = np.arange(0, max_p, 1)
        return np.sum((-k) ** i / factorial(i) * (t - i * tau) ** i * np.heaviside(t - i * tau, 1))
l = np.vectorize(l, )

def get_theo_var_l(ts, tau, D, k = 1):
    if tau > 0:
        dt = ts[1]-ts[0]
        max_p = np.max(ts)/tau
        l_data = l(k,tau,ts, max_p)
        var = np.zeros(len(ts))
        var[1:] = (2*D*np.cumsum(l_data**2) *dt)[:-1]
        return var
    else:
        return D/k*(1 - np.exp(-2*k*ts))



#### Simulation
def simulate_traj(N_p, N_loop, N_t, ntau, s, dt, border, force):
    pos = np.empty((N_loop,N_p,N_t))
    vel = s*np.random.randn(N_loop,N_p,N_t)*1/np.sqrt(dt)

    pos[:,:,:ntau] = -border
    vel[:,:,:ntau] = 0

    for i in tqdm(range(ntau,N_t), leave=False):
        pos[:,:,i] = pos[:,:,i-1] + vel[:,:,i-1]*dt
        vel[:,:,i] += force(pos[:,:,i-ntau])
    return pos


#### General functions
def get_var_hist(hists, x_s):
    if isinstance(hists, list):
        hists = np.stack(hists)
    if hists.ndim == 2:
        p = hists/np.sum(hists, axis = 1)[:,None]
        return np.sum(p*x_s[None,:]**2 - (p*x_s[None,:])**2, axis = 1)
    if hists.ndim == 1:
        p = hists/np.sum(hists)
        return np.sum(p*x_s**2 - (p*x_s)**2)
    else:
        assert "Wrong number of dim in hists"
        
        
def get_steady_mean(data, i = None, max_err = 1, thresh = 0.1, min_states = 5):
    if i is None:
        i = len(data)-1
    ref_var = data[i]
    test_points = data[i::-1]
    mask = np.cumprod(np.abs(test_points - ref_var) < ref_var*thresh).astype(bool)
    if np.sum(mask) < min_states:
        return False
    steady_points = test_points[mask]
    relerr = np.std(steady_points)/np.sqrt(len(steady_points)) / np.mean(steady_points)
    if relerr < max_err:
        return np.mean(steady_points), np.std(steady_points)/np.sqrt(len(steady_points))
    return False
        
        