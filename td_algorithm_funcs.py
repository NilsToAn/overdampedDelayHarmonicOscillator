import numpy as np
from scipy.linalg import expm
from tqdm.notebook import tqdm

def get_prop(x_s, force,D, dt, dx):
    N_x = len(x_s)
    # x(t-tau), x(t), res
    R = np.zeros(( N_x,N_x, N_x))
    
    F = force(x_s)
    lp = D / dx**2 * np.exp((F*dx/D)/2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(F*dx/D)/2)  # r_i+1->i

    R[:,0,0] = -lp
    R[:,-1,-1] = -ln
    R[:,np.arange(1,N_x-1),np.arange(1,N_x-1)] = -(lp[:,None]+ln[:,None]) # -(r_i->i+1 + r_i->i-1) ????
    R[:,np.arange(0,N_x-1),np.arange(1,N_x)] = ln[:,None]
    R[:,np.arange(1,N_x),np.arange(0,N_x-1)] = lp[:,None]
    prop = expm(R*dt, )
    return prop

def get_prop_abs(x_s, force, D,dt,dx, N_border):
    # x(t-tau), x(t), res
    N_x = len(x_s)
    R_abs = np.zeros(( N_x,N_x, N_x))
    
    F = force(x_s)
    lp = D / dx**2 * np.exp((F*dx/D)/2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(F*dx/D)/2)  # r_i+1->i
    
    R_abs[:,0,0] = -lp
    R_abs[:,np.arange(1,N_border+1),np.arange(1,N_border+1)] = -(lp[:,None]+ln[:,None]) # -(r_i->i+1 + r_i->i-1) ????
    R_abs[:,np.arange(0,N_border),np.arange(1,N_border+1)] = ln[:,None]
    R_abs[:,np.arange(1,N_border),np.arange(0,N_border-1)] = lp[:,None]
    prop_abs = expm(R_abs*dt, )
    return prop_abs


def get_hists(N_t, N_x, ntau, i_zero, prop, thresh =  0.000005, return_final_states = False):

    def filter_prob(p):
        if p > thresh:
            return 1.
        return p/thresh
    filter_prob = np.vectorize(filter_prob)

    states = np.full((1,ntau+1),i_zero,dtype=int)
    probs = np.array([1.])
    total_prob = [1]
    unaltered_props = [1]

    hists = np.empty((N_t,N_x))

    for i in tqdm(np.arange(N_t)):
        a_s = np.swapaxes(prop,1,2)[states[:,0],states[:,-1],:]

        total_probs = a_s*probs[:,None]
        unaltered_props.append(np.sum(total_probs))
        i_s = np.where(total_probs > thresh)
        new_probs = total_probs[i_s]
        #i_s = np.where(total_probs > thresh*np.random.rand(*total_probs.shape))
        #new_probs = total_probs[i_s]/filter_prob(total_probs[i_s])
        new_states = np.repeat(states[:,None,:],N_x, axis = 1,)[i_s]
        new_states[:,:-1] = new_states[:,1:]
        new_states[:,-1]  = i_s[1]


        uni, indxs = np.unique(new_states, axis=0, return_inverse=True)
        states = uni
        new_probs = np.stack([np.sum(new_probs[indxs == i]) for i in range(len(uni))])
        total_prob.append(np.sum(new_probs))
        probs = new_probs/np.sum(new_probs)

        hists[i] = [np.sum(probs[states[:,-1] == i]) for i in range(N_x)] 
    print('Total number of final states:', states.shape)
    if return_final_states:
        return hists, unaltered_props, total_prob, new_states
    return hists, unaltered_props, total_prob

def simulate_traj(N_p, N_loop, N_t, ntau, s, dt, border, force):
    pos = np.empty((N_loop,N_p,N_t))
    vel = s*np.random.randn(N_loop,N_p,N_t)*1/np.sqrt(dt)

    pos[:,:,:ntau+1] = -border
    vel[:,:,:ntau+1] = 0

    for i in tqdm(range(ntau+1,N_t)):
        pos[:,:,i] = pos[:,:,i-1] + vel[:,:,i-1]*dt
        vel[:,:,i] += force(pos[:,:,i-ntau])
    return pos


def get_rate(hists, dt,total_props=None):
    if total_props is None:
        sur_l = np.sum(hists[:,x_s <= 0],axis = 1)/np.sum(hists[0])
        sur_r = np.sum(hists[:,x_s > 0],axis = 1)/np.sum(hists[0])
    else:
        sur_l = np.sum(hists[:,x_s <= 0],axis = 1)/np.sum(hists[0]) * np.cumprod(total_props[1:])
        sur_r = np.sum(hists[:,x_s > 0],axis = 1)/np.sum(hists[0]) * np.cumprod(total_props[1:])
    rate = -np.diff(sur_l)/(sur_l[:-1]-sur_r[:-1])/dt
    return sur_l, rate