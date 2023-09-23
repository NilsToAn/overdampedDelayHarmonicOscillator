import numpy as np
from scipy.linalg import expm
from tqdm.notebook import tqdm
from scipy.sparse import dok_array

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


def get_rate(hists,x_s, dt,total_props=None):
    if total_props is None:
        sur_l = np.sum(hists[:,x_s <= 0],axis = 1)/np.sum(hists[0])
        sur_r = np.sum(hists[:,x_s > 0],axis = 1)/np.sum(hists[0])
    else:
        sur_l = np.sum(hists[:,x_s <= 0],axis = 1)/np.sum(hists[0]) * np.cumprod(total_props[1:])
        sur_r = np.sum(hists[:,x_s > 0],axis = 1)/np.sum(hists[0]) * np.cumprod(total_props[1:])
    rate = -np.diff(sur_l)/(sur_l[:-1]-sur_r[:-1])/dt
    return sur_l, rate


def get_hists_v4(N_t, N_x, ntau, i_zero, prop, thresh =  1e-7, return_final_states = False):
    def filter_prob(p):
        if p > thresh:
            return 1.
        return p/thresh
    filter_prob = np.vectorize(filter_prob)

    states = np.array([np.sum(i_zero*(N_x**np.arange(ntau+1, dtype = int)))])#np.full((1,ntau+1),i_zero,dtype=int)
    probs = dok_array((N_x**(ntau+1),1), dtype=np.float32)
    probs[states] = 1.
    #probs = np.array([1.])
    total_prob = [1]
    unaltered_props = [1]

    hists = np.empty((N_t,N_x))

    for i in tqdm(np.arange(N_t)):
        new_states = states//N_x
        s0 = states%N_x
        a_s = np.swapaxes(prop,1,2)[s0+new_states%N_x,states//(N_x**ntau),:]

        total_probs = a_s*probs[states].todense()
        #print(total_probs.shape)
        unaltered_props.append(np.sum(total_probs))
        probs.clear()
        
        new_state_list = []
        for s in range(N_x):
            s0filter = s == s0
            i_s = np.where(total_probs[s0filter] > thresh)
            if len(i_s[0]) > 0:
                #i_s = np.where(total_probs > thresh*np.random.rand(*total_probs.shape))
                #new_probs = total_probs[i_s]/filter_prob(total_probs[i_s])
                l_new_states = np.repeat(new_states[s0filter,None],N_x, axis = 1,)[i_s]
                l_new_states += i_s[1]*(N_x**ntau)
                #print(l_new_states)
                #print(~(probs[l_new_states,[0]] != 0).todense()[0])
                new_state_list.append(l_new_states[~(probs[l_new_states,[0]] != 0).todense()[0]])
                probs[l_new_states,[0]] += total_probs[s0filter][i_s]
        


        #states = uni
        states = np.concatenate(new_state_list)
        total_prob.append(np.sum(probs))
        probs = probs/np.sum(probs)

        hists[i] = [np.sum(probs[states[states//(N_x**ntau) == i]]) for i in range(N_x)] 
    print('Total number of final states:', states.shape)
    if return_final_states:
        return hists, unaltered_props, total_prob, new_states
    return hists, unaltered_props, total_prob


def get_prop_v2(x_s, force,D, dt, dx):
    N_x = len(x_s)
    half_x_s = np.arange(x_s[0],x_s[-1]+dx/4,dx/2)
    # x(t-tau), x(t), res
    R = np.zeros(( len(half_x_s),N_x, N_x))
    
    F = force(half_x_s)
    lp = D / dx**2 * np.exp((F*dx/D)/2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(F*dx/D)/2)  # r_i+1->i

    R[:,0,0] = -lp
    R[:,-1,-1] = -ln
    R[:,np.arange(1,N_x-1),np.arange(1,N_x-1)] = -(lp[:,None]+ln[:,None]) # -(r_i->i+1 + r_i->i-1) ????
    R[:,np.arange(0,N_x-1),np.arange(1,N_x)] = ln[:,None]
    R[:,np.arange(1,N_x),np.arange(0,N_x-1)] = lp[:,None]
    prop = expm(R*dt, )
    return prop

def get_prop_abs_v2(x_s, force, D,dt,dx, N_border):
    # x(t-tau), x(t), res
    N_x = len(x_s)
    half_x_s = np.arange(x_s[0],x_s[-1]+dx/4,dx/2)
    
    R_abs = np.zeros(( len(half_x_s),N_x, N_x))
    
    F = force(half_x_s)
    lp = D / dx**2 * np.exp((F*dx/D)/2)  # r_i->i+1
    ln = D / dx**2 * np.exp(-(F*dx/D)/2)  # r_i+1->i
    
    R_abs[:,0,0] = -lp
    R_abs[:,np.arange(1,N_border+1),np.arange(1,N_border+1)] = -(lp[:,None]+ln[:,None]) # -(r_i->i+1 + r_i->i-1) ????
    R_abs[:,np.arange(0,N_border),np.arange(1,N_border+1)] = ln[:,None]
    R_abs[:,np.arange(1,N_border),np.arange(0,N_border-1)] = lp[:,None]
    prop_abs = expm(R_abs*dt, )
    return prop_abs