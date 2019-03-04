# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:23:32 2018

@author: rodri
"""

# =============================================================================
#  Solves the HH problem under VFI and Simulates the Economy in the scenario of
# acess to a risk-free asset with borrowing constraint.
# =============================================================================
# Case of access to assets market.
# For details of the model see my github:

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from quantecon import compute_fixed_point
import quantecon as qe
from scipy.interpolate import LinearNDInterpolator as linear_interp
import os
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/python')
from data_functions_albert import gini
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import Rbf
import seaborn as sns
from linear_interp_3d import interp_3d_vec

from model_class import HH_Model

#Import model class
cp =HH_Model()
A, B = cp.A, cp.B
α, γ, ρ, β  = cp.α, cp.γ, cp.ρ, cp.β 
N, N_a, N_m, N_θ, N_ε = cp.N, cp.N_a, cp.N_m, cp.N_θ, cp.N_ε
r, p = cp.r, cp.p
b = cp.b

#For b=0:
folder = 'C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/Model assets/no borrowing/'
save = True

print(r<(1/β-1))


a_grid, m_grid, ε_grid, θ_grid = cp.a_grid, cp.m_grid, cp.ε_grid, cp.θ_grid

pi_ε, pi_θ = cp.pi_ε, cp.pi_θ 
u, y1, y2 = cp.u, cp.y1, cp.y2
        

# === Set empty Value and Policy functions: === #
N = cp.N_a*cp.N_m*cp.N_m*cp.N_θ*cp.N_ε
V_new = np.empty((N_a, N_m, N_m, N_θ, N_ε))
policy_a = np.empty((N_a, N_m, N_m, N_θ, N_ε))
policy_m1 = np.empty((N_a, N_m, N_m, N_θ, N_ε))
policy_m2 = np.empty((N_a, N_m, N_m, N_θ, N_ε))
policy_c = np.empty((N_a, N_m, N_m, N_θ, N_ε))


V_0_func = lambda a, m1, m2, θ, ε: u(θ*ε*A*m1**α +ε*B*m2**γ+(1+r)*a -p*(m1 +m2)-a)/(1-β)

def V_0_func_2(a, m1, m2, θ, ε, l, m):
    
    Expt1 = pi_θ[l,0]*pi_ε[m,0]*θ_grid[0]*ε_grid[0] +pi_θ[l,1]*pi_ε[m,0]*θ_grid[1]*ε_grid[0]+pi_θ[l,0]*pi_ε[m,1]*θ_grid[0]*ε_grid[1]+pi_θ[l,1]*pi_ε[m,1]*θ_grid[1]*ε_grid[1]
    Expt2 = pi_ε[m,0]*ε_grid[0] +pi_ε[m,1]*ε_grid[1] 
    gm1 = (p/(Expt1*α*A))**(1/(α-1))
    gm2 = (p/(Expt2*γ*B))**(1/(γ-1))    
    return u(θ*ε*A*20*m1**α +ε*B*20*m2**γ+(1+r)*10*a -p*(gm1 +gm2)-a)/(1-β) +200

#improve v_guess
V_guess = np.zeros(N).reshape(len(a_grid),len(m_grid),len(m_grid),len(θ_grid),len(ε_grid))
'''
for i,a in enumerate(a_grid):
    for j,m1 in enumerate(m_grid):
        for k,m2 in enumerate(m_grid):
            for l,θ in enumerate(θ_grid):
                for m,ε in enumerate(ε_grid):
                    V_guess[i,j,k,l,m] = np.nan_to_num(V_0_func(a, m1, m2, θ, ε))
'''

 

m=np.empty((N_a, N_m, N_m, N_θ, N_ε, N_a, N_m, N_m))

@jit(nopython=True)
def bellman_operator_jit(V): 
    def u(c):
        if ρ==1:
            return np.log(c)
        else:
            return (c**(1-ρ) -1) / (1-ρ)        
    #Given state variables, find optimal choices (g_a, g_m1, g_m2):
    for i_a, a in enumerate(a_grid):
        for i_m1, m1 in enumerate(m_grid):
            for i_m2, m2 in enumerate(m_grid):
                for i1,i_θ in enumerate(θ_grid):
                    for i2,i_ε in enumerate(ε_grid):                              
                        for i_gm1, g_m1 in enumerate(m_grid):
                            for i_gm2, g_m2 in enumerate(m_grid):                           
                                m[i_a,i_m1,i_m2,i1, i2, :,i_gm1,i_gm2] = u(i_θ*i_ε*A*m1**α +i_ε*B*m2**γ +(1+r)*a -p*g_m1 -p*g_m2 -a_grid)+β*(pi_θ[i1,0]*pi_ε[i2,0]*V[:, i_gm1, i_gm2,0,0] +pi_θ[i1,1]*pi_ε[i2,0]*V[:, i_gm1, i_gm2,1,0] +pi_θ[i1,0]*pi_ε[i2,1]*V[:, i_gm1, i_gm2,0,1] +pi_θ[i1,1]*pi_ε[i2,1]*V[:, i_gm1, i_gm2,1,1])   
                                #print(m[i_a,i_m1,i_m2,i1, i2, :,i_gm1,i_gm2])
                            
                        V_new[i_a,i_m1,i_m2,i1,i2] = np.nanmax(m[i_a,i_m1,i_m2, i1,i2,:,:,:])
                       
    return V_new 


def bellman_operator_discrete(V, return_policies=False): 
            
    #Given state variables, find optimal choices (g_a, g_m1, g_m2):
    for i_a, a in enumerate(a_grid):
        for i_m1, m1 in enumerate(m_grid):
            for i_m2, m2 in enumerate(m_grid):
                for i1,i_θ in enumerate(θ_grid):
                    for i2,i_ε in enumerate(ε_grid):                              
                        for i_gm1, g_m1 in enumerate(m_grid):
                            for i_gm2, g_m2 in enumerate(m_grid):                           
                                m[i_a,i_m1,i_m2,i1, i2, :,i_gm1,i_gm2] = u(i_θ*i_ε*A*m1**α +i_ε*B*m2**γ +(1+r)*a -p*g_m1 -p*g_m2 -a_grid)+β*(pi_θ[i1,0]*pi_ε[i2,0]*V[:, i_gm1, i_gm2,0,0] +pi_θ[i1,1]*pi_ε[i2,0]*V[:, i_gm1, i_gm2,1,0] +pi_θ[i1,0]*pi_ε[i2,1]*V[:, i_gm1, i_gm2,0,1] +pi_θ[i1,1]*pi_ε[i2,1]*V[:, i_gm1, i_gm2,1,1])   
                                #print(m[i_a,i_m1,i_m2,i1, i2, :,i_gm1,i_gm2])
                            
                        V_new[i_a,i_m1,i_m2,i1,i2] = np.nanmax(m[i_a,i_m1,i_m2, i1,i2,:,:,:])
                        #print(V_new)
                        
    return V_new 
    
def bellman_operator_policies(V): 
            
    #Given state variables, find optimal choices (g_a, g_m1, g_m2):
    for i_a, a in enumerate(a_grid):
        for i_m1, m1 in enumerate(m_grid):
            for i_m2, m2 in enumerate(m_grid):
                for i1,i_θ in enumerate(θ_grid):
                    for i2,i_ε in enumerate(ε_grid):                              
                        for i_gm1, g_m1 in enumerate(m_grid):
                            for i_gm2, g_m2 in enumerate(m_grid):                           
                                m[i_a,i_m1,i_m2,i1, i2, :,i_gm1,i_gm2] = u(np.fmax(i_θ*i_ε*A*m1**α +i_ε*B*m2**γ +(1+r)*a -p*g_m1 -p*g_m2 -a_grid,1e-3*np.ones(N_a)))+β*(pi_θ[i1,0]*pi_ε[i2,0]*V[:, i_gm1, i_gm2,0,0] +pi_θ[i1,1]*pi_ε[i2,0]*V[:, i_gm1, i_gm2,1,0] +pi_θ[i1,0]*pi_ε[i2,1]*V[:, i_gm1, i_gm2,0,1] +pi_θ[i1,1]*pi_ε[i2,1]*V[:, i_gm1, i_gm2,1,1])   
                                #print(m[i_a,i_m1,i_m2,i1, i2, :,i_gm1,i_gm2])
                            
                        V_new[i_a,i_m1,i_m2,i1,i2] = np.nanmax(m[i_a,i_m1,i_m2, i1,i2,:,:,:])
                        
                        policy_a[i_a,i_m1,i_m2,i1,i2] = a_grid[np.unravel_index(np.argmax(m[i_a,i_m1,i_m2, i1,i2,:,:,:], axis=None), m[i_a,i_m1,i_m2, i1,i2,:,:,:].shape)[0]]
                        policy_m1[i_a,i_m1,i_m2,i1,i2] = m_grid[np.unravel_index(np.argmax(m[i_a,i_m1,i_m2, i1,i2,:,:,:], axis=None), m[i_a,i_m1,i_m2, i1,i2,:,:,:].shape)[1]]
                        policy_m2[i_a,i_m1,i_m2,i1,i2] = m_grid[np.unravel_index(np.argmax(m[i_a,i_m1,i_m2, i1,i2,:,:,:], axis=None), m[i_a,i_m1,i_m2, i1,i2,:,:,:].shape)[2]]
                        policy_c[i_a,i_m1,i_m2,i1,i2] = i_θ*i_ε*A*m1**α +i_ε*B*m2**γ +(1+r)*a -p*policy_m1[i_a,i_m1,i_m2,i1,i2] -p*policy_m2[i_a,i_m1,i_m2,i1,i2] -policy_a[i_a,i_m1,i_m2,i1,i2]
    
    return V_new, policy_a, policy_m1, policy_m2, policy_c

    


qe.tic()
V = compute_fixed_point(bellman_operator_discrete, V_guess, max_iter=1, error_tol=0.1)


V2 = compute_fixed_point(bellman_operator_discrete, V, max_iter=100, error_tol=0.01)
qe.toc()
V_next, g_a, g_m1, g_m2, g_c = bellman_operator_policies(V2)



#%% Asset policy functions across shocks
        
def plot_policy(policy, policy_name, m0=0, save=False, folder=folder):      
    fig,ax = plt.subplots()
    ax.plot(a_grid, np.array(policy[:,m0,m0,1,1]), label=policy_name+r' under $\theta=1, \varepsilon=1)$')
    ax.plot(a_grid, np.array(policy[:,m0,m0,0,1]), label=policy_name+r' under $\theta=0, \varepsilon=1)$')
    ax.plot(a_grid, np.array(policy[:,m0,m0,1,0]), label=policy_name+r' under $\theta=1, \varepsilon=0)$')
    ax.plot(a_grid, np.array(policy[:,m0,m0,0,0]), label=policy_name+r' under $\theta=0, \varepsilon=0)$')
    ax.legend()
    ax.set_xlabel('assets')
    ax.set_title(policy_name)
    plt.show()


# Value function
for m0 in range(N_m-1):
    plot_policy(V_next, policy_name='Value function', m0=m0, save=save)

# Consumption policy
for m0 in range(N_m-1):
    plot_policy(g_c, policy_name='g_c', m0=m0, save=save)

# Assets policy
for m0 in range(N_m-1):
    plot_policy(g_a, policy_name='g_a', m0=m0, save=save)

# Input high policy
for m0 in range(N_m-1):
    plot_policy(g_m1, policy_name='g_m1', m0=m0, save=save)

# Input low policy
for i in range(N_m-1):
    plot_policy(g_m2, policy_name='g_m2', m0=m0, save=save)


        
#%% Asset policy functions across shocks

#3d graphs

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

def plot_policy_3d( policy, policy_name, grid1=m_grid, grid2=m_grid, a0=0):

    x, y = np.meshgrid(m_grid, m_grid)  
    fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
    surf = fig.add_subplot(111, projection='3d')
    surf.plot_surface(x,
                y,
                np.array(policy[a0,:,:,1,1]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='good-good')
    surf.plot_surface(x,
                y,
                np.array(policy[a0,:,:,1,0]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='good-bad')
    surf.plot_surface(x,
                y,
                np.array(policy[a0,:,:,0,1]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='bad-good')
    surf.plot_surface(x,
                y,
                np.array(policy[a0,:,:,0,0]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='bad-bad')          
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d   
    surf.set_title(policy_name)
    if save==True:
        fig.savefig(folder+policy_name+'inputs_2dgrid'+'_'+str(round(a_grid[a0],ndigits=2))+'.png')                 
    return plt.show()


## Plot Value function
for a0 in range(N_a):
    plot_policy_3d(policy=V_new, a0=a0, save=save, policy_name='Value function')

## Plot Value function
    plot_policy_3d(policy=g_c, a0=a0, save=save, policy_name='Consumption Policy')
    
## Plot Value function
    plot_policy_3d(policy=g_a, a0=a0, save=save, policy_name='Insurance Policy')
    
## Plot Value function
    plot_policy_3d(policy=g_m1, save=save, a0=a0, policy_name='Input high Policy')

## Plot Value function
    plot_policy_3d(policy=g_m2, save=save, a0=a0, policy_name='Input low Policy')





#%%
# =============================================================================
# Compute stationary distribution
# =============================================================================
T= 1000
N= 1000
mc_θ = qe.MarkovChain(pi_θ)
mc_ε = qe.MarkovChain(pi_ε)

# == Compute the stationary distribution of Pi == #
θ_0 = mc_θ.stationary_distributions[0]
ε_0 = mc_ε.stationary_distributions[0]

θ_h_0 = θ_0[1] #Proportional of people in good(high) state of rainfall at t=0
ε_h_0 = ε_0[1] #Proportional of people in good(high) state of basis risk at t=0




def invariant_distr(pi_θ, pi_ε, T=1000, N=1000, θ_h_0= 0.5, ε_h_0 = 0.5):
    θ_shock = np.empty(T*N).reshape(T,N)
    ε_shock = np.empty(T*N).reshape(T,N)
    m1_state = np.empty(T*N).reshape(T,N)
    m2_state = np.empty(T*N).reshape(T,N)
    a_state = np.empty(T*N).reshape(T,N)
    y_state = np.empty(T*N).reshape(T,N)
    c_state = np.empty((T,N))
    mean_m1 = np.empty(T)
    mean_m2 = np.empty(T)
    mean_a = np.empty(T)
    mean_y = np.empty(T)
    mean_c = np.empty(T)
    draw = np.random.uniform(size=N)
    θ_shock[0, :] = draw<θ_h_0
    ε_shock[0,:] = draw<ε_h_0
    m1_state[0,:] = m_grid[int(N_m/2)]
    m2_state[0,:] = m_grid[int(N_m/2)]
    a_state[0,:] = a_grid[int(N_a/2)]
    y_state[0,:] = y1(m1_state[0,:],θ_shock[0, :],ε_shock[0,:]) +y2(m2_state[0,:],ε_shock[0,:])
    input1_list = m_grid.tolist()
    input2_list = m_grid.tolist()

    a_list = a_grid.tolist()


    for t in range(1, T):        
        for n in range(0,N):
            draw = np.random.uniform()
            draw2 = np.random.uniform()
            m1_state[t,n] = g_m1[a_list.index(a_state[t-1,n]),input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]
            m2_state[t,n] = g_m2[a_list.index(a_state[t-1,n]),input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]
            a_state[t,n] = g_a[a_list.index(a_state[t-1,n]),input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]          
            c_state[t,n] = g_c[a_list.index(a_state[t-1,n]),input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]             
            θ_shock[t, n] = int(draw < pi_θ[int(θ_shock[t-1, n]),1])
            ε_shock[t,n] =  int(draw2 < pi_ε[int(ε_shock[t-1, n]),1])
            y_state [t,n] = y1(m1_state[t-1,n],θ_grid[int(θ_shock[t-1, n])], ε_grid[int(ε_shock[t-1, n])]) +y2(m2_state[t-1,n], ε_grid[int(ε_shock[t-1, n])])
        
        mean_m1[t] = np.mean(m1_state[t,:]) 
        mean_m2[t] = np.mean(m2_state[t,:])
        mean_a[t] = np.mean(a_state[t,:])
        mean_y[t] = np.mean(y_state[t,:])
        mean_c[t] = np.mean(c_state[t,:])
            
    return θ_shock[T-1,:], ε_shock[T-1,:], m1_state[T-1,:], m2_state[T-1,:] , a_state[T-1,:], y_state[T-1,:], c_state[T-1,:], mean_m1, mean_m2, mean_a, mean_y, mean_c

θ_shock, ε_shock, m1_state, m2_state, a_state, y_state, c_state, mean_m1, mean_m2, mean_a, mean_y, mean_c = invariant_distr(pi_θ, pi_ε, T=T, N=N, θ_h_0=θ_h_0 , ε_h_0 = ε_h_0)






#%% Averages across time and distribution plots

mean_list = [mean_m1, mean_m2, mean_y, mean_a, mean_c]
names_list = ['High input','Low Input','Output','Assets','Consumption']

### Plot means across time
fig, ax = plt.subplots()
for i,mean in enumerate(mean_list):
    ax.plot(range(0,T), mean, label=names_list[i])
    ax.legend()
    ax.set_xlabel('Time')
ax.set_title('Average across time')
if save==True:
    fig.savefig(folder+'averages_across_time.png')                    
plt.show()     



def plot_distribution(state, state_name, save=False, folder=folder):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.distplot(state, label=state_name)
    plt.title('Distribution of '+state_name)
    plt.xlabel(state_name)
    plt.ylabel("Density")
    plt.legend()
    if save==True:
        fig.savefig(folder+'distribution'+state_name+'.png')
    return plt.show()

plot_distribution(m1_state, 'input high', save=save)   
plot_distribution(m2_state, 'input Low', save=save)  
plot_distribution(y_state, 'Output', save=save)  
plot_distribution(a_state, 'Insurance', save=save)  
plot_distribution(c_state, 'Consumption', save=save)  


#%% Compute Sample statistics

def compute_stats_list(state_list, state_names):
    mean = np.empty(len(state_list))
    variance = np.empty(len(state_list))
    sd = np.empty(len(state_list))
    gini_stat = np.empty(len(state_list))
    
    for i,state in enumerate(state_list):
        mean[i] = np.mean(state)
        variance[i] = np.var(state)
        sd[i] = np.sqrt(np.var(state))
        gini_stat[i] = gini(state)
    
    data = {'state':state_names, 'mean': mean, 'Variance': variance, 'Sd': sd, 'Gini': gini_stat }
    df = pd.DataFrame.from_dict(data)
    return df

state_list = [m1_state, m2_state, y_state, a_state, c_state]

stats = compute_stats_list(state_list, names_list)

print(stats.to_latex())
