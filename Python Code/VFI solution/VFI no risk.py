# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:55:59 2019

@author: rodri
"""


# =============================================================================
#  Solves the HH problem under VFI and Simulates the Economy in the Primary scenario Without risk.
# =============================================================================
# Case of no access to assets or insurance market.
# For details of the model see my github:


import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from quantecon import compute_fixed_point
import quantecon as qe
from scipy.interpolate import LinearNDInterpolator as linear_interp
import os
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/python')
from model_class import HH_Model
from data_functions_albert import gini
import seaborn as sns
import pandas as pd
from scipy.optimize import minimize

folder='C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/model no risk/'
save= True


# Import the model class. Gives direct access to parameters, grids, functions and 
# other utilities to specify the model
cp = HH_Model()

# Import set-up parameters, grids and functions
A, B = cp.A, cp.B
α, γ, ρ, β  = cp.α, cp.γ, cp.ρ, cp.β 
N_m, N_θ, N_ε = cp.N_m, cp.N_θ, cp.N_ε
p = cp.p
m_grid, ε_grid, θ_grid = cp.m_grid, cp.ε_grid, cp.θ_grid
pi_ε, pi_θ = cp.pi_ε, cp.pi_θ 
u, u_float, y1, y2 = cp.u, cp.u_float, cp.y1, cp.y2

# Compute expected value of  shocks:
mc_θ = qe.MarkovChain(pi_θ)
mc_ε = qe.MarkovChain(pi_ε)

# == Compute the stationary distribution of Pi == #
θ_0 = mc_θ.stationary_distributions[0]
ε_0 = mc_ε.stationary_distributions[0]

# compute expected value of shocks
exp_eps = ε_0[0]*ε_grid[0] + ε_0[1]*ε_grid[1]
exp_theta = θ_0[0]*θ_grid[0] + θ_0[1]*θ_grid[1] 

# Production function with no risk:
y1 = lambda m1: exp_eps*exp_theta*A*m1**α
y2 = lambda m2: exp_eps*B*m2**γ

# === Set empty Value and Policy functions: === #
V_new = np.empty(( N_m, N_m))
policy_a = np.empty(( N_m, N_m))
policy_m1 = np.empty(( N_m, N_m))
policy_m2 = np.empty(( N_m, N_m))
policy_c = np.empty(( N_m, N_m))



## Generate guess on V:
V_0_func = lambda m1, m2: u_float(exp_theta*exp_eps*A*m1**α +exp_eps*B*m2**γ -p*(m1 +m2))/(1-β)



#improve v_guess
V_guess = np.zeros(( N_m, N_m))


for j,m1 in enumerate(m_grid):
    for k,m2 in enumerate(m_grid):
        V_guess[j,k] = np.nan_to_num(V_0_func(m1, m2))


 

#Check that production funcions accomplish the conditions of expected return and expected marginal returns.
#Set-up functions


y = lambda y1, y2: y1 +y2
diff_y1 = lambda m1: α*exp_eps*exp_theta*A*m1**(α-1)
diff_y2 = lambda m2: γ*exp_eps*B*m2**(γ-1)


fig,ax = plt.subplots()
ax.plot(m_grid, y1(m_grid), label='y1')
ax.plot(m_grid, y2(m_grid), label='y2')
ax.legend()
ax.set_xlabel('Input Quantity')
ax.set_title('value of the production function without risk.')
plt.show()  

fig,ax = plt.subplots()
ax.plot(m_grid, diff_y1(m_grid), label='y1_diff')
ax.plot(m_grid, diff_y2(m_grid) , label='y2_diff')
ax.plot(m_grid, p*np.ones(len(m_grid)), label='p')
ax.legend()
ax.set_xlabel('Input Quantity')
ax.set_title('marginal products vs Price.')
plt.show()  

def optimal_m1(m1):
    return (diff_y1(m1) -p)**2

def optimal_m2(m2):
    return (diff_y2(m2) -p)**2


x0 = [12]
res = minimize(optimal_m1, x0, method='Nelder-Mead', tol=1e-6)
m1_star = res.x

res = minimize(optimal_m2, x0, method='Nelder-Mead', tol=1e-6)
m2_star = res.x

profit_star = y1(m1_star)+y2(m2_star)-p*(m1_star+m2_star)

#%%
m=np.empty(( N_m, N_m, N_m, N_m))


def bellman_operator_discrete(V, return_policies=False):             
    #Given state variables, find optimal choices (g_a, g_m1, g_m2):    
    for i_m1, m1 in enumerate(m_grid):
        for i_m2, m2 in enumerate(m_grid):                                                 
            for i_gm2, g_m2 in enumerate(m_grid):                           
                m[i_m1,i_m2, :,i_gm2] = u(exp_theta*exp_eps*A*m1**α +exp_eps*B*m2**γ -p*m_grid -p*g_m2)+β*(V[:, i_gm2])  
                            #print(m[i_m1,i_m2,, :,i_gm1,i_gm2])
                            
            V_new[i_m1,i_m2] = np.nanmax(m[i_m1,i_m2,:,:])
                        #print(V_new)
                        
    return V_new 
    
def bellman_operator_policies(V):             
    #Given state variables, find optimal choices (g_a, g_m1, g_m2): 
    for i_m1, m1 in enumerate(m_grid):
        for i_m2, m2 in enumerate(m_grid):                            
            for i_gm2, g_m2 in enumerate(m_grid):                           
                m[i_m1,i_m2,:,i_gm2] = u(exp_theta*exp_eps*A*m1**α +exp_eps*B*m2**γ -p*m_grid -p*g_m2) +β*(V[:, i_gm2])   
                                #print(m[i_m1,i_m2,, :,i_gm1,i_gm2])
                            
            V_new[i_m1,i_m2] = np.nanmax(m[i_m1,i_m2, :,:])                        
                    
            policy_m1[i_m1,i_m2] = m_grid[np.unravel_index(np.argmax(m[i_m1,i_m2, :,:], axis=None), m[i_m1,i_m2, :,:].shape)[0]]
            policy_m2[i_m1,i_m2] = m_grid[np.unravel_index(np.argmax(m[i_m1,i_m2, :,:], axis=None), m[i_m1,i_m2, :,:].shape)[1]]
            policy_c[i_m1,i_m2] = exp_theta*exp_eps*A*m1**α +exp_eps*B*m2**γ -p*policy_m1[i_m1,i_m2] -p*policy_m2[i_m1,i_m2] 
    
    return V_new, policy_m1, policy_m2, policy_c

    
qe.tic()
V = compute_fixed_point(bellman_operator_discrete, V_guess, max_iter=1, error_tol=0.1)


V2 = compute_fixed_point(bellman_operator_discrete, V, max_iter=100, error_tol=0.01)
qe.toc()
V_next, g_m1, g_m2, g_c = bellman_operator_policies(V2)



#%% Asset policy functions across shocks
        
def plot_policy(policy, policy_name, m0=0, save=False, folder=folder):      
    fig,ax = plt.subplots()
    ax.plot(m_grid, np.array(policy[m0,:]), label=policy_name)
    ax.legend()
    ax.set_xlabel('M2 input state')
    ax.set_title(policy_name+' under m2='+str(round(m_grid[m0],ndigits=2)))
    if save == True:
        fig.savefig(folder+policy_name+'m='+str(round(m_grid[m0],ndigits=2))+'.png')                
    return plt.show()


# Value function
for m0 in range(N_m-1):
    plot_policy(V_next, policy_name='V', m0=m0, save=save)

# Consumption policy
for m0 in range(N_m-1):
    plot_policy(g_c, policy_name='g_c', m0=m0, save=save)


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

def plot_policy_3d( policy, policy_name, save=False, folder=folder, grid1=m_grid, grid2=m_grid):

    x, y = np.meshgrid(m_grid, m_grid)  
    fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
    surf = fig.add_subplot(111, projection='3d')
    surf.plot_surface(x,
                y,
                np.array(policy[:,:]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black')
          
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d   
    surf.set_title(policy_name)
    if save==True:
        fig.savefig(folder+policy_name+'inputs_2dgrid.png')                 
    return plt.show()


## Plot Value function
plot_policy_3d(policy=V_new, policy_name='Value function', save=save)

## Plot Value function
plot_policy_3d(policy=g_c, policy_name='Consumption Policy', save=save)

## Plot Value function
plot_policy_3d(policy=g_m1, policy_name='Input high Policy', save=save)

## Plot Value function
plot_policy_3d(policy=g_m2, policy_name='Input low Policy', save=save)




#%%
# =============================================================================
# Compute stationary distribution: Simulate the Economy
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



def invariant_distr( T=1000, N=1000, θ_h_0= 0.5, ε_h_0 = 0.5):
    m1_state = np.empty(T*N).reshape(T,N)
    m2_state = np.empty(T*N).reshape(T,N)
    y_state = np.empty(T*N).reshape(T,N)
    c_state = np.empty((T,N))
    mean_m1 = np.empty(T)
    mean_m2 = np.empty(T)
    mean_y = np.empty(T)
    mean_c = np.empty(T)
    m1_state[0,:] = m_grid[1]
    m2_state[0,:] = m_grid[1]
    y_state[0,:] = y1(m1_state[0,:]) +y2(m2_state[0,:])
    input1_list = m_grid.tolist()
    input2_list = m_grid.tolist()


    for t in range(1, T):        
        for n in range(0,N):
            m1_state[t,n] = g_m1[input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n])]
            m2_state[t,n] = g_m2[input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n])]
            c_state[t,n] =  g_c[input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n])]
            y_state [t,n] = y1(m1_state[t-1,n]) +y2(m2_state[t-1,n])
        mean_m1[t] = np.mean(m1_state[t,:]) 
        mean_m2[t] = np.mean(m2_state[t,:])
        mean_y[t] = np.mean(y_state[t,:])
        mean_c[t] = np.mean(c_state[t,:])
            
    return  m1_state[T-1,:], m2_state[T-1,:] , y_state[T-1,:], c_state[T-1,:], mean_m1, mean_m2, mean_y, mean_c

m1_state, m2_state, y_state,c_state, mean_m1, mean_m2, mean_y, mean_c = invariant_distr(T=T, N=N, θ_h_0=θ_h_0 , ε_h_0 = ε_h_0)



#%% Input states

mean_list = [mean_m1, mean_m2, mean_y, mean_c]
names_list = ['High input','Low Input','Output','Consumption']

### Plot means across time
fig, ax = plt.subplots()
for i,mean in enumerate(mean_list):
    ax.plot(range(5,T), mean[5:], label=names_list[i])
    ax.legend()
    ax.set_xlabel('Time')
ax.set_title('Average across time')
if save==True:
    fig.savefig(folder+'averages_across_time.png')                    
plt.show()     

## Transicion
fig, ax = plt.subplots()
for i,mean in enumerate(mean_list):
    ax.plot(range(0,5), mean[:5], label=names_list[i])
    ax.legend()
    ax.set_xlabel('Time')
ax.set_title('Average across time')
if save==True:
    fig.savefig(folder+'averages_across_time_beginning.png')                    
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

y_profit = y_state -p*(m1_state+m2_state)

state_list = [m1_state, m2_state, y_state, c_state]

stats = compute_stats_list(state_list, names_list)

print(stats.to_latex())


