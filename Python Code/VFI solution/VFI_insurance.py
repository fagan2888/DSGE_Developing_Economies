# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:03:31 2019

@author: rodri
"""

# =============================================================================
#  Solves the HH problem under VFI and Simulates the Economy in the scenario of
# acess to insurance market on θ shock.
# =============================================================================
# Case of no access to assets or insurance market.
# For details of the model see my github:

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from quantecon import compute_fixed_point
import quantecon as qe
import os
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/python')
from data_functions_albert import gini
import seaborn as sns
import pandas as pd
from model_class import HH_Model

cp = HH_Model()
A, B = cp.A, cp.B
α, γ, ρ, β  = cp.α, cp.γ, cp.ρ, cp.β 
N_m, N_x, N_θ, N_ε = cp.N_m, cp.N_x, cp.N_θ, cp.N_ε
p = cp.p
q = cp.q

folder = 'C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/model inputs and insurance/'

m_grid, x_grid, ε_grid,  θ_grid = cp.m_grid, cp.x_grid, cp.ε_grid, cp.θ_grid

pi_ε, pi_θ = cp.pi_ε, cp.pi_θ 
u, u_float, y1, y2 = cp.u, cp.u_float, cp.y1, cp.y2
        

# === Set empty Value and Policy functions: === #
V_new = np.empty(( N_m, N_m, N_x, N_θ, N_ε))
policy_x = np.empty(( N_m, N_m, N_x, N_θ, N_ε))
policy_m1 = np.empty(( N_m, N_m, N_x, N_θ, N_ε))
policy_m2 = np.empty(( N_m, N_m, N_x, N_θ, N_ε))
policy_c = np.empty(( N_m, N_m, N_x, N_θ, N_ε))


V_0_func = lambda  m1, m2,x, θ, ε: u_float(θ*ε*A*m1**α +ε*B*m2**γ+ -p*(m1 +m2)-q*x +x*(θ == θ_grid[0]))/(1-β)


def V_0_func_2(m1, m2, θ, ε, l, m):
    
    Expt1 = pi_θ[l,0]*pi_ε[m,0]*θ_grid[0]*ε_grid[0] +pi_θ[l,1]*pi_ε[m,0]*θ_grid[1]*ε_grid[0]+pi_θ[l,0]*pi_ε[m,1]*θ_grid[0]*ε_grid[1]+pi_θ[l,1]*pi_ε[m,1]*θ_grid[1]*ε_grid[1]
    Expt2 = pi_ε[m,0]*ε_grid[0] +pi_ε[m,1]*ε_grid[1] 
    gm1 = (p/(Expt1*α*A))**(1/(α-1))
    gm2 = (p/(Expt2*γ*B))**(1/(γ-1))    
    return u(θ*ε*A*20*m1**α +ε*B*20*m2**γ+ -p*(gm1 +gm2))/(1-β) +200

#improve v_guess
V_guess = np.zeros((N_m, N_m, N_x, N_θ, N_ε))
'''
for i,a in enumerate(a_grid):
    for j,m1 in enumerate(m_grid):
        for k,m2 in enumerate(m_grid):
            for l,θ in enumerate(θ_grid):
                for m,ε in enumerate(ε_grid):
                    V_guess[i,j,k,l,m] = np.nan_to_num(V_0_func(a, m1, m2, θ, ε))
'''

 

#Check that production funcions accomplish the conditions of expected return and expected marginal returns.
#Set-up functions

diff_y1 = lambda m1,ε,θ: α*ε*θ*A*m1**(α-1)
diff_y2 = lambda m2,ε: γ*ε*B*m2**(γ-1)

Ey1 = lambda m1: pi_θ[1,1]*pi_ε[1,1]*A*m1**α
Ey2 = lambda m2: pi_ε[1,1]*B*m2**γ

E_diff_y1 =  lambda m1: pi_θ[1,1]*pi_ε[1,1]*α*A*m1**(α-1)
E_diff_y2 =  lambda m2: pi_ε[1,1]*γ*B*m2**(γ-1)


fig,ax = plt.subplots()
ax.plot(m_grid, Ey1(m_grid), label='E(y1)')
ax.plot(m_grid, Ey2(m_grid), label='E(y2)')
ax.legend()
ax.set_xlabel('Input Quantity')
ax.set_title('Expected value of the production function.')
plt.show()  

fig,ax = plt.subplots()
ax.plot(m_grid, E_diff_y1(m_grid), label='E(y1_diff)')
ax.plot(m_grid, E_diff_y2(m_grid) , label='E(y2_diff)')
ax.plot(m_grid, p*np.ones(len(m_grid)), label='p')
ax.legend()
ax.set_xlabel('Input Quantity')
ax.set_title('Expected value of the marginal products.')
plt.show()  


fig,ax = plt.subplots()
ax.plot(m_grid, diff_y1(m_grid,1,1), label='y1_diff (no uncertainty)')
ax.plot(m_grid, diff_y2(m_grid,1) , label='y2_diff (no uncertainty)')
ax.plot(m_grid, p*np.ones(len(m_grid)), label='p')
ax.legend()
ax.set_xlabel('Input Quantity')
ax.set_title('Expected value of the marginal products.')
plt.show()  



#

m=np.empty(( N_m, N_m, N_x, N_θ, N_ε, N_m, N_m, N_x))


def bellman_operator_discrete(V, return_policies=False): 
            
    #Given state variables, find optimal choices (g_a, g_m1, g_m2):
    
    for i_m1, m1 in enumerate(m_grid):
        for i_m2, m2 in enumerate(m_grid):
            for i_x, x in enumerate(x_grid): 
                for i_θ, θ in enumerate(θ_grid):
                    for i_ε, ε in enumerate(ε_grid):                                                  
                        for i_gm2, g_m2 in enumerate(m_grid): 
                            for i_gx, g_x in enumerate(x_grid):
                                
                                m[i_m1,i_m2,i_x,i_θ, i_ε, :,i_gm2, i_gx] = u(θ*ε*A*m1**α +ε*B*m2**γ -p*m_grid -p*g_m2 +x*(θ == θ_grid[0]))+β*(pi_θ[i_θ,0]*pi_ε[i_ε,0]*V[:, i_gm2, i_gx, 0,0] +pi_θ[i_θ,1]*pi_ε[i_ε,0]*V[:,i_gm2, i_gx, 1,0] +pi_θ[i_θ,0]*pi_ε[i_ε,1]*V[:, i_gm2, i_gx, 0,1] +pi_θ[i_θ,1]*pi_ε[i_ε,1]*V[:, i_gm2, i_gx, 1,1])   
                            #print(m[i_m1,i_m2,i_θ, i_ε, :,i_gm1,i_gm2])
                            
                        V_new[i_m1,i_m2,i_x,i_θ,i_ε] = np.nanmax(m[i_m1,i_m2,i_x, i_θ,i_ε,:,:,:])
                        #print(V_new)
                        
    return V_new 
    
def bellman_operator_policies(V): 
            
    #Given state variables, find optimal choices (g_a, g_m1, g_m2):
 
    for i_m1, m1 in enumerate(m_grid):
        for i_m2, m2 in enumerate(m_grid):
            for i_x, x in enumerate(x_grid): 
                for i_θ, θ in enumerate(θ_grid):
                    for i_ε,ε in enumerate(ε_grid):                                                  
                        for i_gm2, g_m2 in enumerate(m_grid): 
                            for i_gx, g_x in enumerate(x_grid):
                                
                                m[i_m1,i_m2,i_x,i_θ, i_ε, :,i_gm2, i_gx] = u(θ*ε*A*m1**α +ε*B*m2**γ -p*m_grid -p*g_m2 +x*(θ == θ_grid[0])-q*g_x)+β*(pi_θ[i_θ,0]*pi_ε[i_ε,0]*V[:, i_gm2, i_gx, 0,0] +pi_θ[i_θ,1]*pi_ε[i_ε,0]*V[:, i_gm2,i_gx,1,0] +pi_θ[i_θ,0]*pi_ε[i_ε,1]*V[:, i_gm2,i_gx,0,1] +pi_θ[i_θ,1]*pi_ε[i_ε,1]*V[:, i_gm2, i_gx,1,1])   
                            #print(m[i_m1,i_m2,i_θ, i_ε, :,i_gm1,i_gm2])
                            
                        V_new[i_m1,i_m2,i_x,i_θ,i_ε] = np.nanmax(m[i_m1,i_m2,i_x, i_θ,i_ε,:,:,:])                      
                    
                        policy_m1[i_m1,i_m2,i_x,i_θ,i_ε] = m_grid[np.unravel_index(np.argmax(m[i_m1,i_m2,i_x, i_θ,i_ε,:,:,:], axis=None), m[i_m1,i_m2,i_x,i_θ,i_ε,:,:,:].shape)[0]]
                        policy_m2[i_m1,i_m2,i_x,i_θ,i_ε] = m_grid[np.unravel_index(np.argmax(m[i_m1,i_m2,i_x, i_θ,i_ε,:,:,:], axis=None), m[i_m1,i_m2,i_x,i_θ,i_ε,:,:,:].shape)[1]]
                        policy_x[i_m1,i_m2,i_x,i_θ,i_ε] = x_grid[np.unravel_index(np.argmax(m[i_m1,i_m2,i_x, i_θ,i_ε,:,:,:], axis=None), m[i_m1,i_m2,i_x,i_θ,i_ε,:,:,:].shape)[2]]
                        policy_c[i_m1,i_m2,i_x,i_θ,i_ε] = θ*ε*A*m1**α +ε*B*m2**γ -p*policy_m1[i_m1,i_m2,i_x,i_θ,i_ε] -p*policy_m2[i_m1,i_m2,i_x,i_θ,i_ε] 
    
    return V_new, policy_m1, policy_m2, policy_x, policy_c

    


qe.tic()
V = compute_fixed_point(bellman_operator_discrete, V_guess, max_iter=1, error_tol=0.1)


V2 = compute_fixed_point(bellman_operator_discrete, V, max_iter=100, error_tol=0.01)
qe.toc()
V_next, g_m1, g_m2, g_x, g_c = bellman_operator_policies(V2)



#%% Asset policy functions across shocks
        
def plot_policy(policy, policy_name, m0=0, x0=0, save=False, folder=folder):      
    fig,ax = plt.subplots()
    ax.plot(m_grid, np.array(policy[:,m0,x0,1,1]), label=policy_name+r'($\theta=1, \varepsilon=1)$')
    ax.plot(m_grid, np.array(policy[:,m0,x0,0,1]), label=policy_name+r'($\theta=0, \varepsilon=1)$')
    ax.plot(m_grid, np.array(policy[:,m0,x0,1,0]), label=policy_name+r'($\theta=1, \varepsilon=0)$')
    ax.plot(m_grid, np.array(policy[:,m0,x0,0,0]), label=policy_name+r'($\theta=0, \varepsilon=0)$')
    ax.legend()
    ax.set_xlabel('M1 input')
    ax.set_title(policy_name+' under m2='+str(round(m_grid[m0],ndigits=2))+', x='+str(round(m_grid[x0],ndigits=2)))
    if save == True:
        fig.savefig(folder+policy_name+'m='+str(round(m_grid[m0],ndigits=2))+'_'+str(round(x_grid[x0],ndigits=2))+'.png')             
    plt.show()

save = True
# Value function
for m0 in range(N_m-1):
    plot_policy(V_next, policy_name='V', save=save, m0=m0)

# Consumption policy
for m0 in range(N_m-1):
    plot_policy(g_c, policy_name='g_c', save=save, m0=m0)


# Input high policy
for m0 in range(N_m-1):
    plot_policy(g_m1, policy_name='g_m1', save=save, m0=m0)

# Input low policy
for i in range(N_m-1):
    plot_policy(g_m2, policy_name='g_m2', save=save, m0=m0)
    
# Insurance
for i in range(N_m-1):
    plot_policy(g_x, policy_name='g_x', save=save, m0=m0)

   
## For high level of insurance: ===============================
x0 = 4
for m0 in range(N_m-1):
    plot_policy(V_next, policy_name='Value function', m0=m0, save=save, x0=x0)

# Consumption policy
for m0 in range(N_m-1):
    plot_policy(g_c, policy_name='g_c', m0=m0, save=save, x0=x0)


# Input high policy
for m0 in range(N_m-1):
    plot_policy(g_m1, policy_name='g_m1', m0=m0, save=save, x0=x0)

# Input low policy
for i in range(N_m-1):
    plot_policy(g_m2, policy_name='g_m2', m0=m0, save=save, x0=x0)
    
# Insurance
for i in range(N_m-1):
    plot_policy(g_x, policy_name='g_x', m0=m0, save=save, x0=x0)   


        
#%% Asset policy functions across shocks

#3d graphs

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
x0=0
def plot_policy_3d(policy, policy_name, x0=0, save=False, folder=folder, grid1=m_grid, grid2=m_grid):

    x, y = np.meshgrid(m_grid, m_grid)  
    fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
    surf = fig.add_subplot(111, projection='3d')
    surf.plot_surface(x,
                y,
                np.array(policy[:,:,x0,1,1]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='good-good')
    surf.plot_surface(x,
                y,
                np.array(policy[:,:,x0,1,0]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='good-bad')
    surf.plot_surface(x,
                y,
                np.array(policy[:,:,x0,0,1]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='bad-good')
    surf.plot_surface(x,
                y,
                np.array(policy[:,:,x0,0,0]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='bad-bad')          
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d   
    surf.set_title(policy_name+', x='+str(round(m_grid[x0],ndigits=2)))
    if save==True:
        fig.savefig(folder+policy_name+'inputs_2dgrid'+'_'+str(round(x_grid[x0],ndigits=2))+'.png')                 
    return plt.show()


## Plot Value function
for x0 in range(N_x):
    plot_policy_3d(policy=V_new, x0=x0, save=save, policy_name='Value function')

## Plot Value function
    plot_policy_3d(policy=g_c, x0=x0, save=save, policy_name='Consumption Policy')
    
## Plot Value function
    plot_policy_3d(policy=g_x, x0=x0, save=save, policy_name='Insurance Policy')
    
## Plot Value function
    plot_policy_3d(policy=g_m1, save=save, x0=x0, policy_name='Input high Policy')

## Plot Value function
    plot_policy_3d(policy=g_m2, save=save, x0=x0, policy_name='Input low Policy')




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
    y_state = np.empty(T*N).reshape(T,N)
    c_state = np.empty((T,N))
    x_state = np.empty((T,N))
    mean_m1 = np.empty(T)
    mean_m2 = np.empty(T)
    mean_y = np.empty(T)
    mean_x = np.empty(T)
    mean_c =np.empty(T)
    draw = np.random.uniform(size=N)
    θ_shock[0, :] = draw<θ_h_0
    ε_shock[0,:] = draw<ε_h_0
    m1_state[0,:] = m_grid[int(N_m/2)]
    m2_state[0,:] = m_grid[int(N_m/2)]
    y_state[0,:] = y1(m1_state[0,:],θ_shock[0, :],ε_shock[0,:]) +y2(m2_state[0,:],ε_shock[0,:])
    input1_list = m_grid.tolist()
    input2_list = m_grid.tolist()
    x_list = x_grid.tolist()
    

    for t in range(1, T):        
        for n in range(0,N):
            draw = np.random.uniform()
            draw2 = np.random.uniform()
            m1_state[t,n] = g_m1[input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),x_list.index(x_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]
            m2_state[t,n] = g_m2[input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),x_list.index(x_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]
            x_state[t,n] = g_x[input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),x_list.index(x_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]
            c_state[t,n] = g_c[input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),x_list.index(x_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]           
            θ_shock[t, n] = int(draw < pi_θ[int(θ_shock[t-1, n]),1])
            ε_shock[t,n] =  int(draw2 < pi_ε[int(ε_shock[t-1, n]),1])
            y_state [t,n] = y1(m1_state[t-1,n],θ_grid[int(θ_shock[t-1, n])], ε_grid[int(ε_shock[t-1, n])]) +y2(m2_state[t-1,n], ε_grid[int(ε_shock[t-1, n])])
        mean_m1[t] = np.mean(m1_state[t,:]) 
        mean_m2[t] = np.mean(m2_state[t,:])
        mean_y[t] = np.mean(y_state[t,:])
        mean_x[t] = np.mean(x_state[t,:])
        mean_c[t] = np.mean(c_state[t,:])
    return θ_shock[T-1,:], ε_shock[T-1,:], m1_state[T-1,:], m2_state[T-1,:] , x_state[T-1,:], y_state[T-1,:], c_state[T-1,:], mean_m1, mean_m2, mean_y, mean_x, mean_c

θ_shock, ε_shock, m1_state, m2_state, x_state, y_state, c_state, mean_m1, mean_m2, mean_y, mean_x, mean_c = invariant_distr(pi_θ, pi_ε, T=T, N=N, θ_h_0=θ_h_0 , ε_h_0 = ε_h_0)






#%% Averages across time and distribution plots

mean_list = [mean_m1, mean_m2, mean_y, mean_x, mean_c]
names_list = ['High input','Low Input','Output','Insurance','Consumption']

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
plot_distribution(x_state, 'Insurance', save=save)  
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

state_list = [m1_state, m2_state, y_state, x_state, c_state]

stats = compute_stats_list(state_list, names_list)

print(stats.to_latex())


