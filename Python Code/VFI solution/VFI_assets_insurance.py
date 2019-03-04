# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:18:45 2018

@author: rodri
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from quantecon import compute_fixed_point
import quantecon as qe
import os
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/python')
from data_functions_albert import gini
import seaborn as sns
from model_class import HH_Model

## Import model class
cp = HH_Model()
## Import parameters
A, B = cp.A, cp.B
α, γ, ρ, β  = cp.α, cp.γ, cp.ρ, cp.β 
N_a, N_m,N_x, N_θ, N_ε = cp.N_a, cp.N_m, cp.N_x, cp.N_θ, cp.N_ε
r, p, q = cp.r, cp.p, cp.q

# check
print(r<(1/β-1))


a_grid, m_grid, ε_grid, θ_grid, x_grid = cp.a_grid, cp.m_grid, cp.ε_grid, cp.θ_grid, cp.x_grid

pi_ε, pi_θ = cp.pi_ε, cp.pi_θ 
u, u_float, y1, y2 = cp.u, cp.u_float, cp.y1, cp.y2
        

# === Set empty Value and Policy functions: === #
V_new = np.empty((N_a, N_m, N_m,N_x, N_θ, N_ε))
policy_a = np.empty((N_a, N_m, N_m, N_x, N_θ, N_ε))
policy_m1 = np.empty((N_a, N_m, N_m,N_x, N_θ, N_ε))
policy_m2 = np.empty((N_a, N_m, N_m,N_x, N_θ, N_ε))
policy_x = np.empty((N_a, N_m, N_m,N_x, N_θ, N_ε))
policy_c = np.empty((N_a, N_m, N_m,N_x, N_θ, N_ε))





V_0_func = lambda a, m1, m2,x, θ, ε: u_float(θ*ε*A*m1**α +ε*B*m2**γ+(1+r)*a -p*(m1 +m2)-a -q*x +x*(θ == θ_grid[0]))/(1-β)
V_guess = np.zeros((N_a, N_m, N_m,N_x, N_θ, N_ε))
for i,a in enumerate(a_grid):
    for j,m1 in enumerate(m_grid):
        for k,m2 in enumerate(m_grid):
            for l,θ in enumerate(θ_grid):
                for m,ε in enumerate(ε_grid):
                    for n,x in enumerate(x_grid):
                        V_guess[i,j,k,n,l,m] = np.nan_to_num(V_0_func(a, m1, m2,x, θ, ε))
                    
V_guess = np.zeros((N_a, N_m, N_m,N_x, N_θ, N_ε))
#%

m=np.empty((N_a, N_m, N_m, N_x, N_θ, N_ε, N_a, N_m, N_m, N_x))

jit(nopython=True)
def bellman_operator_discrete(V, return_policies=False): 
    def u(c):
        if ρ==1:
            return np.log(c)
        else:
            return (c**(1-ρ) -1) / (1-ρ)            
    #Given state variables, find optimal choices (g_a, g_m1, g_m2):
    for i_a, a in enumerate(a_grid):
        for i_m1, m1 in enumerate(m_grid):
            for i_m2, m2 in enumerate(m_grid):
                for i_x, x in enumerate(x_grid):
                    for i1,i_θ in enumerate(θ_grid):
                        for i2,i_ε in enumerate(ε_grid):
                            for i_gm1, g_m1 in enumerate(m_grid):
                                for i_gm2, g_m2 in enumerate(m_grid):
                                    for i_gx, g_x in enumerate(x_grid):                             
                                        m[i_a,i_m1,i_m2,i_x,i1, i2, :,i_gm1,i_gm2,i_gx] = u(np.fmax(i_θ*i_ε*A*m1**α +i_ε*B*m2**γ +(1+r)*a -p*g_m1 -p*g_m2 -a_grid -q*g_x +x*(i_θ  == θ_grid[0]),1e-3*np.ones(N_a)))+β*(pi_θ[i1,0]*pi_ε[i2,0]*V[:, i_gm1, i_gm2,i_gx,0,0] +pi_θ[i1,1]*pi_ε[i2,0]*V[:, i_gm1, i_gm2,i_gx,1,0] +pi_θ[i1,0]*pi_ε[i2,1]*V[:, i_gm1, i_gm2,i_gx,0,1] +pi_θ[i1,1]*pi_ε[i2,1]*V[:, i_gm1, i_gm2,i_gx,1,1])   
                                        #print(m[i_a,i_m1,i_m2,i_x,i1, i2, :,i_gm1,i_gm2,i_gx])
                                        
                            V_new[i_a,i_m1,i_m2,i_x,i1,i2] = np.nanmax(m[i_a,i_m1,i_m2,i_x,i1,i2,:,:,:,:])
                        #print(V_new)
                            policy_a[i_a,i_m1,i_m2,i_x,i1,i2] = a_grid[np.unravel_index(np.argmax(m[i_a,i_m1,i_m2,i_x, i1,i2,:,:,:,:], axis=None), m[i_a,i_m1,i_m2,i_x, i1,i2,:,:,:,:].shape)[0]]
                            policy_m1[i_a,i_m1,i_m2,i_x,i1,i2] = m_grid[np.unravel_index(np.argmax(m[i_a,i_m1,i_m2,i_x, i1,i2,:,:,:,:], axis=None), m[i_a,i_m1,i_m2,i_x, i1,i2,:,:,:,:].shape)[1]]
                            policy_m2[i_a,i_m1,i_m2,i_x,i1,i2] = m_grid[np.unravel_index(np.argmax(m[i_a,i_m1,i_m2,i_x, i1,i2,:,:,:,:], axis=None), m[i_a,i_m1,i_m2,i_x, i1,i2,:,:,:,:].shape)[2]]
                            policy_x[i_a,i_m1,i_m2,i_x,i1,i2] = x_grid[np.unravel_index(np.argmax(m[i_a,i_m1,i_m2,i_x, i1,i2,:,:,:,:], axis=None), m[i_a,i_m1,i_m2,i_x, i1,i2,:,:,:,:].shape)[3]]
                            policy_c[i_a,i_m1,i_m2,i_x,i1,i2] = i_θ*i_ε*A*m1**α +i_ε*B*m2**γ +(1+r)*a -p*policy_m1[i_a,i_m1,i_m2,i_x,i1,i2] -p*policy_m2[i_a,i_m1,i_m2,i_x,i1,i2] - policy_a[i_a,i_m1,i_m2,i_x,i1,i2] -q*policy_x[i_a,i_m1,i_m2,i_x,i1,i2] +x*(i_θ  == θ_grid[0])
    if return_policies == False:
        return V_new
    else:
        return V_new, policy_a, policy_m1, policy_m2, policy_x, policy_c

qe.tic()
V = compute_fixed_point(bellman_operator_discrete, V_guess, max_iter=1, error_tol=0.1)
V2 = compute_fixed_point(bellman_operator_discrete, V, max_iter=100, error_tol=0.01)
qe.toc()
V_next, g_a, g_m1, g_m2, g_x, g_c = bellman_operator_discrete(V, return_policies=True)





#%% Asset policy functions across shocks

def plot_policy(self,policy, policy_name, m0=0, x0=0, save=False, folder='C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/Model assets'):      
        fig,ax = plt.subplots()
        ax.plot(self.a_grid, np.array(policy[:,m0,m0,x0,1,1]), label=policy_name+r' under $\theta=1, \varepsilon=1)$')
        ax.plot(self.a_grid, np.array(policy[:,m0,m0,x0,0,1]), label=policy_name+r' under $\theta=0, \varepsilon=1)$')
        ax.plot(self.a_grid, np.array(policy[:,m0,m0,x0,1,0]), label=policy_name+r' under $\theta=1, \varepsilon=0)$')
        ax.plot(self.a_grid, np.array(policy[:,m0,m0,x0,0,0]), label=policy_name+r' under $\theta=0, \varepsilon=0)$')
        ax.legend()
        ax.set_xlabel('assets')
        ax.set_title(policy_name)
        plt.show() 


# Value function
for m0 in range(N_m-1):
    plot_policy(V_next, policy_name='Value function', m0=m0)

# Consumption policy
for m0 in range(N_m-1):
    plot_policy(g_c, policy_name='g_c', m0=m0)

# Assets policy
for m0 in range(N_m-1):
    plot_policy(g_a, policy_name='g_a', m0=m0)
    
# Insurance policy
for i in range(N_m-1):
    plot_policy(g_x, policy_name='g_x', m0=m0)

# Input high policy
for m0 in range(N_m-1):
    plot_policy(g_m1, policy_name='g_m1', m0=m0)

# Input low policy
for i in range(N_m-1):
    plot_policy(g_m2, policy_name='g_m2', m0=m0)



#%% Asset policy functions across shocks

#3d graphs

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

x, y = np.meshgrid(m_grid, m_grid)  

# 3D plot good-good scenario  ===================================

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,y, np.array(g_m1[1,:,:,1,1,1]),
                rstride=2, cstride=2, alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x, y, np.array(g_m2[1,:,:,1,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Inputs policy functions under $\theta=1, \varepsilon=1$. blue = $g^{m1}$, Orange =$g^{m2}$ ')
plt.show()

# 3D plot bad-good scenario ================================

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,y,np.array(g_m1[1,:,:,1,0,1]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x,y,np.array(g_m2[1,:,:,1,0,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Inputs policy functions under $\theta=0, \varepsilon=1$. blue = $g^{m1}$, Orange =$g^{m2}$ ')
plt.show()


# 3D plot good-bad scenario ===========================

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,
                y, np.array(g_m1[1,:,:,1,1,0]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x,y,np.array(g_m2[1,:,:,1,1,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Inputs policy functions under $\theta=1, \varepsilon=0$. blue = $g^{m1}$, Orange =$g^{m2}$ ')
plt.show()

# 3D plot bad-bad scenario  ===================================

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x, y,np.array(g_m1[1,:,:,1,0,0]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x,y,np.array(g_m2[1,:,:,1,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Inputs policy functions under $\theta=0, \varepsilon=0$. blue = $g^{m1}$, Orange =$g^{m2}$ ')
plt.show()



#%% Asset policy functions across shocks

#3d graphs

# 3D plot good-good scenario
x, y = np.meshgrid(m_grid, m_grid)  

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x, y,np.array(g_a[1,:,:,1,1,1]),
                rstride=2, cstride=2, alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')

surf.plot_surface(x,y,np.array(g_x[1,:,:,1,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
                
surf.plot_surface(x, y,np.array(g_c[1,:,:,1,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Assets and consumption policy functions under $\theta=1, \varepsilon=1$. blue = $g^{a}$, Orange =$g^{c}$ ')
plt.show()

# 3D plot bad-good scenario
fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')

surf.plot_surface(x,y,np.array(g_a[1,:,:,1,0,1]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')

surf.plot_surface(x, y, np.array(g_x[1,:,:,1,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')   
            
surf.plot_surface(x,y,np.array(g_c[1,:,:,1,0,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Assets and consumption policy functions under $\theta=0, \varepsilon=1$. blue = $g^{a}$, Orange =$g^{c}$ ')
plt.show()


# 3D plot good-bad scenario
fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,y,np.array(g_a[1,:,:,1,1,0]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
surf.plot_surface(x,y,np.array(g_x[1,:,:,1,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')                
surf.plot_surface(x, y,np.array(g_c[1,:,:,1,1,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Assets and consumption policy functions under $\theta=1, \varepsilon=0$. blue = $g^{a}$, Orange =$g^{c}$ ')
plt.show()

# 3D plot bad-bad scenario

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,y,np.array(g_a[1,:,:,1,0,0]),
                alpha=0.7, linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')

surf.plot_surface(x, y,np.array(g_x[1,:,:,1,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

surf.plot_surface(x, y,np.array(g_c[1,:,:,1,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2') 
               
surf.plot_surface(x, y,np.array(g_c[1,:,:,1,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Assets and consumption policy functions under $\theta=0, \varepsilon=0$. blue = $g^{a}$, Orange =$g^{c}$ ')
plt.show()



#%% Value function plots

x, y = np.meshgrid(m_grid, m_grid)  

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x, y, np.array(V_next[1,:,:,1,1,1]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x, y, np.array(V_next[1,:,:,1,1,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

surf.plot_surface(x, y, np.array(V_next[1,:,:,1,0,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

surf.plot_surface(x, y, np.array(V_next[1,:,:,1,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')


#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Value function  different shocks')
plt.show()


#%%
# =============================================================================
# Grid on assets and insurance
# =============================================================================

#3d graphs

x, y = np.meshgrid(a_grid, x_grid)  

# 3D plot good-good scenario  ===================================

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,y, np.array(g_m1[:,1,1,:,1,1]),
                rstride=2, cstride=2, alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x, y, np.array(g_m2[:,1,1,:,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Inputs policy functions under $\theta=1, \varepsilon=1$. blue = $g^{m1}$, Orange =$g^{m2}$ ')
plt.show()

# 3D plot bad-good scenario ================================

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,y,np.array(g_m1[:,1,1,:,0,1]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x,y,np.array(g_m2[:,1,1,:,0,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Inputs policy functions under $\theta=0, \varepsilon=1$. blue = $g^{m1}$, Orange =$g^{m2}$ ')
plt.show()


# 3D plot good-bad scenario ===========================

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,
                y, np.array(g_m1[:,1,1,:,1,0]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x,y,np.array(g_m2[:,1,1,:,1,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Inputs policy functions under $\theta=1, \varepsilon=0$. blue = $g^{m1}$, Orange =$g^{m2}$ ')
plt.show()

# 3D plot bad-bad scenario  ===================================

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x, y,np.array(g_m1[:,1,1,:,0,0]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x,y,np.array(g_m2[:,1,1,:,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Inputs policy functions under $\theta=0, \varepsilon=0$. blue = $g^{m1}$, Orange =$g^{m2}$ ')
plt.show()



#%% Asset, insurance, consumption policy functions across shocks

# 3D plot good-good scenario
x, y = np.meshgrid(a_grid, x_grid)  

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x, y,np.array(g_a[:,1,1,:,1,1]),
                rstride=2, cstride=2, alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')

surf.plot_surface(x,y,np.array(g_x[:,1,1,:,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
                
surf.plot_surface(x, y,np.array(g_c[:,1,1,:,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Assets, insurance and consumption policy functions under $\theta=1, \varepsilon=1$. blue = $g^{a}$, Orange =$g^{c}$ ')
plt.show()

# 3D plot bad-good scenario
fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')

surf.plot_surface(x,y,np.array(g_a[:,1,1,:,0,1]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')

surf.plot_surface(x, y, np.array(g_x[:,1,1,:,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')   
            
surf.plot_surface(x,y,np.array(g_c[:,1,1,:,0,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Assets, insurance and consumption policy functions under $\theta=0, \varepsilon=1$. blue = $g^{a}$, Orange =$g^{c}$ ')
plt.show()


# 3D plot good-bad scenario
fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,y,np.array(g_a[:,1,1,:,1,0]),
                rstride=2, cstride=2,
                alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
surf.plot_surface(x,y,np.array(g_x[:,1,1,:,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')                
surf.plot_surface(x, y,np.array(g_c[:,1,1,:,1,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Assets, insurance and consumption policy functions under $\theta=1, \varepsilon=0$. blue = $g^{a}$, Orange =$g^{c}$ ')
plt.show()

# 3D plot bad-bad scenario

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x,y,np.array(g_a[:,1,1,:,0,0]),
                alpha=0.7, linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')

surf.plot_surface(x, y,np.array(g_x[:,1,1,:,1,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

surf.plot_surface(x, y,np.array(g_c[:,1,1,:,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2') 
               
surf.plot_surface(x, y,np.array(g_c[:,1,1,:,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Assets and consumption policy functions under $\theta=0, \varepsilon=0$. blue = $g^{a}$, Orange =$g^{c}$ ')
plt.show()



#%% Value function plots

x, y = np.meshgrid(a_grid, x_grid)  

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x, y, np.array(V_next[:,1,1,:,1,1]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x, y, np.array(V_next[:,1,1,:,1,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

surf.plot_surface(x, y, np.array(V_next[:,1,1,:,0,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

surf.plot_surface(x, y, np.array(V_next[:,1,1,:,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')


#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Value function  different shocks')
plt.show()


surf.plot_surface(x,y,np.array(g_c[:,1,1,:,0,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')
#ax.set_zlim(0, m_max)
fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
surf = fig.add_subplot(111, projection='3d')
surf.plot_surface(x, y, np.array(g_c[:,1,1,:,1,1]),
                rstride=2, cstride=2,alpha=0.7,
                linewidth=0.01, facecolor='y', edgecolor='black', label='g_m1')
                
surf.plot_surface(x, y, np.array(g_c[:,1,1,:,1,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

surf.plot_surface(x, y, np.array(g_c[:,1,1,:,0,1]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')

surf.plot_surface(x, y, np.array(g_c[:,1,1,:,0,0]),
                alpha=0.5, linewidth=0.25,facecolor='y', edgecolor='black', label='g_m2')


#ax.set_zlim(0, m_max)
#surf._edgecolors2d=surf._edgecolors3d
surf.set_title(r'Value function  different shocks')
plt.show()



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
    x_state = np.empty((T,N))
    mean_x = np.empty(T)
    mean_m1 = np.empty(T)
    mean_m2 = np.empty(T)
    mean_a = np.empty(T)
    mean_y = np.empty(T)
    draw = np.random.uniform(size=N)
    θ_shock[0, :] = draw<θ_h_0
    ε_shock[0,:] = draw<ε_h_0
    m1_state[0,:] = m_grid[int(N_m/2)]
    m2_state[0,:] = m_grid[int(N_m/2)]
    a_state[0,:] = a_grid[int(N_a/2)]
    x_state[0,:] = x_grid[0]
    y_state[0,:] = y1(m1_state[0,:],θ_shock[0, :],ε_shock[0,:]) +y2(m2_state[0,:],ε_shock[0,:])
    input1_list = m_grid.tolist()
    input2_list = m_grid.tolist()

    a_list = a_grid.tolist()
    x_list = x_grid.tolist()

    for t in range(1, T):        
        for n in range(0,N):  
            draw = np.random.uniform()
            draw2 = np.random.uniform()
            m1_state[t,n] = g_m1[a_list.index(a_state[t-1,n]),input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),x_list.index(x_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]
            m2_state[t,n] = g_m2[a_list.index(a_state[t-1,n]),input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),x_list.index(x_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]
            a_state[t,n] = g_a[a_list.index(a_state[t-1,n]),input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),x_list.index(x_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]          
            x_state[t,n] = g_x[a_list.index(a_state[t-1,n]),input1_list.index(m1_state[t-1,n]),input2_list.index(m2_state[t-1,n]),x_list.index(x_state[t-1,n]),int(θ_shock[t-1, n]), int(ε_shock[t-1, n])]          

            θ_shock[t, n] = int(draw < pi_θ[int(θ_shock[t-1, n]),1])
            ε_shock[t,n] =  int(draw2 < pi_ε[int(ε_shock[t-1, n]),1])
            y_state [t,n] = y1(m1_state[t-1,n],θ_grid[int(θ_shock[t-1, n])], ε_grid[int(ε_shock[t-1, n])]) +y2(m2_state[t-1,n], ε_grid[int(ε_shock[t-1, n])])
        mean_m1[t] = np.mean(m1_state[t,:]) 
        mean_m2[t] = np.mean(m2_state[t,:])
        mean_a[t] = np.mean(a_state[t,:])
        mean_y[t] = np.mean(y_state[t,:])
            
    return θ_shock[T-1,:], ε_shock[T-1,:], m1_state[T-1,:], m2_state[T-1,:] , a_state[T-1,:], y_state[T-1,:], x_state[T-1,:], mean_m1, mean_m2, mean_a, mean_y, mean_x

θ_shock, ε_shock, m1_state, m2_state, a_state, y_state, x_state, mean_m1, mean_m2, mean_a, mean_y, mean_x = invariant_distr(pi_θ, pi_ε, T=T, N=N, θ_h_0=θ_h_0 , ε_h_0 = ε_h_0)






#%% Input states
fig, ax = plt.subplots()
ax.plot(range(0,T), mean_m1, label='Average_input1')
ax.legend()
ax.set_xlabel('Time')
ax.set_title('Average input1 across time')
plt.show() 

fig, ax = plt.subplots()
ax.plot(range(0,T), mean_m2, label='Average_input2')
ax.legend()
ax.set_xlabel('Time')
ax.set_title('Average input2 across time')
plt.show() 

## Sample statistics of the input1 state
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(m1_state, label="input1s")
plt.title('Distribution of input1s')
plt.xlabel('input1s')
plt.ylabel("Density")
plt.legend()
plt.show()

## Sample statistics of the input2 state
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(m2_state, label="input2")
plt.title('Distribution of input2')
plt.xlabel('input2')
plt.ylabel("Density")
plt.legend()
plt.show()

mean_m1t = mean_m1[T-1]
var_m1t = np.var(m1_state)
sd_m1t = np.sqrt(var_m1t)

mean_m2t = mean_m2[T-1]
var_m2t = np.var(m2_state)
sd_m2t = np.sqrt(var_m2t)




gini_input1s = gini(m1_state)
gini_m2 = gini(m2_state)

input1s_stats = [mean_m1t, sd_m1t, gini_input1s]
input2_stats = [mean_m2t, sd_m2t, gini_m2]

print(input1s_stats)
print(input2_stats)


#%% Assets 

fig, ax = plt.subplots()
ax.plot(range(0,T), mean_a, label='Average_asset')
ax.legend()
ax.set_xlabel('Time')
ax.set_title('Average asset across time')
plt.show() 

## Sample statistics of the assets state
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(a_state, label="Assets")
plt.title('Distribution of Assets')
plt.xlabel('Assets')
plt.ylabel("Density")
plt.legend()
plt.show()


mean_at = mean_a[T-1]
var_at = np.var(a_state)
sd_at = np.sqrt(var_at)


gini_a = gini(a_state)


a_stats = [mean_at, sd_at, gini_a]

print(a_stats)


#%% Insurance

fig, ax = plt.subplots()
ax.plot(range(0,T), mean_a, label='Average Insurance')
ax.legend()
ax.set_xlabel('Time')
ax.set_title('Average insurance across time')
plt.show() 

## Sample statistics of the assets state
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(a_state, label="Insurance")
plt.title('Distribution of Insurance')
plt.xlabel('Insurance')
plt.ylabel("Density")
plt.legend()
plt.show()


mean_xt = mean_x[T-1]
var_xt = np.var(x_state)
sd_xt = np.sqrt(var_xt)


gini_x = gini(x_state)


x_stats = [mean_xt, sd_xt, gini_x]

print(x_stats)

#%% Output

fig, ax = plt.subplots()
ax.plot(range(0,T), mean_y, label='Average_output')
ax.legend()
ax.set_xlabel('Time')
ax.set_title('Average output across time')
plt.show() 

## Sample statistics of the assets state
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(y_state, label="Assets")
plt.title('Distribution of Output')
plt.xlabel('Output')
plt.ylabel("Density")
plt.legend()
plt.show()


mean_yt = mean_y[T-1]
var_yt = np.var(y_state)
sd_yt = np.sqrt(var_yt)


gini_y = gini(y_state)


y_stats = [mean_yt, sd_yt, gini_y]

print(y_stats)



