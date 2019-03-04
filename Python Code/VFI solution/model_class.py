# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:54:43 2019

@author: rodri
"""


# =============================================================================
# Class of the HH problem Agricultural DSGE
# =============================================================================

import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.optimize import minimize
from numba import njit
import matplotlib.pyplot as plt


class HH_Model:
    '''
    
    
    '''
    
    
    def __init__(self, A = np.exp(2), α = 0.4, B = np.exp(2), γ = 0.25, ρ = 2, β = 0.95,
                  b= 0, m_min=0.05, m_max=60, N_m=20, N_a = 5, a_max=60,  
                  N_x=20, c_bar=0.05,
                  r = 0.05, p = 0.3, q=0.5, x_min=0,
                  θ_grid = [0, 1], pi_θ = [[0.5,0.5],[0.5,0.5]], ε_grid = [0.5, 1], pi_ε = [[0.5,0.5],[0.3,0.7]]):
        
        self.A, self.B, self.α , self.γ,  self.ρ, self.β = A, B, α, γ, ρ, β 
        self.c_bar = c_bar
        self.N_a, self.N_m, self.N_x = N_a, N_m, N_x
        self.ε_grid = np.asarray(ε_grid)       
        self.θ_grid = np.asarray(θ_grid)
        self.N_θ, self.N_ε = len(np.asarray(θ_grid)), len(np.asarray(θ_grid))
        self.pi_ε = np.asarray(pi_ε)
        self.pi_θ = np.asarray(pi_θ)
       
        
        x_max = A*m_max**α
        self.x_max = x_max
        self.x_min = x_min
        
        self.q = q        
        self.r = r
        self.p = p
       
        self.b = b
        self.m_max = m_max
        self.m_min = m_min
        N_θ, N_ε = len(θ_grid),len(ε_grid)
        
        self.a_grid = np.linspace(b+1e-2, a_max, N_a) 
        self.m_grid = np.linspace(m_min, m_max, N_m) 
        self.x_grid = np.linspace(x_min, x_max, N_x)
        
        # === Set empty Value and Policy functions: === #
        self.N = N_a*N_m*N_m*N_ε*N_θ
        self.V_new = np.empty((N_a,N_m,N_m,N_ε,N_θ))
        self.policy_a = np.empty((N_a,N_m,N_m,N_ε,N_θ))
        self.policy_m1 = np.empty((N_a,N_m,N_m,N_ε,N_θ))
        self.policy_m2 = np.empty((N_a,N_m,N_m,N_ε,N_θ))
        self.policy_c = np.empty((N_a,N_m,N_m,N_ε,N_θ))
    
    
    
    #@njit
    def u(self,c_array):
        output = np.zeros(len(c_array))
        for i, c  in enumerate(c_array):
            if c<self.c_bar:
                output[i] = -np.inf

            else:     
                output[i] = ((c-self.c_bar)**(1-self.ρ)) / (1-self.ρ)
        return output
    
    def u_float(self, c):
        if c<self.c_bar:
            return -np.inf

        else:     
            return ((c-self.c_bar)**(1-self.ρ)) / (1-self.ρ)
        

    def u_prime(self,c_array):
        if self.ρ ==1:
            return 1/(c_array-self.c_bar)
        else:     
            return (c_array-self.c_bar)**(-self.ρ)


    def y1(self,m1,θ,ε):
        return ε*θ*self.A*m1**self.α

    def y2(self, m2,ε): 
        return ε*self.B*m2**self.γ

    
    def generate_V_guess(self,):        
        V_0_func = lambda a, m1, m2, θ, ε: self.u(θ*ε*self.A*m1**self.α +ε*self.B*m2**self.γ+(1+self.r)*a -self.p*(m1 +m2)-a)/(1-self.β)
        V_guess = np.zeros(self.N_a,self.N_m,self.N_m,self.N_ε,self.N_θ)
        for i,a in enumerate(self.a_grid):
            for j,m1 in enumerate(self.m_grid):
                for k,m2 in enumerate(self.m_grid):
                    for l,θ in enumerate(self.θ_vals):
                        for m,ε in enumerate(self.ε_vals):
                            V_guess[i,j,k,l,m] = np.nan_to_num(V_0_func(a, m1, m2, θ, ε))
                            
        return V_guess
            
                    
    #### Plots ======================================
    def plot_policy(self,policy, policy_name, m0=0, save=False, folder='C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/Model assets'):      
        fig,ax = plt.subplots()
        ax.plot(self.a_grid, np.array(policy[:,m0,m0,1,1]), label=policy_name+r' under $\theta=1, \varepsilon=1)$')
        ax.plot(self.a_grid, np.array(policy[:,m0,m0,0,1]), label=policy_name+r' under $\theta=0, \varepsilon=1)$')
        ax.plot(self.a_grid, np.array(policy[:,m0,m0,1,0]), label=policy_name+r' under $\theta=1, \varepsilon=0)$')
        ax.plot(self.a_grid, np.array(policy[:,m0,m0,0,0]), label=policy_name+r' under $\theta=0, \varepsilon=0)$')
        ax.legend()
        ax.set_xlabel('assets')
        ax.set_title(policy_name)
        plt.show() 
        
        
        
    #@jit 
    def bellman_operator(self, V, u, x0):
        """
        Returns the approximate value function TV by applying the
        Bellman operator associated with the model to the function V.


        Parameters
        ----------
        V : array_like(float)
            Array representing an approximate value function. Previous Value function
        u: function
            Utility function.
        x0: Initial values for the maximization routine.


        Returns
        -------
        s_policy : array_like(float)
            The greedy policy computed from V.  Only returned if
            return_policies == True
        new_V : array_like(float)
            The updated value function Tv, as an array representing the
            values TV(x) over x in x_grid.

        """
        # === simplify names, set up arrays, etc. === #
        
        a_grid, m_grid, θ_vals, ε_vals =  self.a_grid, self.m_grid, self.θ_vals, self.ε_vals
        β, x0 = self.x0, self.β
        r, p = self.r, self.p
        θ_vals, ε_vals =  self.θ_vals, self.ε_vals
        pi_ε, pi_θ = self.pi_ε, self.pi_θ
        b = self.b
        α,γ, A, B = self.α, self.γ, self.A, self.B
        a_max = self.a_max
        m_max=self.m_max
        
        #Number of possible states        
        N = len(a_grid)*len(m_grid)*len(m_grid)*len(θ_vals)*len(ε_vals)
 
        # === Set empty Value and Policy functions: === #
        V_new = np.zeros(N).reshape(len(a_grid),len(m_grid),len(m_grid),len(θ_vals),len(ε_vals))
        policy_a = np.zeros(N).reshape(len(a_grid),len(m_grid),len(m_grid),len(θ_vals),len(ε_vals))
        policy_m1 = np.zeros(N).reshape(len(a_grid),len(m_grid),len(m_grid),len(θ_vals),len(ε_vals))
        policy_m2 = np.zeros(N).reshape(len(a_grid),len(m_grid),len(m_grid),len(θ_vals),len(ε_vals))
        
        
        # === Interpolation function for tomorrow's value function:  === #
        my_interpolating_function = rgi((a_grid, m_grid, m_grid, θ_vals,ε_vals ), V)
        Vf = lambda a, m1, m2, θ, ε: my_interpolating_function(np.array([a,m1,m2,θ,ε]).T)
        
        #Given state variables, find optimal choices (g_a, g_m1, g_m2):
        for i_a, a in enumerate(a_grid):
            print(i_a)
            for i_m1, m1 in enumerate(m_grid):
                for i_m2, m2 in enumerate(m_grid):
                    for i1,i_θ in enumerate(θ_vals):
                        for i2,i_ε in enumerate(ε_vals):
                            for j1, j_θ in enumerate(θ_vals):  #try a zip format?? it makes function not scalar?
                                for j2, j_ε in enumerate(ε_vals):
                                    V_next = np.sum(Vf(a,m1,m2,j_θ,j_ε)*pi_θ[i1, j1]*pi_ε[i2,j2])
                                    def objective(x):
                                        g_a = x[0]
                                        g_m1 = x[1]
                                        g_m2 = x[2]
                                        return -u(i_θ*i_ε*A*m1**α +i_ε*B*m2**γ +(1+r)*a -p*(g_m1 + g_m2) -g_a) -β*V_next  
                                    x0 = np.array([3,1,1])
                                    bounds = ((-b,a_max), (0, m_max), (0, m_max)) 
                                    res = minimize(objective, x0, method='SLSQP' , bounds=bounds)
                                    policy_a[i_a, i_m1, i_m2, i1, i2] = res.x[0]
                                    policy_m1[i_a,i_m1,i_m2,i1,i2] = res.x[1]
                                    policy_m2[i_a,i_m1,i_m2,i1,i2] = res.x[2]
                                    V_new[i_a,i_m1,i_m2,i1,i2] =res.fun
                                    print(res.success)
                                    print(V_next)
                                    print(V_new)
                                   
        
        
        

        return policy_a, policy_m1, policy_m2, V_new