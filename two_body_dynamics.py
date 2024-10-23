import coordinate_conversion
import numpy as np
import sympy as sp
import scipy
import scipy.optimize
import scipy.integrate
import time
import matplotlib.pyplot as plt
from sympy.utilities.autowrap import autowrap
import matplotlib
import cvxpy as cp
from scipy.integrate import odeint
plt.style.use('dark_background')

class two_body:
    def __init__(self,mode='mee') -> None:
            self.mu = sp.Symbol('mu')
            self.c = sp.Symbol('c')

            self.tau = sp.Matrix(3, 1, sp.symbols('tau0:3'))

            tau_len = sp.sqrt((self.tau.T@self.tau)[0])


            self.params = sp.Matrix([self.mu,self.c])

            if mode == 'mee':

                self.p, self.f, self.g, self.h, self.k, self.L,self.z = sp.symbols('p f g h k L z')

                self.states = sp.Matrix(6,1, [ self.p, self.f, self.g, self.h, self.k, self.L] )


                q = 1 + self.f*sp.cos(self.L) + self.g*sp.sin(self.L)
                s2 = 1 + self.h**2 + self.k**2

                self.A = sp.Matrix([0, 0, 0, 0, 0, sp.sqrt(self.p)*(q/self.p)**2])

                self.B = sp.Matrix([
                    [0,                2*self.p/q*sp.sqrt(self.p),                            0],
                    [sp.sqrt(self.p)*sp.sin(self.L), sp.sqrt(self.p)/q*((q+1)*sp.cos(self.L)+self.f), -sp.sqrt(self.p)*self.g/q*(self.h*sp.sin(self.L)-self.k*sp.cos(self.L))],
                    [-sp.sqrt(self.p)*sp.cos(self.L), sp.sqrt(self.p)/q*((q+1)*sp.sin(self.L)+self.g),  sp.sqrt(self.p)*self.f/q*(self.h*sp.sin(self.L)-self.k*sp.cos(self.L))],
                    [0,                              0,           sp.sqrt(self.p)*s2*sp.cos(self.L)/2/q],
                    [0,                              0,           sp.sqrt(self.p)*s2*sp.sin(self.L)/2/q],
                    [0,                              0,   sp.sqrt(self.p)/q*(self.h*sp.sin(self.L)-self.k*sp.cos(self.L))],
                ])
            elif mode =='sph':
                self.r, self.theta, self.phi, self.v_r, self.v_theta, self.v_phi = sp.symbols('r theta phi v_r v_theta v_phi')
                
                self.states = sp.Matrix(6,1, [self.r, self.theta, self.phi, self.v_r, self.v_theta, self.v_phi])
                
                self.A = sp.Matrix([
                    self.v_r,
                    self.v_theta / self.r,
                    self.v_phi / (self.r * sp.sin(self.theta)),
                    self.v_theta**2 / self.r + self.v_phi**2 / self.r - self.mu / self.r**2,
                    -self.v_r * self.v_theta / self.r + self.v_phi**2 * sp.cos(self.theta) / (self.r * sp.sin(self.theta)),
                    -self.v_r * self.v_phi / self.r - self.v_theta * self.v_phi * sp.cos(self.theta) / (self.r * sp.sin(self.theta))])
                
                            
                self.B = sp.Matrix([
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [1.0,0,0],
                    [0,1.0,0],
                    [0,0,1.0],                    
                ])
            self.n_x = self.A.shape[0]
            self.n_u = self.B.shape[1]
            self.state_dot = self.B*self.tau + self.A
            self.zdot =  -tau_len/self.c
            self.A_der = sp.ImmutableDenseMatrix(self.A.diff(self.states).reshape(6,6))

            self.state_dot_x = sp.ImmutableDenseMatrix(self.state_dot.diff(self.states).reshape(6,6).transpose())
            self.state_dot_u = sp.ImmutableDenseMatrix(self.state_dot.diff(self.tau).reshape(3,6).transpose())
            

    def set_params(self,mu,c):
        param_dic = {self.mu:mu, self.c: c}
        self.state_dot_sub = self.state_dot.subs(param_dic)
        self.zdot_sub = self.zdot.subs(param_dic)
        self.A_sub = self.A.subs(param_dic)
        self.A_der_sub = self.A_der.subs(param_dic)

    def compile(self):

        if self.state_dot_sub is None:
            print('set params first')
            return



        
        self.state_dot_fun = autowrap(self.state_dot_sub,args= (*self.states,*self.tau),backend='cython')
        self.zdot_fun = autowrap(self.zdot_sub,args= (*self.tau,),backend='cython')
        self.A_fun_temp = autowrap(self.A_sub,args=(*self.states,),backend='cython')
        self.A_fun =  lambda *x,: self.A_fun_temp(*x,).squeeze() 
        self.A_der_fun = autowrap(self.A_der_sub,args=(*self.states,),backend='cython')
        self.B_fun = autowrap(self.B,args=(*self.states,),backend='cython')
        
        self.state_dot_x_fun = autowrap(self.state_dot_x,args=(*self.states,*self.tau),backend='cython')
        self.state_dot_u_fun = autowrap(self.state_dot_u,args=(*self.states,),backend='cython')


def get_planet_ode(mu):
    def planet_ode(_,x):
        xdot = np.zeros((6,))
        xdot[:3] = x[3:]
        r = x[:3]
        len_r = np.linalg.norm(r)    
        xdot[3:] = -(mu*r)/len_r**3

        return xdot
    return planet_ode