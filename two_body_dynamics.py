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
import casadi as ca
import os
from scipy.integrate import odeint
plt.style.use('dark_background')

class TwoBody:
    def __init__(self):
            
            self.backend = 'sympy'
            self.c = sp.Symbol('c')

            self.tau = sp.Matrix(3, 1, sp.symbols('tau0:3'))

            tau_len = sp.sqrt((self.tau.T@self.tau)[0])


            self.params = sp.Matrix([self.c])

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
            self.n_x = self.A.shape[0]
            self.n_u = self.B.shape[1]
            self.state_dot = self.B*self.tau + self.A
            self.zdot =  -tau_len/self.c
            self.A_der = sp.ImmutableDenseMatrix(self.A.diff(self.states).reshape(6,6))

            self.state_dot_x = sp.ImmutableDenseMatrix(self.state_dot.diff(self.states).reshape(6,6).transpose())
            self.state_dot_u = sp.ImmutableDenseMatrix(self.state_dot.diff(self.tau).reshape(3,6).transpose())
            

    def set_params(self,c):
        param_dic = {self.c: c}
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


class TwoBodyCa:
    def __init__(self):
        # Define symbolic variables
        self.backend = 'casadi'
        self.c = ca.MX.sym('c')
        self.tau = ca.MX.sym('tau', 3)
        self.states = ca.MX.sym('states', 6)

        p = self.states[0]
        f = self.states[1]
        g = self.states[2]
        h = self.states[3]
        k = self.states[4]
        L = self.states[5]

        tau0 = self.tau[0]
        tau1 = self.tau[1]
        tau2 = self.tau[2]

        self.n_x = 6
        self.n_u = 3

        # Additional variables
        q = 1 + f * ca.cos(L) + g * ca.sin(L)
        s2 = 1 + h**2 + k**2

        # Define A and B matrices
        self.A = ca.vertcat(0, 0, 0, 0, 0, ca.sqrt(p) * (q / p)**2)

        self.B = ca.MX(6, 3)
        # Define the B matrix using CasADi's vertcat and horzcat
        self.B = ca.vertcat(
        ca.horzcat(0, 2*p/q * ca.sqrt(p), 0),
        ca.horzcat(ca.sqrt(p)*ca.sin(L), ca.sqrt(p)/q*((q+1)*ca.cos(L)+f), -ca.sqrt(p)*g/q*(h*ca.sin(L)-k*ca.cos(L))),
        ca.horzcat(-ca.sqrt(p)*ca.cos(L), ca.sqrt(p)/q*((q+1)*ca.sin(L)+g), ca.sqrt(p)*f/q*(h*ca.sin(L)-k*ca.cos(L))),
        ca.horzcat(0, 0, ca.sqrt(p)*s2*ca.cos(L)/2/q),
        ca.horzcat(0, 0, ca.sqrt(p)*s2*ca.sin(L)/2/q),
        ca.horzcat(0, 0, ca.sqrt(p)/q*(h*ca.sin(L)-k*ca.cos(L)))
        )

        # Define state_dot and zdot
        self.state_dot = ca.mtimes(self.B, self.tau) + self.A
        tau_len = ca.sqrt(tau0**2 + tau1**2 + tau2**2)
        self.zdot = -tau_len / self.c

        # Define derivatives
        self.A_der = ca.jacobian(self.A, self.states)
        self.state_dot_x = ca.jacobian(self.state_dot, self.states)
        self.state_dot_u = ca.jacobian(self.state_dot, self.tau)

        # Second-order derivatives
        self.state_dot_xx = ca.jacobian(self.state_dot_x.reshape((-1, 1)), self.states)
        self.state_dot_xu = ca.jacobian(self.state_dot_x.reshape((-1, 1)), self.tau)

        # Derivative of B w.r.t. states
        self.B_der = ca.jacobian(self.B.reshape((-1,1)), self.states)

    def set_params(self, c_val):
        # Substitute parameter values using lists
        self.state_dot_sub = ca.substitute(self.state_dot, self.c, c_val)
        self.zdot_sub = ca.substitute(self.zdot, self.c, c_val)
        self.A_sub = ca.substitute(self.A, self.c, c_val)
        self.A_der_sub = ca.substitute(self.A_der, self.c, c_val)
        self.B_der_sub = ca.substitute(self.B_der, self.c, c_val)


    def compile(self):
        if self.state_dot_sub is None:
            print('Set parameters first')
            return

        # Generate and compile C code for each function
        self.state_dot_func_aux = ca.Function('state_dot_func', [self.states, self.tau], [self.state_dot_sub])
        self.state_dot_func_aux.generate('state_dot.c')
        os.system('gcc -fPIC -shared state_dot.c -o state_dot.so')

        self.zdot_func_aux = ca.Function('zdot_func', [self.tau], [self.zdot_sub])
        self.zdot_func_aux.generate('zdot.c')
        os.system('gcc -fPIC -shared zdot.c -o zdot.so')

        self.A_func_aux = ca.Function('A_func', [self.states], [self.A_sub])
        self.A_func_aux.generate('A.c')
        os.system('gcc -fPIC -shared A.c -o A.so')

        self.B_func_aux = ca.Function('B_func', [self.states], [self.B])
        self.B_func_aux.generate('B.c')
        os.system('gcc -fPIC -shared B.c -o B.so')


        self.A_der_func_aux = ca.Function('A_der_func', [self.states], [self.A_der_sub])
        self.A_der_func_aux.generate('A_der.c')
        os.system('gcc -fPIC -shared A_der.c -o A_der.so')

        self.state_dot_x_func_aux = ca.Function('state_dot_x_func', [self.states, self.tau], [self.state_dot_x])
        self.state_dot_x_func_aux.generate('state_dot_x.c')
        os.system('gcc -fPIC -shared state_dot_x.c -o state_dot_x.so')

        self.state_dot_u_func_aux = ca.Function('state_dot_u_func', [self.states], [self.state_dot_u])
        self.state_dot_u_func_aux.generate('state_dot_u.c')
        os.system('gcc -fPIC -shared state_dot_u.c -o state_dot_u.so')

        self.state_dot_xx_func_aux = ca.Function('state_dot_xx_func', [self.states, self.tau], [self.state_dot_xx])
        self.state_dot_xx_func_aux.generate('state_dot_xx.c')
        os.system('gcc -fPIC -shared state_dot_xx.c -o state_dot_xx.so')

        self.state_dot_xu_func_aux = ca.Function('state_dot_xu_func', [self.states, self.tau], [self.state_dot_xu])
        self.state_dot_xu_func_aux.generate('state_dot_xu.c')
        os.system('gcc -fPIC -shared state_dot_xu.c -o state_dot_xu.so')

        # Generate and compile the B_der function
        self.B_der_func_aux = ca.Function('B_der_func', [self.states], [self.B_der_sub])
        self.B_der_func_aux.generate('B_der.c')
        os.system('gcc -fPIC -shared B_der.c -o B_der.so')

        # Load the compiled shared libraries
        self.state_dot_func = ca.external('state_dot_func', './state_dot.so')
        self.zdot_func = ca.external('zdot_func', './zdot.so')
        self.A_func = ca.external('A_func', './A.so')
        self.B_func = ca.external('B_func', './B.so')
        self.A_der_func = ca.external('A_der_func', './A_der.so')
        self.state_dot_x_func = ca.external('state_dot_x_func', './state_dot_x.so')
        self.state_dot_u_func = ca.external('state_dot_u_func', './state_dot_u.so')
        self.state_dot_xx_func = ca.external('state_dot_xx_func', './state_dot_xx.so')
        self.state_dot_xu_func = ca.external('state_dot_xu_func', './state_dot_xu.so')
        self.B_der_func = ca.external('B_der_func', './B_der.so')

    def state_dot_fun(self, states, tau):
        states_casadi = ca.DM(states)
        tau_casadi = ca.DM(tau)
        result = self.state_dot_func(states_casadi, tau_casadi)
        return np.array(result.full()).flatten()

    def zdot_fun(self, tau):
        tau_casadi = ca.DM(tau)
        result = self.zdot_func(tau_casadi)
        return np.array(result.full()).flatten()

    def A_fun(self, states):
        states_casadi = ca.DM(states)
        result = self.A_func(states_casadi)
        return np.array(result.full()).flatten()

    def B_fun(self, states):
        states_casadi = ca.DM(states)
        result = self.B_func(states_casadi)
        return np.array(result.full()).reshape((6, 3))


    def A_der_fun(self, states):
        states_casadi = ca.DM(states)
        result = self.A_der_func(states_casadi)
        return np.array(result.full()).reshape((6, 6),order='F')

    def state_dot_x_fun(self, states, tau):
        states_casadi = ca.DM(states)
        tau_casadi = ca.DM(tau)
        result = self.state_dot_x_func(states_casadi, tau_casadi)
        return np.array(result.full()).reshape((6, 6))

    def state_dot_u_fun(self, states):
        states_casadi = ca.DM(states)
        result = self.state_dot_u_func(states_casadi)
        return np.array(result.full()).reshape((3, 6))

    def state_dot_xx_fun(self, states, tau):
        states_casadi = ca.DM(states)
        tau_casadi = ca.DM(tau)
        result = self.state_dot_xx_func(states_casadi, tau_casadi)
        return np.array(result.full()).reshape((6, 6, 6))
    
    def state_dot_xu_fun(self, states, tau):
        states_casadi = ca.DM(states)
        tau_casadi = ca.DM(tau)
        result = self.state_dot_xu_func(states_casadi, tau_casadi)
        return np.array(result.full()).reshape((3, 6, 6))

    def B_der_fun(self, states):
        states_casadi = ca.DM(states)
        result = self.B_der_func(states_casadi)
        return np.array(result.full()).reshape((6, 3, 6),order='F')

def get_planet_ode(mu):
    def planet_ode(_,x):
        xdot = np.zeros((6,))
        xdot[:3] = x[3:]
        r = x[:3]
        len_r = np.linalg.norm(r)    
        xdot[3:] = -(mu*r)/len_r**3

        return xdot
    return planet_ode