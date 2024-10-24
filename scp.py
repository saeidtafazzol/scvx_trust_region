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
import numpy as np
import cvxpy as cp
import scipy.integrate
from scipy.integrate import odeint

class cvx_program:
    def __init__(self, dynamics, option):
        self.option = option.copy()
        self.dynamics = dynamics  # Set dynamics as an attribute of the class

        # Set parameters from option, using default values if not provided
        self.rho0 = option.get('rho0', 0.04)
        self.rho1 = option.get('rho1', 0.2)
        self.rho2 = option.get('rho2', 0.7)

        self.trust_region = option.get('trust_region', np.array([1e1, 1e-1, 1e1, 1e0, 1e0, 1e1, 3e1]))
        self.min_trust = self.trust_region/1e3

        self.alpha = option.get('alpha', 1.5)
        self.beta = option.get('beta', 1.5)

        self.tolerance = option.get('tolerance', 5e-3)
        self.C = option.get('C', 1.0)

        # Set variables from option
        self.N = option['N']
        self.dt = option['dt']
        self.t_f = option['t_f']
        self.T_max = option['T_max']
        self.c = option['c']

        self.n_x = dynamics.n_x
        self.n_u = dynamics.n_u

        self.x_i = option['x_i']
        self.x_f = option['x_f']
        self.z_0 = option['z_0']

        # Initialize trajectories and controls
        self.x_p = option['init_traj']
        self.z_p = option['init_z_map']
        self.u_p = option['init_control']

        self.previous_nonlinear_cost = None
        self.previous_linear_cost = None

        # Initialize matrices for discretization
        self.A_bar = np.zeros([self.N - 1, self.n_x, self.n_x])
        self.B_bar = np.zeros([self.N - 1, self.n_x, self.n_u])
        self.C_bar = np.zeros([self.N - 1, self.n_x, self.n_u])
        self.z_bar = np.zeros([self.N - 1, self.n_x])

        # Vector indices for flat matrices
        x_end = self.n_x
        A_bar_end = self.n_x * (1 + self.n_x)
        B_bar_end = self.n_x * (1 + self.n_x + self.n_u)
        C_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_u)
        z_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_u + 1)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.C_bar_ind = slice(B_bar_end, C_bar_end)
        self.z_bar_ind = slice(C_bar_end, z_bar_end)

        self.V0 = np.zeros((self.n_x * (1 + self.n_x + self.n_u + self.n_u + 1),))
        self.V0[self.A_bar_ind] = np.eye(self.n_x).reshape(-1)

    def piecewise_ode(self, x, t, u0, u1):
        u = u0 + (t / self.dt) * (u1 - u0)
        if self.dynamics.backend=='sympy':

            return np.squeeze(self.dynamics.state_dot_fun(*x, *u))
        else:
            return np.squeeze(self.dynamics.state_dot_fun(x, u))

    def piecewise_ode_pmass(self, z, t, u0, u1):
        u = u0 + (t / self.dt) * (u1 - u0)
        return -u / self.c

    def calculate_discretization(self, X, U):
        """
        Calculate discretization for given states, inputs and total time.
        """
        for i in range(self.N - 1):
            self.V0[self.x_ind] = X[i, :]
            V = odeint(
                self._ode_dVdt,
                self.V0,
                (0, self.dt),
                args=(U[i, :], U[i + 1, :])
            )[1, :]

            # Flatten matrices in column-major (Fortran) order for CVXPY
            Phi = V[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[i, :] = Phi
            self.B_bar[i, :] = Phi @ V[self.B_bar_ind].reshape((self.n_x, self.n_u))
            self.C_bar[i, :] = Phi @ V[self.C_bar_ind].reshape((self.n_x, self.n_u))
            self.z_bar[i, :] = Phi @ V[self.z_bar_ind]

        return self.A_bar, self.B_bar, self.C_bar, self.z_bar

    def _ode_dVdt(self, V, t, u_t0, u_t1):
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        x = V[self.x_ind]
        u = u_t0 + (t / self.dt) * (u_t1 - u_t0)

        Phi_A_xi = np.linalg.inv(V[self.A_bar_ind].reshape((self.n_x, self.n_x)))
        if self.dynamics.backend=='sympy':
            A_der_subs = self.dynamics.A_der_fun(*x).squeeze().T
            A_subs = self.dynamics.A_fun(*x).squeeze()
            B_subs = self.dynamics.B_fun(*x).squeeze()
            f_subs = self.dynamics.state_dot_fun(*x, *u).squeeze()
            dVdt = np.zeros_like(V)
            dVdt[self.x_ind] = f_subs
            dVdt[self.A_bar_ind] = (A_der_subs @ V[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
            dVdt[self.B_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1) * alpha
            dVdt[self.C_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1) * beta
            z_t = A_subs - A_der_subs @ x

            dVdt[self.z_bar_ind] = Phi_A_xi @ z_t
        else:
            A_der_subs = self.dynamics.A_der_fun(x).squeeze()
            B_der_subs = self.dynamics.B_der_fun(x).squeeze()
            A_subs = self.dynamics.A_fun(x).squeeze()
            B_subs = self.dynamics.B_fun(x).squeeze()
            f_subs = self.dynamics.state_dot_fun(x, u).squeeze()

            dVdt = np.zeros_like(V)
            dVdt[self.x_ind] = f_subs
            dVdt[self.A_bar_ind] = ((np.einsum('ax,xy->ay',A_der_subs , V[self.A_bar_ind].reshape((self.n_x, self.n_x))))).reshape(-1) #+ np.einsum('xuo,xy,u->oy',B_der_subs,V[self.A_bar_ind].reshape((self.n_x, self.n_x)),u)
            dVdt[self.B_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1) * alpha
            dVdt[self.C_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1) * beta
            z_t = A_subs - np.einsum('ax,x->a',A_der_subs , x)# - np.einsum('xuo,x,u->o',B_der_subs,x,u)

            dVdt[self.z_bar_ind] = Phi_A_xi @ z_t


        return dVdt

    def get_prog(self):
        x_dim = self.n_x
        u_dim = self.n_u
        N = self.N
        dt = self.dt
        t_f = self.t_f
        T_max = self.T_max
        c = self.c

        x = cp.Variable((N, x_dim))
        z = cp.Variable(N)
        tau = cp.Variable((N, u_dim))
        tau_len = cp.Variable(N)
        h = cp.Variable((N - 1, x_dim + 1))

        self.x = x
        self.z = z
        self.tau = tau
        self.tau_len = tau_len
        self.h = h

        X = cp.hstack([x, z[:, None]])
        X_p = np.hstack([self.x_p, self.z_p[:, None]])

        objective = self.C * cp.sum(cp.abs(h))

        constraints = []

        constraints.append(cp.norm_inf(X - X_p, axis=0) <= self.trust_region)

        constraints.append(x[0, :] == self.x_i)
        constraints.append(x[-1, :] == self.x_f)
        constraints.append(z[0] == self.z_0)

        constraints.append(cp.norm(tau, 2, axis=1) <= tau_len)

        A_bar, B_bar, C_bar, z_bar = self.calculate_discretization(self.x_p, self.u_p)

        for i in range(N - 1):
            objective += dt / 2 * (tau_len[i] + tau_len[i + 1])

            constraints.append(
                x[i + 1, :] == A_bar[i] @ x[i, :] + B_bar[i] @ tau[i, :] +
                C_bar[i] @ tau[i + 1, :] + z_bar[i] + h[i, :x_dim]
            )
            constraints.append(
                z[i + 1] - z[i] == dt / 2 * (-tau_len[i] - tau_len[i + 1]) / c + h[i, x_dim]
            )

            constraints.append(
                tau_len[i] <= self.T_max * cp.exp(-self.z_p[i]) * (1 - (z[i] - self.z_p[i]))
            )

        constraints.append(
            tau_len[N - 1] <= self.T_max * cp.exp(-self.z_p[N - 1]) * (1 - (z[N - 1] - self.z_p[N - 1]))
        )

        return cp.Minimize(objective), constraints

    def piece_wise_int(self, x_l, z_l, taus, tau_lens):
        x_nl = np.zeros_like(x_l)
        z_nl = np.zeros_like(z_l)
        x_nl[0, :] = x_l[0, :]
        z_nl[0] = z_l[0]

        for i in range(self.N - 1):
            x_nl[i + 1, :] = scipy.integrate.odeint(
                self.piecewise_ode,
                x_l[i, :],
                (0, self.dt),
                args=(taus[i, :], taus[i + 1, :])
            )[1, :]
            z_nl[i + 1] = scipy.integrate.odeint(
                self.piecewise_ode_pmass,
                z_l[i],
                (0, self.dt),
                args=(tau_lens[i], tau_lens[i + 1])
            )[1, :]

        return x_nl, z_nl

    def cal_nonlinear_cost(self):
        if self.x.value is None or self.tau_len.value is None:
            return None

        x_nl, z_nl = self.piece_wise_int(
            self.x.value, self.z.value, self.tau.value, self.tau_len.value
        )
        cost = (
            (self.tau_len.value[1:-1].sum() +
             self.tau_len.value[0] / 2 +
             self.tau_len.value[-1] / 2) * self.t_f / (self.N - 1) +
            self.C * np.sum(np.abs(x_nl - self.x.value)) +
            self.C * np.sum(np.abs(z_nl - self.z.value))
        )
        return cost

    def optimize(self):
        obj, const = self.get_prog()
        prob = cp.Problem(obj, const)
        linear_cost = prob.solve(solver=cp.ECOS)
        nonlinear_cost = self.cal_nonlinear_cost()

        self.linear_cost = linear_cost
        self.nonlinear_cost = nonlinear_cost

        actual = None
        predicted = None
        if self.previous_nonlinear_cost is not None:
            predicted = self.previous_nonlinear_cost - linear_cost
            actual = self.previous_nonlinear_cost - nonlinear_cost

            ratio = actual / predicted if predicted != 0 else 0

            if ratio < self.rho0:
                self.trust_region /= self.alpha
            else:
                if ratio < self.rho1:
                    self.trust_region /= self.alpha
                elif self.rho2 <= ratio:
                    self.trust_region *= self.beta
                self.x_p = np.copy(self.x.value)
                self.z_p = np.copy(self.z.value)
                self.u_p = np.copy(self.tau.value)
        else:
            self.x_p = np.copy(self.x.value)
            self.z_p = np.copy(self.z.value)
            self.u_p = np.copy(self.tau.value)


        self.previous_nonlinear_cost = nonlinear_cost
        self.previous_linear_cost = linear_cost

        self.trust_region = np.maximum(self.trust_region, self.min_trust)
        print(predicted)
        if predicted is not None:
            return predicted < self.tolerance, linear_cost, nonlinear_cost
