
class cvx_program:
    def __init__(self,dynamics,option):
        self.rho0 = 0.04
        self.rho1 = 0.2
        self.rho2 = 0.7

        self.trust_region = np.array([1e2,1.0,1e2,1.0,1.0,1e2,3e2])
        self.min_trust = self.trust_region/1e4

        self.alpha = 1.5
        self.beta = 1.5

        self.tolerance = 1e-2
        self.C = 1.0


        self.x_p = init_traj
        self.z_p = init_z_map

        self.u_p = init_control 

        self.previous_nonlinear_cost = None
        self.previous_linear_cost = None

        self.n_x = dynamics.n_x
        self.n_u = dynamics.n_u

        self.A_bar = np.zeros([N - 1 , dynamics.n_x , dynamics.n_x])
        self.B_bar = np.zeros([N - 1 , dynamics.n_x , dynamics.n_u])
        self.C_bar = np.zeros([N - 1 , dynamics.n_x , dynamics.n_u])
        self.z_bar = np.zeros([N - 1 , dynamics.n_x])

        # vector indices for flat matrices
        x_end = dynamics.n_x
        A_bar_end = dynamics.n_x * (1 + dynamics.n_x)
        B_bar_end = dynamics.n_x * (1 + dynamics.n_x + dynamics.n_u)
        C_bar_end = dynamics.n_x * (1 + dynamics.n_x + dynamics.n_u + dynamics.n_u)
        z_bar_end = dynamics.n_x * (1 + dynamics.n_x + dynamics.n_u + dynamics.n_u + 1)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.C_bar_ind = slice(B_bar_end, C_bar_end)
        self.z_bar_ind = slice(C_bar_end, z_bar_end)


        self.V0 = np.zeros((dynamics.n_x * (1 + dynamics.n_x + dynamics.n_u + dynamics.n_u + 1),))
        self.V0[self.A_bar_ind] = np.eye(dynamics.n_x).reshape(-1)

        self.dt = dt

    def piecewise_ode(self,x,t,u0,u1):
        u = u0 + (t / dt) * (u1 - u0)
        return np.squeeze(dynamics.state_dot_fun(*x,*u))
    
    def piecewise_ode_pmass(self,z,t,u0,u1):
        u = u0 + (t / dt) * (u1 - u0)
        return -u/c
    
    def calculate_discretization(self, X, U):
        """
        Calculate discretization for given states, inputs and total time.

        :param X: Matrix of states for all time points
        :param U: Matrix of inputs for all time points
        :return: The discretization matrices
        """
        for i in range(N-1):
            self.V0[self.x_ind] = X[i,:]
            V = np.array(odeint(self._ode_dVdt, self.V0, (0, self.dt), args=(U[i, :], U[i+1, :]))[1, :])

            # flatten matrices in column-major (Fortran) order for CVXPY
            Phi = V[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[i, :] = Phi
            self.B_bar[i, :] = np.matmul(Phi, V[self.B_bar_ind].reshape((self.n_x, self.n_u)))
            self.C_bar[i, :] = np.matmul(Phi, V[self.C_bar_ind].reshape((self.n_x, self.n_u)))
            self.z_bar[i, :] = np.matmul(Phi, V[self.z_bar_ind])

        return self.A_bar, self.B_bar, self.C_bar, self.z_bar


    def _ode_dVdt(self, V, t, u_t0, u_t1):
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        x = V[self.x_ind]
        u = u_t0 + (t / self.dt) * (u_t1 - u_t0)

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(V[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        # A_subs = dynamics.state_dot_x_fun(*x, *u)
        # B_subs = dynamics.state_dot_u_fun(*x)
        # f_subs = dynamics.state_dot_fun(*x, *u).squeeze()


        A_der_subs = dynamics.A_der_fun(*x).squeeze().T
        A_subs = dynamics.A_fun(*x).squeeze()
        B_subs = dynamics.B_fun(*x).squeeze()
        f_subs = dynamics.state_dot_fun(*x, *u).squeeze()


        # print(A)
        dVdt = np.zeros_like(V)
        dVdt[self.x_ind] = f_subs
        dVdt[self.A_bar_ind] = np.matmul(A_der_subs, V[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dVdt[self.B_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * alpha
        dVdt[self.C_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * beta
        z_t = A_subs - np.matmul(A_der_subs, x)

        dVdt[self.z_bar_ind] = np.matmul(Phi_A_xi, z_t)
        return dVdt
    


    def get_prog(self):

        x = cp.Variable((N,x_dim))
        z = cp.Variable((N))
        tau = cp.Variable((N,tau_dim))
        tau_len = cp.Variable((N,))
        h = cp.Variable((N,x_dim+1))


        self.x = x
        self.z = z
        self.tau = tau
        self.tau_len = tau_len
        self.h = h        
        
        dt = t_f/(N-1)
        
        X = cp.hstack((x,z[:,None]))
        X_p = np.hstack((self.x_p,self.z_p[:,None]))

        objective = self.C*cp.sum(cp.abs(h))

        constraints = []

        constraints.append(cp.norm_inf(X-X_p,axis=0) <=  self.trust_region)


        constraints.append(x_i == x[0])
        constraints.append(x_f == x[-1])
        constraints.append(z_0 == z[0])
        
        constraints.append(cp.pnorm(tau,2,1) <= tau_len)
        


        # A_pre = dynamics.A_fun(*self.x_p[0],)
        # A_der_pre = dynamics.A_der_fun(*self.x_p[0],)
        # B_pre = dynamics.B_fun(*self.x_p[0],)

        # A_var_pre = np.zeros((x_dim,))

        # for j in range(x_dim):
        #     A_var_pre = A_der_pre[j]*(x[0,j] - self.x_p[0,j]) + A_var_pre
        A_bar, B_bar, C_bar, z_bar = self.calculate_discretization(self.x_p,self.u_p)

        for i in range(N-1):
            # A = dynamics.A_fun(*self.x_p[i],)
            # A_der = dynamics.A_der_fun(*self.x_p[i],)
            # B = dynamics.B_fun(*self.x_p[i],)


            # A_var = np.zeros((x_dim,))


            # for j in range(x_dim):
            #     A_var = A_der[j]*(x[i,j] - self.x_p[i,j]) + A_var

            objective = dt/2*(tau_len[i] + tau_len[i+1]) + objective

            # xdot_i1 = A_pre + A_var_pre + B_pre@tau[i-1]
            # xdot_i2 = A + A_var + B@tau[i]


            constraints.append(x[i+1] == A_bar[i]@x[i] + B_bar[i]@tau[i] + C_bar[i]@tau[i+1] + z_bar[i] + h[i,:6])
            constraints.append(z[i+1] - z[i] == dt/2*(-tau_len[i] -tau_len[i+1])/c + h[i,6])

            constraints.append(tau_len[i] <= T_max * np.exp(-self.z_p[i])*(1 - (z[i] - self.z_p[i])))

            # A_pre = A
            # A_der_pre = A_der
            # B_pre = B
            # A_var_pre = A_var

        constraints.append(tau_len[N-1] <= T_max * np.exp(-self.z_p[N-1])*(1 - (z[N-1] - self.z_p[N-1])))

        
        return cp.Minimize(objective),constraints

            

    def piece_wise_int(self,x_l,z_l,taus,tau_lens):
        x_nl= np.zeros_like(x_l)
        z_nl = np.zeros_like(z_l)
        x_nl[0,:] = x_l[0,:]
        z_nl[0] = z_l[0]

        for i in range(N-1):
            x_nl[i+1, :] = scipy.integrate.odeint(self.piecewise_ode, x_l[i, :], (0, dt), args=(taus[i, :], taus[i+1, :]))[1, :]
            z_nl[i+1] = scipy.integrate.odeint(self.piecewise_ode_pmass, z_l[i], (0, dt), args=(tau_lens[i], tau_lens[i+1]))[1, :]

        return x_nl,z_nl


    def cal_nonlinear_cost(self):
        if self.x.value is None or self.tau_len.value is None:
            return None
        
        x_nl,z_nl = self.piece_wise_int(self.x.value,self.z.value,self.tau.value,self.tau_len.value)
        # print('z_nl',z_nl)
        # print('z_l', self.z.value + self.h.value[:,6])
        return  (self.tau_len.value[1:-1].sum() + self.tau_len.value[0]/2 + self.tau_len.value[-1]/2)*t_f/(N-1) + self.C*np.sum(np.abs(x_nl - self.x.value)) + self.C*np.sum(np.abs(z_nl - self.z.value))

    def optimize(self):
        obj,const = self.get_prog()
        prob = cp.Problem(obj,const)
        linear_cost = prob.solve(solver=cp.ECOS)
        nonlinear_cost = self.cal_nonlinear_cost()

        self.linear_cost,self.nonlinear_cost = linear_cost,nonlinear_cost
        # print(self.u_p.shape)

        actual = None
        predicted = None
        if self.previous_nonlinear_cost is not None:
            predicted = self.previous_nonlinear_cost - linear_cost
            actual = self.previous_nonlinear_cost - nonlinear_cost

            ratio = actual/predicted
            # print('costs',self.previous_nonlinear_cost,nonlinear_cost,self.previous_linear_cost,linear_cost)
            # print(ratio,self.trust_region)
            if ratio < self.rho0 and predicted>=0:
                self.trust_region/=self.alpha
                
                if predicted>self.tolerance:
                    return self.optimize()
            elif ratio<self.rho1:
                self.trust_region/=self.alpha
            elif self.rho2<= ratio:
                self.trust_region*=self.beta
        
        self.x_p = np.copy(self.x.value)
        self.z_p = np.copy(self.z.value)
        self.u_p = np.copy(self.tau.value)
        
        # self.trust_region = max(self.trust_region,self.min_trust)
        self.previous_nonlinear_cost = nonlinear_cost
        self.previous_linear_cost = linear_cost

        if predicted is not None:
            return predicted<self.tolerance , linear_cost, nonlinear_cost
