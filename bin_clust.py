import numpy as np
import pandas as pd


class binclust(object):
    """Row clustering binary data with two coding scheme"""

    def __init__(self, X, R=2):
        # X is a binary data frame
        if isinstance(X, pd.DataFrame):
            self.X = X.as_matrix()
        else:
            self.X = np.array(X)
        # Specifying number of clusters Default is 3
        self.R = R
        self.Precis = -20

    @property
    def para_init(self):
        self.N = self.X.shape[0]
        self.M = self.X.shape[1]
        # M by R array of initial thetas
        theta = np.random.rand(self.M, self.R)
        # initial cluster weights
        pi_r = np.random.rand(self.R)
        pi_r = self.log_sum_exp(pi_r)
        pi_r /= sum(pi_r)
        # initial coding scheme weights
        pi_s = np.random.rand(1)
        # initial posterior mean of weights R by N
        pi_N = np.tile(pi_r, (self.N, 1)).T
        # initial posterior mean of coding scheme weights 1 by N
        s_N = pi_s * np.ones(self.N)
        return theta, pi_r, pi_s, pi_N, s_N

    def E_step(self, theta, pi_r, pi_s, pi_N, s_N):
        # Pre-allocation for speed
        pi_N_new = np.zeros((self.R, self.N))
        s_N_new = np.zeros((self.N))

        # Variational E-step
        i = 0
        loglik = 0

        # Initialize terms for M-step
        M_term1 = np.zeros(self.R)
        M_term2 = 0
        for y in self.X:
            # The variational E_step
            pi_N_new[:, i] = self.compute_pi_N(y, theta, pi_r, s_N[i])
            temp = self.compute_s_N(y, theta, pi_s, pi_N_new[:, i])
            s_N_new[i] = temp[0]

            # Compute log-likelihood
            loglik += s_N_new[i] * temp[1]
            loglik += (1 - s_N_new[i]) * temp[2]
            loglik += np.dot(pi_N_new[:, i], np.log(pi_r))

            # Prepare for summation terms for M-step
            M_term1 += pi_N_new[:, i]
            M_term2 += s_N_new[i]
            i += 1
        s_N_temp = np.tile(s_N_new, (self.R, 1))
        one_s_N_temp = 1 - s_N_temp
        temp1 = np.dot(np.multiply(pi_N_new, one_s_N_temp), 1 - self.X).T
        temp2 = np.dot(np.multiply(pi_N_new, s_N_temp), self.X).T
        M_term3 = temp1 + temp2
        return pi_N_new, s_N_new, M_term1, M_term2, M_term3, loglik

    def M_step(self, pi_N_new, s_N_new, M_term1, M_term2, M_term3):
        # Pre-allocate for speed
        pi_r_new = np.zeros(self.R)
        pi_s_new = 0

        # M_step
        pi_r_new = M_term1 / sum(M_term1)

        pi_s_new = M_term2 / self.N
        cut_off = np.exp(self.Precis)
        if pi_s_new < cut_off:
            pi_s_new = cut_off
        elif pi_s_new > 1 - cut_off:
            pi_s_new = 1 - cut_off
        else:
            pass

        theta_new = np.divide(M_term3, np.tile(M_term1, (self.M, 1)))
        return theta_new, pi_r_new, pi_s_new, pi_N_new, s_N_new

    def log_sum_exp(self, x):
        # chopp-off precision = e^-100
        k = self.Precis
        e = x - np.max(x)
        y = np.exp(e) / sum(np.exp(e))
        y[e < k] = 0
        y = y / sum(y)
        return y

    def compute_pi_N(self, y, theta, pi_r, s_N):
        # 1 - y
        one_y = 1 - y

        # log(theta) in M by R
        log_theta = np.log(theta)
        one_log_theta = np.log(1 - theta)

        # All M by R terms in equation 12
        temp1 = np.log(pi_r) + s_N * (np.dot(y, log_theta) + np.dot(one_y, one_log_theta))
        temp2 = (1 - s_N) * (np.dot(one_y, log_theta) + np.dot(y, one_log_theta))
        log_pi_N = temp1 + temp2

        # log_sum_exp trick to avoid underflow
        pi_N = self.log_sum_exp(log_pi_N)
        return pi_N

    def compute_s_N(self, y, theta, pi_s, pi_N):
        log_theta = np.log(theta)
        one_log_theta = np.log(1 - theta)

        # sum term1
        temp1 = np.dot(np.dot(y, log_theta), pi_N)

        # sum term 2
        temp2 = np.dot(np.dot(1 - y, one_log_theta), pi_N)

        # sum term 3
        temp3 = np.dot(np.dot(1 - y, log_theta), pi_N)

        # sum term 4
        temp4 = np.dot(np.dot(y, one_log_theta), pi_N)

        # pi_s + sum over M row and R columns
        sum_log_s_N_num_1 = np.log(pi_s) + temp1 + temp2

        # 1-pi_s + sum over M row and R columns
        sum_log_s_N_num_2 = np.log(1 - pi_s) + temp3 + temp4

        # log-sum-exp trick
        temp = [sum_log_s_N_num_1, sum_log_s_N_num_2]
        s_N_all = self.log_sum_exp(temp)
        s_N = s_N_all[0]
        return s_N, sum_log_s_N_num_1, sum_log_s_N_num_2
