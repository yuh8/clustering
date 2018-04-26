import numpy as np
import pandas as pd


class binclust(object):
    """Clustering binary data with two coding scheme"""

    def __init__(self, X, R=2):
        # X is a binary data frame
        if isinstance(X, pd.Dataframe):
            self.X = X.as_matrix()
        else:
            self.X = np.array(X)
        # Specifying number of clusters Default is 3
        self.R = R

    @property
    def para_init(self):
        self.N = self.X.shape[0]
        self.M = self.X.shape[1]
        # M by R array of initial thetas
        theta = np.random.rand(self.M, self.R)
        # initial cluster weights
        pi_r = np.random.rand(self.R)
        # initial coding scheme weights
        pi_s = np.random.rand(2)
        # initial posterior mean of weights
        pi_N = np.tile(pi_r, (self.N, 1)).T
        # initial posterior mean of coding scheme weights
        s_N = np.tile(pi_s, (self.N, 1)).T
        return theta, pi_r, pi_s, pi_N, s_N

    def E_step(self, *arg):
        theta, pi_r, pi_s, pi_N, s_N = arg[0:]
        # Pre-allocation for speed
        pi_N_new = np.zeros((self.R, self.N))
        s_N_new = np.zeros((2, self.N))
        # Variational E-step
        i = 0
        for y in self.X:
            pi_N_new[:, i] = self.compute_pi_n(y, theta, pi_r, s_N[:, i])
            s_N_new[:, i] = self.compute_s_n(y, theta, pi_s, pi_N[:, i])
            i += 1
        return pi_N_new, s_N_new

    @staticmethod
    def log_sum_exp(x):
        # chopped off precision
        k = -1e-6
        e = x - np.max(x)
        y = np.exp(e) / sum(np.exp(e))
        y[e < k] = 0
        return y

    def compute_pi_n(self, *arg):
        y, theta, pi_r, s_n = arg[0:]
        # outer product of Y[n,:] I_R, size M by R
        Y_MR = np.tile(y, (self.R, 1)).T
        # 1- Y_MR
        one_Y_MR = 1 - Y_MR
        # log(theta) in M by R
        log_theta = np.log(theta)
        one_log_theta = np.log(1 - theta)
        # All M by R terms in equation 12
        log_pi_n_num = s_n * (np.multiply(Y_MR, log_theta) + np.multiply(one_Y_MR, one_log_theta)) + (1 - s_n) * (np.multiply(one_Y_MR, log_theta) + np.multiply(Y_MR, one_log_theta))
        # Sum over M rows
        sum_log_pi_n_num = np.log(pi_r) + np.sum(log_pi_n_num, axis=0)
        # log_sum_exp trick to prevent underflow
        pi_n = self.log_sum_exp(sum_log_pi_n_num)
        pi_n /= sum(pi_n)
        return pi_n

    def compute_s_n(self, *arg):
        y, theta, pi_s, pi_n = arg[0:]
        # outer product of Y[n,:] I_R and elementwise product with outer product of I_M and z_Rn
        temp1 = np.tile(y, (self.R, 1)).T
        temp2 = np.tile(pi_n, (self.M, 1))
        #
        Y_MR = np.multiply(temp1, temp2)
        one_Y_MR = np.multiply(1 - temp1, temp2)
        log_theta = np.log(theta)
        one_log_theta = np.log(1 - theta)
        # pi_s[0] + sum over M row and R columns
        log_s_n_num_1 = np.multiply(Y_MR, log_theta) + np.multiply(one_Y_MR, one_log_theta)
        sum_log_s_n_num_1 = np.log(pi_s[0]) + np.sum(log_s_n_num_1)
        # pi_s[1] + sum over M row and R columns
        log_s_n_num_2 = np.multiply(one_Y_MR, log_theta) + np.multiply(Y_MR, one_log_theta)
        sum_log_s_n_num_2 = np.log(pi_s[1]) + np.sum(log_s_n_num_2)
        temp = [sum_log_s_n_num_1, sum_log_s_n_num_2]
        pi_s = self.log_sum_exp(temp)
        pi_s /= sum(pi_s)
        return pi_s
