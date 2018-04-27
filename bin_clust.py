import numpy as np
import pandas as pd
import pdb


class binclust(object):
    """Clustering binary data with two coding scheme"""

    def __init__(self, X, R=2):
        # X is a binary data frame
        if isinstance(X, pd.DataFrame):
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
        pi_r /= sum(pi_r)
        # initial coding scheme weights
        pi_s = np.random.rand(1)
        # initial posterior mean of weights
        pi_N = np.tile(pi_r, (self.N, 1)).T
        # initial posterior mean of coding scheme weights
        s_N = pi_s * np.ones(self.N)
        return theta, pi_r, pi_s, pi_N, s_N

    def E_step(self, *arg):
        theta, pi_r, pi_s, pi_N, s_N = arg
        # Pre-allocation for speed
        pi_N_new = np.zeros((self.R, self.N))
        s_N_new = np.zeros((self.N))
        # Variational E-step
        i = 0
        loglik = 0
        M_term1 = np.zeros(self.R)
        M_term2 = 0
        M_term3 = np.zeros((self.M, self.R))
        for y in self.X:
            # The variational E_step
            arg_pi_n = [y, theta, pi_r, s_N[i]]
            pi_N_new[:, i] = self.compute_pi_n(*arg_pi_n)
            arg_s_n = [y, theta, pi_s, pi_N[:, i]]
            temp = self.compute_s_n(*arg_s_n)

            s_N_new[i] = temp[0]
            # Compute log-likelihood
            loglik += s_N_new[i] * (temp[1] + np.log(pi_s))
            loglik += 1 - s_N_new[i] * (temp[2] + np.log(1 - pi_s))
            loglik += sum(np.multiply(pi_N_new[:, i], np.log(pi_r)))
            # Prepare for summation terms for M-step
            M_term1 += pi_N_new[:, i]
            M_term2 += s_N_new[i]
            temp1 = np.tile(1 - s_N_new[i] * (1 - y), (self.R, 1)).T
            temp1 = np.multiply(temp1, np.tile(pi_N_new[:, i], (self.M, 1)))
            M_term3 += temp1
            temp2 = np.tile(s_N_new[i] * y, (self.R, 1)).T
            temp2 = np.multiply(temp2, np.tile(pi_N_new[:, i], (self.M, 1)))
            pdb.set_trace()
            M_term3 += temp2
            # M by R M_term3
            i += 1
        return pi_N_new, s_N_new, M_term1, M_term2, M_term3, loglik

    def M_step(self, *arg):
        pi_N_new, s_N_new, M_term1, M_term2, M_term3 = arg
        # Pre-allocate for speed
        pi_r_new = np.zeros(self.R)
        pi_s_new = 0
        # M_step
        pi_r_new = M_term1 / sum(M_term1)
        pi_s_new = M_term2 / self.N
        theta_new = np.divide(M_term3, np.tile(M_term1, (self.M, 1)))
        pdb.set_trace()
        return theta_new, pi_r_new, pi_s_new, pi_N_new, s_N_new

    @staticmethod
    def log_sum_exp(x):
        # chopped off precision
        k = -10
        e = x - np.max(x)
        y = np.exp(e) / sum(np.exp(e))
        y[e < k] = 0
        return y

    def compute_pi_n(self, *arg):
        y, theta, pi_r, s_n = arg
        # Y_MR size M by R to match theta
        Y_MR = np.tile(y, (self.R, 1)).T
        # 1- Y_MR
        one_Y_MR = 1 - Y_MR
        # log(theta) in M by R
        log_theta = np.log(theta)
        one_log_theta = np.log(1 - theta)
        # All M by R terms in equation 12
        log_pi_n_num = s_n * (np.multiply(Y_MR, log_theta) + np.multiply(one_Y_MR, one_log_theta)) + 1 - s_n * (np.multiply(one_Y_MR, log_theta) + np.multiply(Y_MR, one_log_theta))
        # Sum over M rows
        sum_log_pi_n_num = np.log(pi_r) + np.sum(log_pi_n_num, axis=0)
        # log_sum_exp trick to prevent underflow
        pi_n = self.log_sum_exp(sum_log_pi_n_num)
        # Re-normalize to ensure summing up to 1
        pi_n /= sum(pi_n)
        return pi_n

    def compute_s_n(self, *arg):
        y, theta, pi_s, pi_n = arg
        # For computing Y_MR
        temp1 = np.tile(y, (self.R, 1)).T
        temp2 = np.tile(pi_n, (self.M, 1))
        Y_MR = np.multiply(temp1, temp2)
        one_Y_MR = np.multiply(1 - temp1, temp2)
        log_theta = np.log(theta)
        one_log_theta = np.log(1 - theta)
        # pi_s + sum over M row and R columns
        log_s_n_num_1 = np.multiply(Y_MR, log_theta) + np.multiply(one_Y_MR, one_log_theta)
        sum_log_s_n_num_1 = np.log(pi_s) + np.sum(log_s_n_num_1)
        # 1-pi_s + sum over M row and R columns
        log_s_n_num_2 = np.multiply(one_Y_MR, log_theta) + np.multiply(Y_MR, one_log_theta)
        sum_log_s_n_num_2 = np.log(1 - pi_s) + np.sum(log_s_n_num_2)
        temp = [sum_log_s_n_num_1, sum_log_s_n_num_2]
        s_n_all = self.log_sum_exp(temp)
        s_n = s_n_all[0] / sum(s_n_all)
        return s_n, sum_log_s_n_num_1, sum_log_s_n_num_2
