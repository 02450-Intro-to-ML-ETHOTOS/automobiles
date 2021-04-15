import numpy as np
import scipy.stats as st

def t_test_paired(zA, zB, alpha = 0.05):
    """Paired T-test

    Assumes equal sample sizes and variance

    Paired, two-tailed t-test, i.e. the hypotheses are:
    * H0: model A and B have the same perforamnce, Z = 0
    * H1: model A and B have different performance, Z != 0

    As such, the p-value indicates that the means are different in either "direction".
    This means that the comparison B vs. RR is equivalent to RR vs. B.

    Verbose implementation, clarity > fancy one-liners.
    See ex7_2_1.py for inspiration.

    TODO:
    * report - consider these aspects
    * is zA supposed to be the mean E_test error? Or the error for each observation?
    In the book box 11.3.4 they write that they recommend n >= 30 samples
    """

    z = zA - zB
    n = len(z)
    z_hat = np.sum(z) / n # z_hat is the mean difference
    # sigma^2 is the variance
    sigma_tilde_sq = np.sum((z - z_hat)**2) / n*(n-1) # TODO: why this form of the variance?
    # sqrt(sigma^2) is the std.dev., needed for computing the Std.err. of the mean, SEM
    sigma_hat = np.sqrt(sigma_tilde_sq) # std.dev.
    sem = sigma_hat / np.sqrt(n)
    
    # p for two-tailed T-test is: P(T > |t|) = 2 * (1 - cdf(t))
    p = 2 * (1 - st.t.cdf(np.abs( z_hat / sem ), df=n-1)) # TODO: ask TA about the df?
    
    # in scipy.stats, loc is the mean and scale is the std.dev.
    CI = st.t.interval(1-alpha, df=n-1, loc=z_hat, scale=sigma_hat)  # Confidence interval

    return p, CI

# p, ci = t_test_paired(np.random.normal(0, 1, 10), np.random.normal(10, 1, 10)) # test
# print(f"p = {p}, with CI: {ci}")