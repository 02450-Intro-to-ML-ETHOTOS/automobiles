import numpy as np
import scipy.stats as st

def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in st.beta.interval(1-alpha, a=p, b=q))

    p = 2*st.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print(f"Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = {CI}")
    print(f"p-value for two-sided test A and B have same accuracy (exact binomial test): p={p}")

    return thetahat, CI, p

# test
# y = np.array([0, 1, 3, 1, 2])
# yA = np.array([2, 1, 0, 1, 2])
# yB = np.array([0, 1, 3, 1, 0])
# mcnemar(y, yA, yB, 0.05)
