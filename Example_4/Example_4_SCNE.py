#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
S. Lorenzini, D. Petturiti, B. Vantaggi.
Stackelberg-Cournot-Nash equilibria under ambiguity and α-maxmin preferences. 2026
"""

import numpy as np
import pandas as pd

N = 20

c_min = np.vstack((2 * np.ones((N, N)) - np.diag(np.ones(N)), 1 * np.ones(N)))

print('c_min:\n', c_min)
print()


c_max = np.vstack((2 * np.ones((N, N)) - np.diag(np.ones(N)), 2 * np.ones(N)))
print('c_max:\n', c_max)
print()



# X-marginal Mobius inverse
delta = 0.6


# Values of alpha to test
step = 0.1
alphas = [0.5]#np.arange(0, 1 + step, step)

ks = []

# Sinkhorn iterations for every alpha
for alpha in alphas:
    print()
    print('COMPUTING alpha:', alpha)
    c = alpha * c_min + (1 - alpha) * c_max
    
    print('c:', c)
    
    # X-marginal Mobius inverse
    w1 = np.ones(N)
    m_mu_orig = np.append(((1 - delta) * (w1 / w1.sum())), delta)
    m_mu_orig = m_mu_orig.reshape((len(m_mu_orig), 1))


    # Y-marginal Mobius inverse
    m_nu_orig = np.array([2*j / (N*(N+1)) for j in range(1, N+1)])
    m_nu_orig = m_nu_orig.reshape((len(m_nu_orig), 1))
    
    (m, n) = c.shape

    epsilon = 0.003
    K = np.exp(- c / epsilon)    
    
    g = np.ones((N, 1))
    f = np.ones((N+1, 1))
    
    m_gamma = np.ones_like(K)
    
    for i in range(1000):
        print('iter:', i)
        m_gamma_old = m_gamma
        u = np.exp(f / epsilon)
        v = np.exp(g / epsilon)
        m_gamma = np.diag(u.squeeze()) @ K @ np.diag(v.squeeze())
        f = epsilon * np.log(m_mu_orig) - epsilon * np.log(K @ np.exp(g / epsilon))
        g = epsilon * np.log(m_nu_orig) - epsilon * np.log(K.T @ np.exp(f / epsilon))
        # Extract the marginals from the joint
        m_mu = np.sum(m_gamma, axis=1)
        m_nu = np.sum(m_gamma, axis=0)
        print('m_mu:', m_mu)
        print('m_mu_orig:', m_mu_orig.squeeze())
        print('m_nu:', m_nu)
        print('m_nu_orig:', m_nu_orig.squeeze())
        if np.maximum(np.sum(np.abs(m_mu - m_mu_orig.squeeze())), np.sum(np.abs(m_nu - m_nu_orig.squeeze()))) < 10**(-8):
            break
        
    
    # Gradient vector
    grad = m_nu_orig**2 + 2 * m_nu_orig 
    
    
    # Print results
    print('\n\nRESULT:')
    m_mu = np.sum(m_gamma, axis=1)
    m_nu = np.sum(m_gamma, axis=0)
    print('m_gamma:\n', np.round(m_gamma, 4), 'with sum =', round(sum(sum(m_gamma)), 4), '\n')
    print('m_mu:', np.round(m_mu, 4), 'with sum = ', round(np.sum(m_gamma), 4))
    print('m_nu:', np.round(m_nu, 4), 'with sum = ', round(np.sum(m_gamma), 4))
    print()
    print('c:', np.round(c, 4))
    print('f:', np.round(f, 4).squeeze())
    print('g:', np.round(g, 4).squeeze())
    print('- g - grad:', np.round(- g - grad,4).squeeze())
    print()
    numeraire = - (- g - grad).min() + 1
    k = - g - grad + numeraire
    print('k:', np.round(k, 6).squeeze())
    print()
    print('k:', np.round(k, 4).squeeze())
    
    
    gc = ((c[:, :] - g.squeeze()).min())
    print('gc:', gc)
  
    ks.append(k.squeeze())


print()
print('OPTIMAL PRICE FUNCTIONS:')
i = 0
for k in ks:
    print('alpha = ', np.round(alphas[i], 1), 'k:',  np.round(k, 4))
    i += 1

# Save the optimal price function in a CSV
df_ks = pd.DataFrame(ks)
df_ks.to_csv('optimal_price_functions.csv', index=False)
