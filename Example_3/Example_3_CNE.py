#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
S. Lorenzini, D. Petturiti, B. Vantaggi.
Stackelberg-Cournot-Nash equilibria with Dempster-Shafer uncertainty 
and α-maxmin preferences. 2005 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from Example_3_prox_E import prox_E
from Example_3_prox_k import prox_k

c_min = np.array([
    [2, 2, 1],
    [1, 2, 2],
    ])

c_max = np.array([
    [3, 2, 2],
    [2, 2, 3]
    ])

# X-marginal Mobius inverse
w1 = np.array([6, 4])
m_mu = w1 / w1.sum()

# Index set for the Y-marginal variables
I = [0, 1, 2]

# Price functions to test
ks = np.array([
    [3, 2, 1],
    [1, 2, 3]])

# Entropic regularization parameter
epsilon = 0.01

def prox_D1(m_eta):
    m_gamma = np.zeros_like(m_eta)
    (m, n) = m_eta.shape
    for i in range(m):
        for j in range(n):
            m_gamma[i, j] = m_mu[i] * m_eta[i, j] / sum(m_eta[i, :])
    return m_gamma

def prox_D2(m_eta):
    m_gamma = np.zeros_like(m_eta)
    (m, n) = m_eta.shape
    old_m_nu = np.zeros(n)
    for j in range(n):
        old_m_nu[j] =  sum(m_eta[:, j])
    (m_nu, opt_val) = prox_k(old_m_nu, I, k, epsilon)
    for i in range(m):
        for j in range(n):
            m_gamma[i, j] = m_nu[j] * m_eta[i, j] / sum(m_eta[:, j])
    return m_gamma

def prox_D3(m_eta):
    m_gamma = np.zeros_like(m_eta)
    (m, n) = m_eta.shape
    old_m_nu = np.zeros(n)
    for j in range(n):
        old_m_nu[j] =  sum(m_eta[:, j])
    (m_nu, opt_val) = prox_E(old_m_nu, I, epsilon)
    for i in range(m):
        for j in range(n):
            m_gamma[i, j] = m_nu[j] * m_eta[i, j] / sum(m_eta[:, j])
    return m_gamma

# Values of alpha to test
step = 0.1
alphas = np.arange(0, 1 + step, step)


i = 0
for k in ks:
    dists = []
    ents = []
    i += 1
    # Dykstra iterations for every alpha
    for alpha in alphas:
        print()
        print('*** COMPUTING alpha:', np.round(alpha, 1), '***')
        print('#k:', i)
        
        c = alpha * c_min + (1 - alpha) * c_max
        print('c:', c)
        
        # Initial matrix of Dykstra's algorithm
        m_gamma0 = np.exp(- c / epsilon)
            
        z1 = np.ones(c.shape)
        z2 = np.ones(c.shape)
        z3 = np.ones(c.shape)
        
        m_gamma = m_gamma0
        for n in range(1000):
            m_gamma_0 = m_gamma
            m_gamma_1 = prox_D1(m_gamma_0 * z1)
            z1 = z1 * (m_gamma_0 / m_gamma_1)
            #
            m_gamma_2 = prox_D2(m_gamma_1 * z2)
            z2 = z2 * (m_gamma_1 / m_gamma_2)
            #
            m_gamma_3 = prox_D3(m_gamma_2 * z3)
            z3 = z3 * (m_gamma_2 / m_gamma_3)
            m_gamma = m_gamma_3
            if (np.sum(np.abs(m_gamma_3 - m_gamma_0)) < 10**(-8)):
                break
                
            
        m_mu = np.sum(m_gamma, axis=1)
        m_nu = np.sum(m_gamma, axis=0)
        ent = -entropy(m_nu)
        print('m_gamma:\n', np.round(m_gamma, 4), 'with sum =', round(sum(sum(m_gamma)), 4), '\n')
        print('m_mu:', np.round(m_mu, 4), 'with sum = ', round(np.sum(m_gamma), 4))
        print('m_nu:', np.round(m_nu, 4), 'with sum = ', round(np.sum(m_gamma), 4))
        print('(Negative) Entropy nu:', ent)
        
        dists.append(m_nu)
        ents.append(ent)
        
    df_dists = pd.DataFrame(dists)
    df_dists.to_csv('dists_' + str(i + 1) + '.csv')
    
    df_ents = pd.DataFrame(ents)
    df_ents.to_csv('ents_' + str(i + 1) + '.csv')
    
    plt.figure(figsize=(6, 4))
    
    plt.title(r'(Negative) Entropy of optimal distribution $\nu^k$')
    plt.plot(alphas, ents, color = 'red')
    
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$H(\nu^k)$')
    plt.savefig('entropy_' + str(i + 1) +'.png', dpi=300)