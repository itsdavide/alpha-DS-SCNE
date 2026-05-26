#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
S. Lorenzini, D. Petturiti, B. Vantaggi.
Stackelberg-Cournot-Nash equilibria under ambiguity and α-maxmin preferences. 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from Example_4_prox_E import prox_E
from Example_4_prox_k import prox_k


n = 20


c_min = np.vstack((2 * np.ones((n, n)) - np.diag(np.ones(n)), 1 * np.ones(n)))

print('c_min:\n', c_min)
print()


c_max = np.vstack((2 * np.ones((n, n)) - np.diag(np.ones(n)), 2 * np.ones(n)))
print('c_max:\n', c_max)
print()



# X-marginal Mobius inverse
delta = 0.6


w1 = np.ones(n)
m_mu_orig = np.append(((1 - delta) * (w1 / w1.sum())), delta)
print(m_mu_orig)


# Index set for the Y-marginal variables
I = [i for i in range(0, n)]

# Price functions to test
ks = np.array([
    np.minimum(np.concatenate((np.arange(n/2,0,-1), np.arange(1,n/2+1))), 3)
    ])

# Entropic regularization parameter
epsilon = 0.01

def prox_D1(m_eta):
    m_gamma = np.zeros_like(m_eta)
    (m, n) = m_eta.shape
    for i in range(m):
        for j in range(n):
            m_gamma[i, j] = m_mu_orig[i] * m_eta[i, j] / sum(m_eta[i, :])
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
alphas = [0.5]


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
            m_mu_0 = np.sum(m_gamma, axis=1)
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
            print('n = ', n)
            print(np.round(m_gamma, 3))
            print()
            print(np.sum(m_gamma, axis=1))
            m_mu_3 = np.sum(m_gamma, axis=1)
            if (np.sum(np.abs(m_mu_3 - m_mu_orig)) < 10**(-8)):
                break
                
            
        m_mu_rec = np.sum(m_gamma, axis=1)
        m_nu_rec = np.sum(m_gamma, axis=0)
        ent = -entropy(m_nu_rec)
        print('m_gamma:\n', np.round(m_gamma, 4), 'with sum =', round(sum(sum(m_gamma)), 4), '\n')
        print('m_mu:', np.round(m_mu_rec, 4), 'with sum = ', round(np.sum(m_gamma), 4))
        print('m_nu:', np.round(m_nu_rec, 4), 'with sum = ', round(np.sum(m_gamma), 4))
        print('(Negative) Entropy nu:', ent)
        
    print('m_nu:\n', m_nu_rec)
    print('k:\n', k)