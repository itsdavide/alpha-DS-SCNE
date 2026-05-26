#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
S. Lorenzini, D. Petturiti, B. Vantaggi.
Stackelberg-Cournot-Nash equilibria under ambiguity and α-maxmin preferences. 2026
"""

import pyomo.environ as pyo
from pyomo.environ import log
import numpy as np

optimizer_path = '/Users/davidepetturiti/bonmin'

tolerance = 0.00001

def prox_k(q, I, k, epsilon): 
    tot = sum(q)
    q = q / tot
    # Create a model in  pyomo
    model = pyo.ConcreteModel()
    
    # Define the index set in pyomo
    model.I = pyo.Set(initialize=I)
    
    # Define the constants in pyomo
    model.q = pyo.Param(model.I, initialize=q)
    model.k = pyo.Param(model.I, initialize=k)
    
    # Define the variables in pyomo
    model.p = pyo.Var(model.I, within=pyo.NonNegativeReals, bounds=(tolerance, np.inf), initialize=tolerance)
    model.KL = pyo.Var(bounds=(-np.inf, np.inf))
    model.E = pyo.Var(bounds=(-np.inf, np.inf))
    
    # Set the constraints
    model.c1 = pyo.Constraint(expr=sum(model.p[i] * log(model.p[i] / model.q[i]) - model.p[i] + model.q[i] for i in model.I) == model.KL)
    model.c2 = pyo.Constraint(expr=sum(model.p[i] * model.k[i] for i in model.I) == model.E)
    model.c3 = pyo.Constraint(expr=sum(model.p[i] for i in model.I) == 1)
    
    model.o = pyo.Objective(expr = model.KL + (1 / epsilon) * model.E, sense=pyo.minimize)
    
    status = pyo.SolverFactory(optimizer_path).solve(model)
    pyo.assert_optimal_termination(status)

    
    opt_p = []
    for i in model.I:
        opt_p.append(pyo.value(model.p[i]))
    
    return (np.array(opt_p),  pyo.value(model.o))