import os
import sys
import numpy as np
import random
import argparse
import json
from time import time
from tqdm import tqdm



#random_seed = 42
#os.environ["PYTHONHASHSEED"] = str(random_seed)
#np.random.seed(random_seed)
#random.seed(random_seed)

from morga import munc, mvar, mconst, objects, mguess
from mconst import mconst
from library import library
from phase_search import phase_search
#from basic_search import phase_search
import mumpy as mp
from sklearn.metrics import roc_auc_score


# -------------------------------
# Load Data
# -------------------------------
data=np.load("exoplanet_data.npz")
data=data["data"]

# Automatically detect number of features
n = data.shape[1]
print(f"Detected number of features: {n}")

# Generate the variables dynamically
#variables = [mvar(chr(97 + i)) for i in range(n)]
variables=["T","a"]
variables=[mvar(v) for v in variables]
chunk_size=2
complexity_weight=0.03

true_solution=variables[0]**2/variables[1]**3


# -------------------------------
# Helper Functions
# -------------------------------
def generate_random_chunks(vars, chunk_size):
    while True:
        chunk = random.sample(vars, chunk_size)
        yield chunk

def solve(func, values_dict):
    return func.solve(1, **values_dict)

def isconst(func, variables):
    if isinstance(func, mconst) or isinstance(func, mguess):
        return True
    def contains_variable(f):
        if isinstance(f, mvar):
            return f.name in [var.name for var in variables]
        if hasattr(f, 'children'):
            return any(contains_variable(child) for child in f.children())
        return False
    return not contains_variable(func)

def eval(func, values_dict, random_values_dict, variables):
    func = func.simplify()
    if isconst(func, variables):
        return 1.e10
    else:
        try:
            loss1 = np.mean(np.abs(func(**values_dict) - 1))
            loss2 = np.mean(-np.abs(func(**random_values_dict) - 1))
            loss2r = np.mean(np.abs(func(**random_values_dict) - 1))
            loss = loss1 / (np.abs(loss2 + loss1) + 1e-10)
            loss = loss1 + max(0, 2-loss2r)
            #loss = (loss1-loss2) / (np.abs(loss2 + loss1) + 1e-10)
            #loss= max(0,1+loss1-loss2)
            # Use complexity weight from args
            loss += complexity_weight * np.log(1 + np.log(1 + func.complexity()))
            return loss
        except Exception:
            return 1.e10

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def update(o1, o2, variables, values_dict):
    if np.random.random() < 0.2 * sigmoid(np.log(o1.complexity())):
        return solve(init(), values_dict)
    if np.random.random() < 0.5:
        o1, o2 = o2, o1
    try:
        return solve(o1.offspring(o2), values_dict)
    except ValueError:
        return mconst(1.e10)

def init():
    return munc().random_function()

def solve_alpha(mean_scores_training):
    if mean_scores_training == 0:
        print("Warning: mean_scores_training is 0, setting alpha to a default value.")
        return 1e-6
    return 1 / mean_scores_training

# -------------------------------
# Main Experiment Loop
# -------------------------------

complx=complexity_weight


#pth=f"results/{ds}/{chunk_size}/{complx}/"
pth="kepler/"
#os.makedirs(pth, exist_ok=True)


# Results container

for i, chunk in enumerate(tqdm(generate_random_chunks(variables, chunk_size))):
    print("working on chunk",i)
    try:
        print(f"\nIteration {i + 1}: Using variables {', '.join([var.name for var in chunk])}")
        values_dict = {var.name: data[:, variables.index(var)] for var in chunk}
        test_values_dict = {var.name: data[:, variables.index(var)] for var in chunk}
        random_values_dict = {
            var.name: np.random.uniform(np.min(values_dict[var.name]), np.max(values_dict[var.name]), len(values_dict[var.name]))
            for var in chunk
        }
        #print(values_dict) 
        #print(random_values_dict)
        #exit()

        sol, error, hist, mats = phase_search(
            lambda f: eval(f, values_dict, random_values_dict, chunk),
            lambda o1, o2: update(o1, o2, chunk, values_dict),
            init, n=30000
        )
    except ValueError as e:
        print(f"Skipping Chunk {i + 1} due to an error: {e}")
        continue

    # Finalize solution
    sol = solve(sol, values_dict)

    dic={}
    def itera(obj):
        if type(obj) is mguess:
            dic[obj.name]=obj.value
    sol.iterate_function(itera)
    print(dic)
            


    #exit()
    print(f"Symbolic Expression for Chunk {i + 1}: {sol}")

    # Training scores
    scores_training = np.abs(sol(**values_dict) - 1)
    mean_scores_training = np.mean(scores_training)
    alpha = solve_alpha(mean_scores_training)

    # Testing scores
    scores = np.abs(np.array([
        sol(**{var.name: data[idx, variables.index(var)] for var in chunk})
        for idx in range(data.shape[0])
    ]) - 1)
    scores = sigmoid(alpha * scores)

    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        print(f"Skipping Chunk {i + 1} due to NaN or inf values in scores.")
        continue

    roc_auc_chunk = -1#roc_auc_score(test_labels, scores)

    indice=str(time())

    # Save results for this chunk
    results={
        "indice":indice,
        "variables": [var.name for var in chunk],
        "symbolic_equation": str(sol),
        "roc_auc_chunk": float(roc_auc_chunk),
        "error":float(error),
        "chunk_size":float(chunk_size),
        "complexity_weight":float(complx),
        "right":float(eval(true_solution, values_dict, random_values_dict, chunk))
    }
    results.update(dic)
    print(json.dumps(results,indent=2))

    np.savez_compressed(f"{pth}/{indice}.npz",
                        train_scores=scores_training,**results)

