import os
import sys
import numpy as np
import random
import argparse
import json
from time import time
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Run symbolic anomaly detection")
parser.add_argument("--data", type=str, required=True, help="Path to dataset (.npz)")
parser.add_argument("--complexity_weight", type=float, default=0.4,
                    help="Weight for complexity penalty (default=0.4)")
parser.add_argument("--num_chunks", type=int, default=None,
                    help="Number of chunks (default: 10 if n<=3 else 20)")
parser.add_argument("--chunk_size", type=int, default=None,
                    help="Chunk size (default: min(3, n))")
parser.add_argument("--loss_bound", type=float, default=1.0,
                    help="Loss2 boundary term (default=1.0)")
args = parser.parse_args()

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
ds=args.data
dspth="data/"+ds+".npz"
data = np.load(dspth)
train_data = data['x']
test_data = data['tx']
test_labels = data['ty']

loss_bound=float(args.loss_bound)

# Automatically detect number of features
n = train_data.shape[1]
print(f"Detected number of features: {n}")

# Generate the variables dynamically
variables = [mvar(chr(97 + i)) for i in range(n)]

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
            loss2 = np.mean(np.abs(func(**random_values_dict) - 1))
            #loss = loss1 / (np.abs(loss2 + loss1) + 1e-10)
            loss= loss1 + np.maximum(0, loss_bound-loss2)
            # Use complexity weight from args
            loss += args.complexity_weight * np.log(1 + np.log(1 + func.complexity()))
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

if args.chunk_size is not None:
    chunk_size = min(args.chunk_size,int(n))
else:
    chunk_size = min(3, n)

complx=args.complexity_weight


pth=f"results/{ds}/{chunk_size}/{loss_bound}/{complx}/"
os.makedirs(pth, exist_ok=True)


# Results container

for i, chunk in enumerate(tqdm(generate_random_chunks(variables, chunk_size))):
    try:
        print("working on chunk",i)
        try:
            print(f"\nIteration {i + 1}: Using variables {', '.join([var.name for var in chunk])}")
            values_dict = {var.name: train_data[:, variables.index(var)] for var in chunk}
            test_values_dict = {var.name: test_data[:, variables.index(var)] for var in chunk}
            random_values_dict = {
                var.name: np.random.uniform(np.min(values_dict[var.name]), np.max(values_dict[var.name]), len(values_dict[var.name]))
                for var in chunk
            }
    
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
        print(f"Symbolic Expression for Chunk {i + 1}: {sol}")
    
        dic={}
        def itera(obj):
            if type(obj) is mguess:
                dic[obj.name]=obj.value
        sol.iterate_function(itera)
        print(dic)
    
        # Training scores
        scores_training = np.abs(sol(**values_dict) - 1)
        mean_scores_training = np.mean(scores_training)
        alpha = solve_alpha(mean_scores_training)
    
        # Testing scores
        scores = np.abs(np.array([
            sol(**{var.name: test_data[idx, variables.index(var)] for var in chunk})
            for idx in range(test_data.shape[0])
        ]) - 1)
        scores = sigmoid(alpha * scores)
    
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            print(f"Skipping Chunk {i + 1} due to NaN or inf values in scores.")
            continue
    
        roc_auc_chunk = roc_auc_score(test_labels, scores)
    
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
            "loss_bound":float(loss_bound)
        }
        results.update(dic)
    
        np.savez_compressed(f"{pth}/{indice}.npz",
                            train_scores=scores_training,
                            test_scores=scores,
                            test_labels=test_labels,**results)
    except Exception as e:
        print(f"An error occurred in Chunk {i + 1}: {e}")
        continue
    
