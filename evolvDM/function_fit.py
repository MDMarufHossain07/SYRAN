import numpy as np
from plt import *

from morga import munc,objects

from morga import mconst,mvar,madd,msub,mmul

import mumpy as mp

from library import library

from phase_search import phase_search

import json

ident="exp"
import sys
if len(sys.argv)>1:
    ident=sys.argv[1]

np.random.seed(12)

x=mvar("x")
alpha=mvar("alpha")


#solution=mp.exp(mp.abs(alpha)*x)
solution=mp.exp(mp.abs(alpha)*x)
if ident=="pow":
    solution=mp.pow(x,3)+alpha
if ident=="sin":
    solution=mp.sin(alpha*x)
if ident=="cosh":
    solution=mp.exp(x)+mp.exp(-1*x)
if ident=="hard":
    solution=x/(alpha+x**2)

xlim=[-2,2]
if ident=="exp":
    xlim=[0,1]
alpha_true=3

x=np.linspace(xlim[0],xlim[1],100)
y=solution(x=x,alpha=alpha_true)
y+=np.random.normal(0,0.1,len(y))

def solve(func):
    return func.solve(y,x=x)

def eval(func):
    try:
        #func=solve(func)
        loss=np.mean(np.square(func(x=x)-y))
        loss+=0.01*np.log(1+np.log(1+func.complexity()))
        return loss
    except:
        return 1.e10

def sigmoid(x):
    return 1/(1+np.exp(-x))

def update(o1,o2):
    if np.random.random()<0.2*sigmoid(np.log(o1.complexity())):
        return solve(init())
    if np.random.random()<0.5:
        o1,o2=o2,o1
    return solve(o1.offspring(o2))

def init():
    func=munc().random_function()
    return func



sol,error,hist,mats=phase_search(eval,update,init)

#sol=solve(sol)

print()
print("perfect func:",eval(solve(solution.copy())))
print()
print(error)
print()
print(sol)
print()
print(sol.extract_guesses())
print()
print(sol.tolatex())
print()
print(sol.complexity())

plt.figure(figsize=(8,8))
inp=np.linspace(xlim[0],xlim[1],100)
out=sol(x=inp)
plt.plot(inp,out,label="fit")
plt.plot(x,y,"o",label="data")
plt.legend()
plt.savefig(f"imgs/{ident}.png",format="png",dpi=300)
plt.savefig(f"imgs/{ident}.pdf",format="pdf")
plt.show()

def encode_hist(hist=hist):
    return {"hist_func":[a for a,b in hist.items()],"hist_qual":[b for a,b in hist.items()]}

data={"ident":ident,"x":x,"y":y,"alpha":alpha_true,"true":repr(solution),"bestfit":repr(sol),"error":error,"guesses":json.dumps(sol.extract_guesses()),**encode_hist(hist),"mats":mats,"xlim":xlim}

np.savez_compressed(f"results/{ident}.npz",**data)


exit()

best_sol=None
best_error=None
iteration=0
while True:
    try:
        iteration+=1
        if not iteration % 100:print(iteration,best_error)

        func=munc().random_function()
        func=func.solve(y,x=x)
        loss=np.mean(np.square(func(x=x)-y))
        if best_error is None or loss<best_error:
            best_error=loss
            best_sol=func
            print("Iteration",iteration)
            print("Error",best_error)
            print("Solution",best_sol)
            print("Values",best_sol.extract_guesses())
    except Exception as e:
        continue
    
exit()

q=q.solve(y,x=x)
print(repr(q))

fit=q(x=x)


plt.plot(x,y,"o")
plt.plot(x,fit)
plt.show()

print("output of gradient descent:",q.extract_guesses())


