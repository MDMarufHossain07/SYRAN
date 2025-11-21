import numpy as np
from plt import *

from morga import munc,objects

from morga import mconst,mvar,madd,msub,mmul

import mumpy as mp

from library import library

from phase_search import phase_search

import json

ident="gauss"
import sys
if len(sys.argv)>1:
    ident=sys.argv[1]

np.random.seed(12)

x=mvar("x")

if ident[:5]=="gauss":
    print("doing gauss")
    data=np.random.normal(0,1,1000)
elif ident=="uniform":
    data=np.random.uniform(-1,1,1000)
elif ident=="multi":
    data1=np.random.normal(-1,0.5,500)
    data2=np.random.normal(1,0.5,500)
    data=np.concatenate((data1,data2))
    np.random.shuffle(data)

solution=mp.exp(-1*mp.square(x)/2)


xlim=[-5,5]

X=np.linspace(xlim[0],xlim[1],10000)

zero=np.zeros_like(data)

def solve(func):
    return func.solve(zero,x=data)

def eval(func):
    try:
        #there are two variables. x for the input we care about, and "d" for the density input.
        func.rename_variables({"d":"x"})#dont allow the network to pick up on d as a variable
        func=mp.abs(func).simplify()
        integral=mp.mean(func)*(xlim[1]-xlim[0])
        integral=integral.rename_variables({"x":"d"})
        func=func/integral
        logloss=mp.log(func)
        logloss=-1*logloss.simplify()
        #print(logloss)

        logloss=solve(logloss)
        eva=logloss(x=data,d=X)#thats a super confusing naming
        loss=np.mean(eva)
        return loss
    except Exception as e:
        #print(e)
        return 1.e10

def sigmoid(x):
    return 1/(1+np.exp(-x))

def update(o1,o2):
    if np.random.random()<0.2*sigmoid(np.log(o1.complexity())):
        return init()
    if np.random.random()<0.5:
        o1,o2=o2,o1
    return o1.offspring(o2)

def init():
    func=munc().random_function()
    return func



sol,error,hist,mats=phase_search(eval,update,init)


sol=solve(sol)
while np.any(np.isnan(list(sol.extract_guesses().values()))):
    sol=solve(sol)


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


inp=np.linspace(xlim[0],xlim[1],100)

plt.figure(figsize=(8,8))

def plotfunc(func):
    out=func(x=inp)
    if not (type(out) is np.ndarray):
        out=np.ones_like(inp)*out
    norm=func(x=X)
    out=out/(np.mean(norm)*(xlim[1]-xlim[0]))
    plt.plot(inp,out,label="fit")
plotfunc(sol)    
plt.hist(data,bins=100,density=True,label="data")
plt.legend()
plt.savefig(f"imgs/{ident}.png",format="png",dpi=300)
plt.savefig(f"imgs/{ident}.pdf",format="pdf")
plt.show()


def encode_hist(hist=hist):
    return {"hist_func":[a for a,b in hist.items()],"hist_qual":[b for a,b in hist.items()]}

data={"ident":ident,"X":X,"data":data,"true":repr(solution),"bestfit":repr(sol),"error":error,"guesses":json.dumps(sol.extract_guesses()),**encode_hist(hist),"mats":mats,"xlim":xlim}

np.savez_compressed(f"results/{ident}.npz",**data)

