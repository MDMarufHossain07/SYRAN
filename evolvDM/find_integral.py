import numpy as np
from plt import *

from morga import munc,objects

from morga import mconst,mvar,madd,msub,mmul

import mumpy as mp

from library import library

from phase_search import phase_search


np.random.seed(12)

x=mvar("x")
alpha=mvar("alpha")

function=mp.sin(x)
function=mp.exp(x+3)


xlim=[-2,2]
X=np.linspace(xlim[0],xlim[1],100)

zero=np.zeros_like(x)

def solve(func):
    return func.solve(zero,x=X)

def eval(func):
    try:
        tomin=mp.square(func.diff(x)-function)
        tomin=solve(tomin)
        loss=np.mean(np.square(tomin(x=X)))
        return loss
    except:
        return 1.e10

def sigmoid(x):
    return 1/(1+np.exp(-x))

def update(o1,o2):
    if np.random.random()<0.5*sigmoid(np.log(o1.complexity())):
        return init()
    if np.random.random()<0.5:
        o1,o2=o2,o1
    return o1.offspring(o2)

def init():
    func=munc().random_function()
    return func



sol,error,hist=phase_search(eval,update,init)

sol=solve(sol)

print("searched for an integral to",function)
print("found")

print(error)
print()
print(sol)
print()
print(sol.extract_guesses())
print()
print(sol.tolatex())


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


