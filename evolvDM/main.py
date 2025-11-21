import numpy as np
from plt import *

from morga import munc,objects

from morga import mconst,mvar,madd,msub,mmul

import mumpy as mp

from library import library

x=mvar("x")
alpha=mvar("alpha")



solution=mp.exp(mp.abs(alpha)*x)

xlim=[0,1]
alpha_true=3

x=np.linspace(xlim[0],xlim[1],100)
y=solution(x=x,alpha=alpha_true)
y+=np.random.normal(0,0.1,len(y))


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


