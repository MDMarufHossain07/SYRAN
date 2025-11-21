from munc import munc, objects

import numpy as np

class mtwoarg(munc):
    def __init__(self,a,b):
        self.a=a
        self.b=b

    def __repr__(self):
        return self.__class__.__name__+"("+repr(self.a)+","+repr(self.b)+")"

    def copy(self,arg1=None,arg2=None):
        if arg1 is None:
            arg1=self.a.copy()
        if arg2 is None:
            arg2=self.b.copy()
        return self.__class__(arg1,arg2)

    def children(self):
        yield self.a
        yield self.b

    def sub_recursion(self,func):
        self.a=func(self.a)
        self.b=func(self.b)
        a=self.a.recursive_update(func)
        b=self.b.recursive_update(func)
        return self.copy(a,b)

    def IamTwoArg(self):pass

    def mutate(self):
        ca,cb=self.a.count_children(),self.b.count_children()
        proba=1/(1+ca+cb)
        if np.random.rand()<proba:
            return self._mutate()
        elif np.random.rand()<ca/(ca+cb):
            return self.copy(self.a.mutate(),self.b)
        else:
            return self.copy(self.a,self.b.mutate())

    def _selfmutate(self):
        if np.random.rand()<0.2:
            return self.copy(self.b.copy(),self.a.copy())
        else:
            return None

objects.mtwoarg=mtwoarg

