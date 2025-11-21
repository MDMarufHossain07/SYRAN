from munc import munc, objects

import numpy as np

class monearg(munc):
    def __init__(self,arg):
        self.arg=arg

    def __repr__(self):
        return self.__class__.__name__+"("+repr(self.arg)+")"

    def copy(self,argument=None):
        if argument is None:
            argument=self.arg.copy()
        return self.__class__(argument)

    def children(self):
        yield self.arg

    def sub_recursion(self,func):
        self.arg=func(self.arg)
        return self.copy(self.arg.recursive_update(func))

    def IamOneArg(self):pass

    def mutate(self):
        chils=self.count_children()
        proba=1/(1+chils)
        if np.random.rand()<proba:
            return self._mutate()
        else:
            return self.copy(self.arg.mutate())

    def _selfmutate(self):
        return None



objects.monearg=monearg

