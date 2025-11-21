from munc import munc, objects

import numpy as np

class mconst(munc):
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return "mconst("+str(self.val)+")"

    def __str__(self):
        return str(self.val)

    def diff(self, x):
        return mconst(0)

    def __call__(self, **kwargs):
        return self.val

    def copy(self,*args,**kwargs):
        return mconst(self.val)

    def complexity(self):
        return 1

    def tolatex(self,*args,**kwargs):
        return str(self.val)

    def compare_parameters(self,other):
        return self.val==other.val

    def _mutate(self):
        if np.random.uniform()<0.5:
            return objects.mguess("rnd",self.val)
        return mconst(np.random.normal(self.val,np.sqrt(np.abs(self.val))+0.000001))


objects.mconst=mconst


