from munc import munc, objects, epsilon
import numpy as np
from monearg import monearg


class msquare(monearg):
    def __str__(self):
        return f"square({self.arg})"

    def diff(self, x):
        return self.arg.diff(x) * self.arg * 2

    def __call__(self, **kwargs):
        arg=self.arg(**kwargs)
        return np.square(arg)

    def simplify(self):
        arg=self.arg.simplify()
        if arg.evaluatable():
            arg= objects.mconst(arg())
        return msquare(arg)

    def tolatex(self,*args,**kwargs):
        return f"\\left({self.arg.tolatex(*args,**kwargs)}\\right)^2"

objects.msquare=msquare


