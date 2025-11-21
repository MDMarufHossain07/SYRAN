from munc import munc, objects, epsilon
import numpy as np
from monearg import monearg


class mabs(monearg):
    def __str__(self):
        return f"|{self.arg}|"

    def diff(self, x):
        return objects.mmul(self.arg.diff(x), self.arg)/mabs(self.arg)

    def __call__(self, **kwargs):
        arg=self.arg(**kwargs)
        return np.abs(arg)

    def simplify(self):
        arg=self.arg.simplify()
        if arg.evaluatable():
            arg= objects.mconst(arg())
        return mabs(arg)

    def tolatex(self,*args,**kwargs):
        return f"\\left|{self.arg.tolatex(*args,**kwargs)}\\right|"

objects.mabs=mabs


