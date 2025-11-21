from munc import munc, objects, epsilon
import numpy as np
from monearg import monearg


class mexp(monearg):
    def __str__(self):
        return f"exp({self.arg})"

    def diff(self, x):
        return objects.mmul(self.arg.diff(x), self.copy())

    def __call__(self, **kwargs):
        arg=self.arg(**kwargs)
        return np.exp(arg)

    def simplify(self):
        arg=self.arg.simplify()
        if arg.evaluatable():
            arg= objects.mconst(arg())
        return mexp(arg)

    def tolatex(self,*args,**kwargs):
        return "\\exp\\left("+self.arg.tolatex(*args,**kwargs)+"\\right)"

objects.mexp=mexp


