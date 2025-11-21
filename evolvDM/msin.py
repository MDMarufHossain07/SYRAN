from munc import munc, objects, epsilon
import numpy as np
from monearg import monearg


class msin(monearg):
    def __str__(self):
        return f"sin({self.arg})"

    def diff(self, x):
        return objects.mmul(self.arg.diff(x), objects.mcos(self.arg.copy()))

    def __call__(self, **kwargs):
        arg=self.arg(**kwargs)
        return np.sin(arg)

    def simplify(self):
        arg=self.arg.simplify()
        if arg.evaluatable():
            arg= objects.mconst(arg())
        return msin(arg)

    def tolatex(self,*args,**kwargs):
        return "\\sin\\left("+self.arg.tolatex(*args,**kwargs)+"\\right)"

objects.msin=msin


