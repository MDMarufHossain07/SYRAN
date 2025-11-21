from munc import munc, objects, epsilon
import numpy as np
from monearg import monearg


class mcos(monearg):
    def __str__(self):
        return f"cos({self.arg})"

    def diff(self, x):
        return objects.mmul(objects.mmul(objects.mconst(-1),self.arg.diff(x)), objects.msin(self.arg.copy()))

    def __call__(self, **kwargs):
        arg=self.arg(**kwargs)
        return np.cos(arg)

    def simplify(self):
        arg=self.arg.simplify()
        if arg.evaluatable():
            arg= objects.mconst(arg())
        return mcos(arg)

    def tolatex(self,*args,**kwargs):
        return "\\cos\\left("+self.arg.tolatex(*args,**kwargs)+"\\right)"

objects.mcos=mcos


