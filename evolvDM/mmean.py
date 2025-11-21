from munc import munc, objects, epsilon
import numpy as np
from monearg import monearg


class mmean(monearg):
    def __str__(self):
        return f"mean({self.arg})"

    def diff(self, x):
        return mmean(self.arg.diff(x))

    def __call__(self, **kwargs):
        arg=self.arg(**kwargs)
        return np.mean(arg)

    def simplify(self):
        arg=self.arg.simplify()
        if arg.evaluatable():
            arg= objects.mconst(arg())
        return self.__class__(arg)

    def tolatex(self,*args,**kwargs):
        return "mean\\left("+self.arg.tolatex(*args,**kwargs)+"\\right)"

    def forbidden(self):pass

objects.mmean=mmean


