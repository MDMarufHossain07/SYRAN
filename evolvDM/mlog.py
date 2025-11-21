from munc import munc, objects, epsilon
import numpy as np
from monearg import monearg


class mlog(monearg):
    def __str__(self):
        return f"log(|{self.arg}|)"

    def diff(self, x):
        return objects.mdiv(self.arg.diff(x), self.arg)

    def __call__(self, **kwargs):
        arg=self.arg(**kwargs)
        arg=np.abs(arg)
        if type(arg) is np.ndarray:
            arg[np.where(arg<epsilon)]=epsilon
        else:
            if arg<epsilon:
                arg=epsilon
        return np.log(arg)

    def simplify(self):
        arg=self.arg.simplify()
        if arg.evaluatable():
            arg=objects.mconst(arg())
        return mlog(arg)

    def tolatex(self,*args,**kwargs):
        return "\\log\\left|"+self.arg.tolatex(*args,**kwargs)+"\\right|"

objects.mlog=mlog


