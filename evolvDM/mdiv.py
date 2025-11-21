from munc import munc, objects, epsilon
from mtwoarg import mtwoarg

import numpy as np


class mdiv(mtwoarg):
    def __str__(self):
        return "("+str(self.a)+"/"+str(self.b)+")"

    def diff(self, x):
        ap,bp=self.a.diff(x),self.b.diff(x)
        return objects.mdiv(objects.msub(objects.mmul(ap,self.b),objects.mmul(self.a,bp)),objects.mmul(self.b,self.b))

    def __call__(self, **kwargs):
        a,b=self.a(**kwargs),self.b(**kwargs)
        if type(b) is np.ndarray:
            b[np.where(np.logical_and(b<epsilon,b>=0))]=epsilon
            b[np.where(np.logical_and(b>-epsilon,b<0))]=-epsilon
        else:
            if b<epsilon and b>=0:
                b=epsilon
            if b>-epsilon and b<0:
                b=-epsilon
        return a/b

    def simplify(self):
        a,b=self.a.simplify(),self.b.simplify()
        if a.evaluatable():
            a=objects.mconst(a())
        if b.evaluatable():
            b=objects.mconst(b())
        if type(a)==objects.mconst and a.val==0:
            return objects.mconst(0)
        return mdiv(a,b)
    
    def tolatex(self,*args,**kwargs):
        return "\\frac{"+self.a.tolatex(*args,**kwargs)+"}{"+self.b.tolatex(*args,**kwargs)+"}"

objects.mdiv=mdiv


