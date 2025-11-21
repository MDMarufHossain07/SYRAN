from munc import munc, objects
from mtwoarg import mtwoarg


class msub(mtwoarg):
    def __str__(self):
        return "("+str(self.a)+"-"+str(self.b)+")"

    def diff(self, x):
        return msub(self.a.diff(x),self.b.diff(x))

    def __call__(self, **kwargs):
        return self.a(**kwargs)-self.b(**kwargs)

    def simplify(self):
        a,b=self.a.simplify(),self.b.simplify()
        if a.evaluatable():
            a=objects.mconst(a())
        if b.evaluatable():
            b=objects.mconst(b())
        if type(a)==objects.mconst and a.val==0:
            return b*(-1)
        if type(b)==objects.mconst and b.val==0:
            return a
        return msub(a,b)

    def tolatex(self,*args,**kwargs):
        return self.a.tolatex(*args,**kwargs)+" - ("+self.b.tolatex(*args,**kwargs)+")"

objects.msub=msub


