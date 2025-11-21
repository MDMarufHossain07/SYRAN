from munc import munc, objects
from mtwoarg import mtwoarg


class mmul(mtwoarg):

    def __str__(self):
        return "("+str(self.a)+"*"+str(self.b)+")"

    def diff(self, x):
        return mmul(self.a.diff(x),self.b)+mmul(self.a,self.b.diff(x))

    def __call__(self, **kwargs):
        return self.a(**kwargs)*self.b(**kwargs)

    def simplify(self):
        a,b=self.a.simplify(),self.b.simplify()
        if a.evaluatable():
            a=objects.mconst(a())
        if b.evaluatable():
            b=objects.mconst(b())
        if type(a)==objects.mconst and a.val==0:
            return objects.mconst(0)
        if type(b)==objects.mconst and b.val==0:
            return objects.mconst(0)
        if type(a)==objects.mconst and a.val==1:
            return b
        if type(b)==objects.mconst and b.val==1:
            return a
        return mmul(a,b)

    def tolatex(self,*args,**kwargs):
        return self.a.tolatex(*args,**kwargs)+" \\cdot "+self.b.tolatex(*args,**kwargs)

objects.mmul=mmul


