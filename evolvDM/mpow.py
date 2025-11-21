from munc import munc, objects, epsilon
from mtwoarg import mtwoarg

import mumpy as mp
import numpy as np

def info(var):
    try:
        return var.shape
    except AttributeError:
        return var

class mpow(mtwoarg):
    def __str__(self):
        return "("+str(self.a)+"^"+str(self.b)+")"

    def diff(self, x):
        a,b=self.a,self.b
        ap,bp=self.a.diff(x),self.b.diff(x)
        return mp.pow(a,(b-1))*(b*ap+a*mp.log(a)*bp)
        
    def __call__(self, **kwargs):
        a,b=self.a(**kwargs),self.b(**kwargs)
        #print("calling mpow with a,b=",a,b)
        #print(**kwargs)
        #print("running mpow.__call__ on",a.shape,b.shape)
        #a=np.cast[np.float64](a)
        #b=np.cast[np.float64](b)
        if type(a) is np.ndarray and type(b) is np.ndarray:
            if np.any(np.logical_and(b<0,np.abs(a)<epsilon)):
                aa=a[np.where(np.logical_and(b<0,np.abs(a)<epsilon))]
                a[np.where(np.logical_and(b<0,np.abs(a)<epsilon))]=np.sign(aa)*epsilon
        elif type(a) is np.ndarray:
            if b<0:
                if np.any(np.abs(a)<epsilon):
                    aa=a[np.where(np.abs(a)<epsilon)]
                    a[np.where(np.abs(a)<epsilon)]=np.sign(aa)*epsilon
        elif type(b) is np.ndarray:
            if a<epsilon:
                ret=np.power(a,b)
                if np.any(b<0):ret[np.where(b<0)]=np.power(epsilon,b[np.where(b<0)])
                return ret
        else:
            if np.abs(a)<epsilon and b<0:
                return np.sign(a)*epsilon**b
        #if a is integer and b<0 make a float
        if type(a) is np.int64 or np.int32 and np.any(b<0):
            a=np.cast[np.float64](a)
        if type(a) is int and np.any(b<0):
            a=float(a)
        return np.power(a,b)

    def simplify(self):
        a,b=self.a.simplify(),self.b.simplify()
        if a.evaluatable():
            a=objects.mconst(a())
        if b.evaluatable():
            b=objects.mconst(b())
        if type(a)==objects.mconst and a.val==0:
            return objects.mconst(0)
        if type(b)==objects.mconst and b.val==0:
            return objects.mconst(1)
        if type(a)==objects.mconst and a.val==1:
            return objects.mconst(1)
        if type(b)==objects.mconst and b.val==1:
            return a
        return mpow(a,b)

    def tolatex(self,*args,**kwargs):
        return "("+self.a.tolatex(*args,**kwargs)+")^{"+self.b.tolatex(*args,**kwargs)+"}"

objects.mpow=mpow


