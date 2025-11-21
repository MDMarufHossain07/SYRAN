from munc import munc,objects

import numpy as np

from const import maximum_variables

all_variables=[]
class mvar(munc):

    def __init__(self, name=None):
        global all_variables
        if name == "rnd":
            name=np.random.choice(all_variables+[None])
        if name is None:
            name=None
            dex=0
            while name is None or name in all_variables:
                name="auto_"+str(dex)
                dex+=1
        self.name = name
        if len(all_variables)<maximum_variables:#arbitrary constant
            if self.name not in all_variables:
                all_variables.append(self.name)

    def __repr__(self):
        return "mvar('"+str(self.name)+"')"

    def __str__(self):
        return self.name

    def diff(self, x):
        if x==self.name or x==self:
            return objects.mconst(1)
        else:
            return objects.mconst(0)

    def copy(self,*args,**kwargs):
        return self.__class__(self.name)

    def __call__(self, **kwargs):
        if self.name in kwargs:
            return kwargs[self.name]
        else:
            raise Exception("mvar: "+self.name+" unknown")

    def evaluatable(self,**kwargs):
        return self.name in kwargs

    def variables(self):
        return set([self.name])

    def complexity(self):
        return 5

    def tolatex(self,*args,**kwargs):
        return self.name.replace("_","\\_").replace(" ","\\ ").replace("\\","\\\\")

    def compare_parameters(self,other):
        return self.name==other.name

    def _selfmutate(self):
        return mvar("rnd")

objects.mvar=mvar








