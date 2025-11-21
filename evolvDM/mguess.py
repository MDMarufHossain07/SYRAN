from munc import munc,objects
from mvar import mvar


class mguess(mvar):

    def __init__(self, name, value):
        self.value = value
        super().__init__(name)

    def __repr__(self):
        return "mguess('"+str(self.name)+"',"+str(self.value)+")"

    def __str__(self):
        return super().__str__()+"°"

    def copy(self,*args,**kwargs):
        return self.__class__(self.name,self.value)

    def __call__(self, **kwargs):
        if self.name in kwargs:
            return kwargs[self.name]
        else:
            return self.value

    def evaluatable(self,**kwargs):
        return True

    def complexity(self):
        return super().complexity()-1

objects.mguess = mguess








