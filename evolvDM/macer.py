from munc import munc,objects

class macer(munc):

    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return "macer("+str(self.name)+")"

    def __str__(self):
        return "$"+self.name+"$"

    def diff(self, x):
        raise Exception("macer: "+self.name+" is not differentiable")

    def copy(self,*args,**kwargs):
        return macer(self.name)

    def __call__(self, **kwargs):
        raise Exception("macer: "+self.name+" is a placeholder")

    def evaluatable(self,**kwargs):
        return self.name in kwargs

    def variables(self):
        return set([])

    def complexity(self):
        return 10

    def tolatex(self,*args,**kwargs):
        return "\textbf{"+self.name.replace("_","\\_").replace(" ","\\ ").replace("\\","\\\\")+"}"


    def compare_parameters(self,other):
        return self.name==other.name

    def _selfmutate(self):
        raise Exception("Why should you ever want to mutate a placeholder?")

objects.macer = macer








