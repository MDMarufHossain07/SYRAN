import numpy as np
import itertools


from const import epsilon





class dictp(dict):
    def __init__(self, *args, **kwargs):
        super(dictp, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, name):
        return self[name]

    def random_onearg(self):
        possib=[key for key,val in self.items() if hasattr(val,"IamOneArg") and not key=="monearg" and not hasattr(val,"forbidden")]
        return self[np.random.choice(possib)]
    def random_twoarg(self):
        possib=[key for key,val in self.items() if hasattr(val,"IamTwoArg") and not key=="mtwoarg" and not hasattr(val,"forbidden")]
        return self[np.random.choice(possib)]

    def random_baseobject(self):
        consts=[0,1,2,3,-1]
        if np.random.random()<0.4:
            return self.mconst(np.random.choice(consts))
        else:
            return self.mvar("rnd")

objects=dictp()

def prep(var):
    if type(var) is float or type(var) is np.float64 or type(var) is np.float32:
        return objects.mconst(float(var))
    if type(var) is int or type(var) is np.int64 or type(var) is np.int32:
        return objects.mconst(int(var))
    if type(var) is str or type(var) is np.str_:
        var=str(var)
        if len(var)>0 and var[0]=='$' and var[-1]=='$':
            return objects.macer(var[1:-1])
        return objects.mvar(str(var))
    if hasattr(var, 'copy'):
        return var.copy()
    return var

class munc(object):
    """base object for mathematical functions. The goal is, to have functions that can easily be modified by evolution, but also for continous parameters to be optimizable by gradient descent"""

    def diff(self, x):
        """returns the derivative of the function"""
        raise NotImplementedError

    def __call__(self, **kwargs):
        """returns the function value"""
        raise NotImplementedError

    def __repr__(self):
        """returns a string representation of the function"""
        return NotImplementedError

    def __str__(self):
        """returns a string representation of the function"""
        return self.__repr__()

    def copy(self,*args,**kwargs):
        """returns a copy of the function"""
        return self.__class__(**self.__dict__)

    def __add__(a,b):
        assert "madd" in objects, "please load madd"
        return objects.madd(prep(a), prep(b))

    def __radd__(a,b):
        assert "madd" in objects, "please load madd"
        return objects.madd(prep(b), prep(a))

    def __sub__(a,b):
        assert "msub" in objects, "please load msub"
        return objects.msub(prep(a), prep(b))

    def __rsub__(a,b):
        assert "msub" in objects, "please load msub"
        return objects.msub(prep(b), prep(a))

    def __mul__(a,b):
        assert "mmul" in objects, "please load mmul"
        return objects.mmul(prep(a), prep(b))

    def __rmul__(a,b):
        assert "mmul" in objects, "please load mmul"
        return objects.mmul(prep(b), prep(a))

    def __truediv__(a,b):
        assert "mdiv" in objects, "please load mdiv"
        return objects.mdiv(prep(a), prep(b))

    def __rtruediv__(a,b):
        assert "mdiv" in objects, "please load mdiv"
        return objects.mdiv(prep(b), prep(a))

    def __pow__(a,b):
        assert "mpow" in objects, "please load mpow"
        return objects.mpow(prep(a), prep(b))

    def __rpow__(a,b):
        assert "mpow" in objects, "please load mpow"
        return objects.mpow(prep(b), prep(a))

    def children(self):
        """iterates over all direct children of this function"""
        if False:
            yield None
        return

    def evaluatable(self, **kwargs):
        """returns true if all variables are defined"""
        for child in self.children():
            if not child.evaluatable(**kwargs):
                return False
        return True

    def variables(self):
        """returns a set of all variables"""
        return set().union(*[child.variables() for child in self.children()])

    def complexity(self):
        """returns a value to estimate the complexity of the function"""
        return sum([2*child.complexity() for child in self.children()])

    def simplify(self):
        """returns a simplified version of the function"""
        return self.copy()

    def tolatex(self,*args,**kwargs):
        """returns a latex representation of the function"""
        return self.__str__()

    def iterate_function(self, func):
        """iterates over all functions in the tree"""
        for child in self.children():
            child.iterate_function(func)
        func(self)
    def iterate_function_topdown(self,func):
        """like iterate_function but top down"""
        func(self)
        for child in self.children():
            child.iterate_function_topdown(func)

    def list_placeholders(self):
        """returns a list of all placeholders"""
        placeholders=set()
        def func(child):
            if type(child) is objects.macer:
                placeholders.add(child.name)
        self.iterate_function(func)
        return placeholders

    def recursive_update(self,func):
        """applies a function to all parts of the function"""
        ret=func(self)
        ret=ret.sub_recursion(func)
        return ret

    def sub_recursion(self,func):
        """applies a function to all children"""
        return self


    def insert_placeholder(self,name,value):
        """inserts a placeholder"""
        def func(child):
            if type(child) is objects.macer and child.name==name:
                return value.copy()
            return child.copy()
        return self.recursive_update(func)

    def has_placeholders(self):
        """returns true if the function contains placeholders"""
        return len(self.list_placeholders())>0

    def compare_parameters(self, other):
        """tests if two sets of parameters are the same"""
        return True

    def equals(self, placeholder, dic=None):
        """Tests if the function follows the placeholder. dic=None: dont allow placeholders"""
        if type(self)==type(placeholder):
            if not self.compare_parameters(placeholder):
                return False
            for child, pchild in zip(self.children(), placeholder.children()):
                if not child.equals(pchild, dic=dic):
                    return False
            return True
        elif type(placeholder) is objects.macer:
            nam=placeholder.name
            if dic is None:
                return False
            if nam in dic:
                return self.equals(dic[nam], dic=None)
            dic[nam]=self.copy()
            return True
        return False


    def __eq__(self, other):
        return self.equals(other,dic=None)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _iterate_alternatives(self, library):
        if type(library) is dict:library=list(library.values())


        for inn,out in library:
            dic={}
            if self.equals(inn,dic=dic):
                curr=out.copy()
                #print("found equivalent function, inserting",curr)
                #print("have dictionary",dic)
                for nam, val in dic.items():
                    curr=curr.insert_placeholder(nam,val)
                #print("result",curr)
                yield curr
        altargs=[[child.copy()]+list(child.iterate_alternatives(library)) for child in self.children()]
        for args in itertools.product(*altargs):
            curr= self.copy(*args)
            yield curr

    def iterate_alternatives(self,library):
        history=set([str(self)])
        todo=[self]
        while len(todo)>0:
            curr=todo.pop()
            for alt in curr._iterate_alternatives(library):
                salt=str(alt)
                if salt not in history:
                    history.add(salt)
                    todo.append(alt)
                    yield alt

    def insert_values(self,dic):
        """replaces mvar(key) by mguess(key,value) for key,value in dic"""
        def func(child):
            if type(child) is objects.mvar or type(child) is objects.mguess:
                if child.name in dic:
                    return objects.mguess(child.name,dic[child.name])
            return child.copy()
        return self.recursive_update(func)

    def rename_variables(self, dic):
        """replaces mvar(key) by mvar(dic[key]), same for mguess"""
        def func(child):
            if type(child) is objects.mvar:
                if child.name in dic:
                    return objects.mvar(dic[child.name])
            if type(child) is objects.mguess:
                if child.name in dic:
                    return objects.mguess(dic[child.name],child.value)
            return child.copy()
        return self.recursive_update(func)

    def random_solve(self,outp,**inp):
        initialize=lambda nam:np.random.normal(0,1)
        method=np.random.choice(["gd","newton"])
        lr=np.exp(-np.random.uniform(0,5))
        max_steps=np.random.randint(1,100)
        start_with_last=np.random.choice([True,False])
        return self.solve(outp,initialize=initialize,method=method,lr=lr,max_steps=max_steps,start_with_last=start_with_last,**inp)

    def solve(self,outp,initialize=lambda nam:np.random.normal(0,1),method="gd",lr=0.05,max_steps=30,start_with_last=True,**inp):
        """Uses gradient descent to find guesses for all parameters that are not provided. These are optimised so that when using inputs inp, outp is given (l2 loss). Afterwards the each mvar outside inp is replaced by mguess objects"""

        param=self.variables()
        ylabel="y"
        while ylabel in param:
            ylabel+="y"
        inputs=set(inp.keys())
        param=param-inputs
        inputs=list(inputs)
        param=list(param)
        if len(param)==0:
            return self.copy()

        starting={para:initialize(para) for para in param}
        for para,val in self.extract_guesses().items():
            starting[para]=val
        if start_with_last:
            last=self.extract_guesses()
            for para in param:
                if para in last:
                    starting[para]=last[para]

        loss=objects.msquare(objects.mvar(ylabel)-self)

        diffs={para:loss.diff(para).simplify() for para in param}
        changes={para:loss/diff for para,diff in diffs.items()}

        curr={key:val for key,val in starting.items()}

        history_tries=[]
        history_loss=[]


        for i in range(max_steps):
            for key in param:
                #newton
                if method=="newton":
                    curr[key]=curr[key]-lr*np.mean(changes[key](**inp,**curr,**{ylabel:outp}))
                #gradient descent
                elif method=="gd":
                    curr[key]=curr[key]-lr*np.mean(diffs[key](**inp,**curr,**{ylabel:outp}))
            lossv=np.mean(loss(**inp,**curr,**{ylabel:outp}))
            history_tries.append(curr.copy())
            history_loss.append(lossv)
            if len(history_loss)>2:
                if history_loss[-1]>history_loss[-2]:
                    break
            #print("step",i,"values",curr)
        #exit()            

        solution=history_tries[np.argmin(history_loss)]
        return self.insert_values(solution)

    def extract_guesses(self):
        """searches for mguess objects, and puts their values into a dictionary"""
        dic={}
        def func(child):
            if type(child) is objects.mguess:
                dic[child.name]=child.value
        self.iterate_function(func)
        return dic

    def count_children(self):
        count=[0]
        def func(child):
            count[0]+=1
        self.iterate_function(func)
        return count[0]

    def _selfmutate(self):
        """changes internal parts of the function"""
        return self.copy()

    def _mutate(self):
        """changes external parts of the function"""
        #25% change to apply another function to this one
        #25% to change to random variable
        #50% to selfmutate
        if np.random.random()<0.5:
            ret= self._selfmutate()
            if ret is not None:
                return ret
        if np.random.random()<0.5:
            return objects.random_baseobject()
            #return objects.mvar("rnd")
        else:
            obj=objects.random_onearg()
            return obj(self.copy())




    def mutate(self):
        """mutates the function, returns a new function"""
        return self._mutate()

    def random_child(self):
        """returns a random child"""
        children=[self]+list(self.children())
        dex=np.random.randint(len(children))
        child=children[dex]
        if dex>0 and np.random.random()<0.2:
            return child.random_child()
        return child.copy()

    def reproduce(self,other):
        """reproduces with another function"""
        if np.random.random()<0.5:
            self,other=other,self
        return objects.random_twoarg()(self.random_child(),other.random_child())
        #return objects.random_twoarg()(self.copy(),other.copy())

    def random_function(self,depth=3):
        if np.random.random()<.5 or depth<=0:
            return objects.random_baseobject()
        if np.random.random()<.5:
            ob=objects.random_onearg()
            return ob(self.random_function(depth-1))
        return objects.random_twoarg()(self.random_function(depth-1),self.random_function(depth-1))


    def offspring(self,other):
        """balanced list of asexual and sexual reproduction"""
        if np.random.random()<0.4:
            return self.mutate().simplify()
        elif np.random.random()<0.5:
            return self.reproduce(other).simplify()
        else:
            return other.random_function().simplify()








        
