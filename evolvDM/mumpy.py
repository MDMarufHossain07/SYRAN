from munc import munc, objects, prep


def pow(a,b):
    return objects.mpow(prep(a), prep(b))

def log(x):
    return objects.mlog(prep(x))

def exp(x):
    return objects.mexp(prep(x))

def abs(x):
    return objects.mabs(prep(x))

def square(x):
    return objects.msquare(prep(x))

def sin(x):
    return objects.msin(prep(x))

def cos(x):
    return objects.mcos(prep(x))

def mean(x):
    return objects.mmean(prep(x))

def const(x):
    return objects.mconst(x)

