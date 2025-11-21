import numpy as np
import matplotlib.pyplot as plt
import sys

fn=sys.argv[0]
fn=fn[:fn.find(".")]

from const import init_range,evaluations,width,parallel_count,matrixmodulo

from tqdm import tqdm

from multiprocessing import Pool

def xy(i,dx,dy):
    return i%dx,i//dx
def I(x,y,dx,dy):
    return x+y*dx

def find_neighbors(i,dx,dy):
    #infinite border conditions
    x,y=xy(i,dx,dy)
    neighbors=[]
    if x>0:
        neighbors.append(i-1)
    else:
        neighbors.append(I(dx-1,y,dx,dy))
    if x<dx-1:
        neighbors.append(i+1)
    else:
        neighbors.append(I(0,y,dx,dy))
    if y>0:
        neighbors.append(i-dx)
    else:
        neighbors.append(I(x,dy-1,dx,dy))
    if y<dy-1:
        neighbors.append(i+dx)
    else:
        neighbors.append(I(x,0,dx,dy))
    return neighbors

def non_iteracting(n,dx,dy):
    print("called non_iteracting",n,dx,dy)
    ret=[]
    taken=[]
    mx=dx*dy
    while len(ret)<n:
        i=None
        while i is None or i in taken:
            i=np.random.randint(0,mx)
        neigb=find_neighbors(i,dx,dy)
        neigb=[x for x in neigb if x not in taken]
        if len(neigb)==0:
            #print("found 0",len(ret),len(taken),mx,n)
            continue
        j=np.random.choice(neigb)
        ret.append([i,j])
        taken.append(i)
        taken.append(j)
    return ret


if __name__=="__main__" and True:
    for zw in non_iteracting(10,10,10):
        print(*zw)
    exit()



def random_neighbor(i,dx,dy):
    return np.random.choice(find_neighbors(i,dx,dy))

def list_to_matrix(lis,dx,dy):
    assert len(lis)==dx*dy
    return np.array(lis).reshape((dx,dy))

def phase_search(call,update,init,n=evaluations,maximum=init_range,temp=100,dx=width,dy=width,matrixmodulo=matrixmodulo):
    parallel=dx*dy

    history={}
    matrices=[]
    library=[init() for _ in range(parallel)]
    score=[]
    bestscore=None
    bestobj=None
    for obj in library:
        ac=call(obj)
        score.append(ac)
        history[repr(obj)]=ac
        if bestscore is None or ac<bestscore:
            bestscore=ac
            bestobj=obj
    score=np.array(score)
    #for i in range(100):print("!!!!!!!finished initial")


    def _update(i,j,bestscore,bestobj):
        merge=update(library[i],library[j])
        newscore=call(merge)
        history[repr(merge)]=newscore
        if np.random.uniform(0,1)<np.exp(temp*(score[i]-newscore)):
        #if newscore<score[i]:
            library[i]=merge
            score[i]=newscore
            if newscore<bestscore:
                bestscore=newscore
                bestobj=merge
                #print(bestobj)
        return bestscore,bestobj

    def update_one(bestscore,bestobj):
        i=np.random.randint(0,parallel)
        j=random_neighbor(i,dx,dy)
        return _update(i,j,bestscore,bestobj)

    def one_step(bestscore,bestobj):
        scores,objs=[],[]
        with Pool(parallel) as pool:
            for i,j in non_iteracting(parallel_count,dx,dy):
                print("starting ",i,j)
                score,obj=pool.apply(update_one,args=(i,j,bestscore,bestobj))
                scores.append(score)
                objs.append(obj)
        bestscore=np.min(scores)
        bestobj=objs[np.argmin(scores)]
        return bestscore,bestobj

    if parallel_count==1:
        for i in tqdm(range(n-dx*dy),total=n-dx*dy):
            bestscore,bestobj=update_one(bestscore,bestobj)
            if not i%matrixmodulo:matrices.append(list_to_matrix(score,dx,dy))
    else:
        for i in tqdm(range(n//parallel_count),total=n//parallel_count):
            bestscore,bestobj=one_step(bestscore,bestobj)

    return bestobj, bestscore, history,np.array(matrices)





