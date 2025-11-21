#from munc import munc, objects
#from macer import macer
from morga import munc,objects,macer,prep

import mumpy as mp


x,y,z=macer("x"),macer("y"),macer("z")

zero,one=prep(0),prep(1)
two,three,minus1=prep(2),prep(3),prep(-1)
library={}


library["permut+"]=[x+y,y+x]
library["permut*"]=[x*y,y*x]
library["permut-"]=[x-y,(y-x)*(-1)]
library["permut/"]=[x/y,one/(y/x)]
library["invers-"]=[x-x,zero]
library["invers/"]=[x/x,one]
library["invar+"]=[x+zero,x]
library["invar*"]=[x*one,x]
library["invar-"]=[x-zero,x]
library["invar/"]=[x/one,x]
library["invar^"]=[x**one,x]
library["simp^2"]=[x**two,x*x]
library["simp^3"]=[x**three,x*x*x]
library["simp^0"]=[x**zero,one]
library["simp^1"]=[x**one,x]
library["simp^-1"]=[x**minus1,one/x]
library["simp*2"]=[x*two,x+x]
library["simp*3"]=[x*three,x+x+x]
library["simp*0"]=[x*zero,zero]
library["simp*1"]=[x*one,x]
library["invlog"]=[mp.log(mp.exp(x)),x]
library["invexp"]=[mp.exp(mp.log(x)),mp.abs(x)]
library["dist"]=[x*(y+z),x*y+x*z]
library["dist2"]=[x*y+x*z,x*(y+z)]





def summarize():
    print("Library:")
    for key,val in library.items():
        print(key,":",val[0],"=",val[1])



if __name__=="__main__":
    summarize()







