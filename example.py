import scipy.optimize as opt
import numpy as np

def objective(x,coeffs):
    
    return coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]


x0 = 3.0
mycoeffs = [1.0,-2.0,0.0]

print(objective(x0, mycoeffs))


# myoptions={'disp':True}
# results = opt.minimize(objective,x0,args=mycoeffs,
# options = myoptions)
# print("Solution: x=%f" % results.x)