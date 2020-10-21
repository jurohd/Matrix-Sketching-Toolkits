from SketchFinal import GRHT
from SketchFinal import get_distortion
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

m = 4096
n = 100
rank = 30
sketchsize = 20
inputdim = m
applyLeft = True #if not specified, apply on the left
repeat = 5 

matrixA = np.random.normal(0, 1**2, (m,n))
#matrixA = np.loadtxt('./data/jester.txt', dtype=float).T[:4096,:]
print(matrixA.shape)

U, S, V = np.linalg.svd(matrixA,full_matrices=False)
matrixA = np.dot(U[:,:rank]*S[:rank],V[:rank,:])
print('The matrix A has', matrixA.shape[0], 'rows and', matrixA.shape[1], 'columns.')
print('rank =',LA.matrix_rank(matrixA))

maxcvalue = math.log10(n)/math.log10(math.log10(n))
print(maxcvalue)
cvalues = np.linspace(0.05,maxcvalue, 5)
GRHTsketch = []
for c in cvalues:
	print('c=',c)
	GRHTerror = []
	while 
	for i in range (repeat):
		GCLASS = GRHT(inputdim, sketchsize)
		matrixC = GCLASS.Apply(matrixA, applyLeft)
		GRHTerror.append(get_distortion(matrixA, matrixC)) 
	GRHTsketch.append(np.mean(GRHTerror))
		
#labelname = 'GHRT_c='+str(c)
plt.plot(cvalues, GRHTsketch, label='GRHT')
plt.legend()
plt.xlabel("sparsity(larger number->dense)")
plt.ylabel("distorsion")
#plt.title('Comparison of different sketching methods')
#fig.savefig("plot.png")
plt.show()
