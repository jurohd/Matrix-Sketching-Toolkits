import numpy as np
from SketchFinal import SketchClass, MagicGraph
from SketchFinal import get_rel_error, get_distortion
from numpy import linalg as LA
import matplotlib.pyplot as plt

m = 4096
n = 100
rank = 40
sketchsize = 30
inputdim = m
applyLeft = True #if not specified, apply on the left

matrixA = np.random.normal(0, 1**2, (m,n))
#matrixA = np.loadtxt('./data/jester.txt', dtype=float).T[:4096,:]
print(matrixA.shape)

U, S, V = np.linalg.svd(matrixA,full_matrices=False)
matrixA = np.dot(U[:,:rank]*S[:rank],V[:rank,:])
print('The matrix A has', matrixA.shape[0], 'rows and', matrixA.shape[1], 'columns.')
print('rank =',LA.matrix_rank(matrixA))



repeat = 2

M = 500
increment = 20

for k in range (2,20,4):
	sindex = []
	Magicsketch = []
	MS_std		= []
	sketchsize = 30
	while sketchsize <= M:
		print(sketchsize)
		sindex.append(sketchsize)
		Magic_error = []
		
		for i in range (repeat):
			print('repeat =',i)
			MagicClass 	= MagicGraph(inputdim, sketchsize)
			matrixC 	= MagicClass.Apply(matrixA, applyLeft, num_randmatch = k)
			Magic_error.append(get_distortion(matrixA, matrixC))
			
		Magicsketch.append(np.mean(Magic_error))
		MS_std.append(np.std(Magic_error))
		
		sketchsize+=increment
		
	plt.plot(sindex, Magicsketch, label='_k='+str(k))


plt.legend()
plt.xlabel("sketching size s",fontsize=15)
plt.ylabel(r'$||(A^TA)^{\dagger/2}\tilde{A}^T\tilde{A}(A^TA)^{\dagger/2}-P_{A^TA}||_2$',fontsize=15)
#plt.ylabel(r'$\frac{||SAx||_2-||Ax||_2}{||Ax||_2}$',fontsize=15)
plt.title('MagicGraph SKETCHING on A with size '+str(m)+' by '+str(n)+' and rank = '+str(rank))
#fig.savefig("plot.png")
plt.show()