import numpy as np
from SketchFinal import SketchClass, GaussianSketch,countSketch
from SketchFinal import SRHT, GRHT, SRFT, MagicGraph, ExpanderGraph
from SketchFinal import get_rel_error, get_distortion
from numpy import linalg as LA
import matplotlib.pyplot as plt


m = 1000
n = 800
rank = 30
sketchsize = 40
inputdim = m
applyLeft = True #if not specified, apply on the left

matrixA = np.random.normal(0, 1**2, (m,n))
#matrixA = np.loadtxt('./data/jester.txt', dtype=float).T[:4096,:]
print(matrixA.shape)

U, S, V = np.linalg.svd(matrixA,full_matrices=False)
matrixA = np.dot(U[:,:rank]*S[:rank],V[:rank,:])
print('The matrix A has', matrixA.shape[0], 'rows and', matrixA.shape[1], 'columns.')
print('rank =',LA.matrix_rank(matrixA))


Gsketch 	= []
Gsketch_std = []
Csketch 	= []
Csketch_std = []
Magicsketch = []
MS_std		= []
exp_1		= []
exp_2		= []
exp_3		= []


repeat = 3
sindex = []
M = 800
increment = 40
while sketchsize <= M:
	print(sketchsize)
	sindex.append(sketchsize)
	Gerror 		= []
	Cerror 		= []	
	Magic_error = []
	e1			= []
	e2			= []
	e3			= []
	
	for i in range (repeat):
		print('repeat =',i)
		Gclass 		= GaussianSketch(inputdim, sketchsize)
		matrixC 	= Gclass.Apply(matrixA, applyLeft)
		Gerror.append(get_distortion(matrixA, matrixC))

		Cclass 		= countSketch(inputdim, sketchsize)
		matrixC 	= Cclass.Apply(matrixA, applyLeft)
		Cerror.append(get_distortion(matrixA, matrixC))

#		MagicClass 	= MagicGraph(inputdim, sketchsize)
#		matrixC 	= MagicClass.Apply(matrixA, applyLeft, num_randmatch = 2)
#		Magic_error.append(get_distortion(matrixA, matrixC))
#		print('m',MagicClass.Matricize())
		
		expander1	= ExpanderGraph(inputdim, sketchsize)
		matrixC		= expander1.Apply(matrixA, applyLeft, sparsity = 1)
		e1.append(get_distortion(matrixA, matrixC))
		
		expander2	= ExpanderGraph(inputdim, sketchsize)
		matrixC		= expander2.Apply(matrixA, applyLeft, sparsity = 2)
		e2.append(get_distortion(matrixA, matrixC))
#		print('e',expander1.Matricize())
		expander3	= ExpanderGraph(inputdim, sketchsize)
		matrixC		= expander3.Apply(matrixA, applyLeft, sparsity = 3)
		e3.append(get_distortion(matrixA, matrixC))
		
	Gsketch.append(np.mean(Gerror))
	Gsketch_std.append(np.std(Gerror))
	
	Csketch.append(np.mean(Cerror))
	Csketch_std.append(np.std(Cerror))
	
	Magicsketch.append(np.mean(Magic_error))
	MS_std.append(np.std(Magic_error))
	
	exp_1.append(np.mean(e1))
	exp_2.append(np.mean(e2))
	exp_3.append(np.mean(e3))
	sketchsize+=increment
	
#np.savez('errorlowrank.npz',m=m, n=n,rank=rank,sindex=sindex, Gsketch=Gsketch,Csketch=Csketch,SRHTsketch=SRHTsketch,SRFTsketch=SRFTsketch,GRHTsketch=GRHTsketch,Magicsketch=Magicsketch,Gsketch_std=Gsketch_std,Csketch_std=Csketch_std,SRHT_std=SRHT_std,SRFT_std=SRFT_std,GRHT_std=GRHT_std,MS_std=MS_std)

#plotting
#plt.plot(sindex, Gsketch,marker='.', label='Gaussian Sketch')
#plt.plot(sindex, Csketch,marker='o', label='Count Sketch')
plt.plot(sindex, exp_1,marker='v', label='expander with row sparsity=1')
plt.plot(sindex, exp_2,marker='s', label='expander with row sparsity=2')
plt.plot(sindex, exp_3,marker='h', label='expander with row sparsity=3')
#plt.plot(sindex, Magicsketch,marker='+', label='MagicGraph')
plt.legend()
plt.xlabel("sketching size s",fontsize=15)
plt.ylabel(r'$||(A^TA)^{\dagger/2}\tilde{A}^T\tilde{A}(A^TA)^{\dagger/2}-P_{A^TA}||_2$',fontsize=15)
#plt.ylabel(r'$\frac{||SAx||_2-||Ax||_2}{||Ax||_2}$',fontsize=15)
plt.title('sketching methods on A with size '+str(m)+' by '+str(n)+' and rank = '+str(rank))
#fig.savefig("plot.png")
plt.show()