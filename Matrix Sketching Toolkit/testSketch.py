import numpy as np
from SketchFinal import SketchClass, GaussianSketch,countSketch
from SketchFinal import SRHT, GRHT, SRFT, MagicGraph
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


Gsketch 	= []
Gsketch_std = []
Csketch 	= []
Csketch_std = []
SRHTsketch 	= []
SRHT_std	= []
SRFTsketch 	= []
SRFT_std	= []
GRHTsketch 	= []
GRHT_std	= []
Magicsketch = []
MS_std		= []

repeat = 5
sindex = []
M = 500
increment = 50
while sketchsize <= M:
	print(sketchsize)
	sindex.append(sketchsize)
	Gerror 		= []
	Cerror 		= []	
	SRHTerror 	= []
	SRFTerror 	= []
	GRHTerror 	= []
	Magic_error = []
	
	for i in range (repeat):
		print('repeat =',i)
		Gclass 		= GaussianSketch(inputdim, sketchsize)
		matrixC 	= Gclass.Apply(matrixA, applyLeft)
#		Gclass.Sanitycheck()
		Gerror.append(get_distortion(matrixA, matrixC))
		Cclass 		= countSketch(inputdim, sketchsize)
		matrixC 	= Cclass.Apply(matrixA, applyLeft)
		Cerror.append(get_distortion(matrixA, matrixC))
		
		SRHTclass 	= SRHT(inputdim, sketchsize)
		matrixC 	= SRHTclass.Apply(matrixA, applyLeft)
		SRHTerror.append(get_distortion(matrixA, matrixC))
		
		GRHTclass 	= GRHT(inputdim, sketchsize)
		matrixC 	= GRHTclass.Apply(matrixA, applyLeft)
		GRHTerror.append(get_distortion(matrixA, matrixC))
		
		SRFTclass 	= SRFT(inputdim, sketchsize)
		matrixC 	= SRFTclass.Apply(matrixA, applyLeft)
		SRFTerror.append(get_distortion(matrixA, matrixC))
		
		MagicClass 	= MagicGraph(inputdim, sketchsize)
		matrixC 	= MagicClass.Apply(matrixA, applyLeft, num_randmatch = 2)
		Magic_error.append(get_distortion(matrixA, matrixC))
		
	Gsketch.append(np.mean(Gerror))
	Gsketch_std.append(np.std(Gerror))
	
	Csketch.append(np.mean(Cerror))
	Csketch_std.append(np.std(Cerror))
	
	SRHTsketch.append(np.mean(SRHTerror))
	SRHT_std.append(np.std(SRHTerror))
	
	SRFTsketch.append(np.mean(SRFTerror))
	SRFT_std.append(np.std(SRFTerror))
	
	GRHTsketch.append(np.mean(GRHTerror))
	GRHT_std.append(np.std(GRHTerror))
	
	Magicsketch.append(np.mean(Magic_error))
	MS_std.append(np.std(Magic_error))
	
	sketchsize+=increment
	
np.savez('errorlowrank.npz',m=m, n=n,rank=rank,sindex=sindex, Gsketch=Gsketch,Csketch=Csketch,SRHTsketch=SRHTsketch,SRFTsketch=SRFTsketch,GRHTsketch=GRHTsketch,Magicsketch=Magicsketch,Gsketch_std=Gsketch_std,Csketch_std=Csketch_std,SRHT_std=SRHT_std,SRFT_std=SRFT_std,GRHT_std=GRHT_std,MS_std=MS_std)

#plotting
plt.plot(sindex, Gsketch,marker='.', label='Gaussian Sketch')
plt.plot(sindex, Csketch,marker='o', label='Count Sketch')
plt.plot(sindex, SRHTsketch,marker='v', label='SRHT')
plt.plot(sindex, SRFTsketch,marker='s', label='SRFT')
plt.plot(sindex, GRHTsketch,marker='h', label='GRHT')
plt.plot(sindex, Magicsketch,marker='+', label='MagicGraph')
plt.legend()
plt.xlabel("sketching size s",fontsize=15)
plt.ylabel(r'$||(A^TA)^{\dagger/2}\tilde{A}^T\tilde{A}(A^TA)^{\dagger/2}-P_{A^TA}||_2$',fontsize=15)
#plt.ylabel(r'$\frac{||SAx||_2-||Ax||_2}{||Ax||_2}$',fontsize=15)
plt.title('sketching methods on A with size '+str(m)+' by '+str(n)+' and rank = '+str(rank))
#fig.savefig("plot.png")
plt.show()