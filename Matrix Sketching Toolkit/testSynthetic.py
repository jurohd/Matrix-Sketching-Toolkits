import numpy as np
from SketchFinal import SketchClass, GaussianSketch,countSketch
from SketchFinal import SRHT, GRHT, SRFT, MagicGraph, ExpanderGraph
from SketchFinal import get_rel_error, get_distortion, get_LR_error, low_rank_approx
from numpy import linalg as LA
import matplotlib.pyplot as plt



m = 4096
n = 1000
rank = 1000
low_rank = 200
sketchsize = 200
inputdim = m
applyLeft = True #if not specified, apply on the left

matrixA = np.random.normal(0, 0.1**2, (m,n))
#matrixA = np.loadtxt('./data/jester.txt', dtype=float).T[:4096,:]
print(matrixA.shape)

#U, S, V = np.linalg.svd(matrixA,full_matrices=False)
#matrixA = np.dot(U[:,:rank]*S[:rank],V[:rank,:])
#print('The matrix A has', matrixA.shape[0], 'rows and', matrixA.shape[1], 'columns.')
#print('rank =',LA.matrix_rank(matrixA))
#_, s, _ = np.linalg.svd(matrixA,full_matrices=False)
#print(len(s))
#d = np.linspace(1,len(s),len(s))
#plt.plot(d,s)
#plt.show()

repeat = 3
sindex = []
M = 1600
increment = 100

show_lr_error = False

if show_lr_error:
	repeat = 2
	M = rank
	increment = M//8

if show_lr_error:
	A_lr = low_rank_approx(matrixA, low_rank)
	standard = LA.norm(matrixA-A_lr,'fro')
	Gsketch_lr 	= []
	Csketch_lr 	= []
	SRFT_lr		= []
	SRHT_lr		= []
	MG_lr		= []
	exp_1_lr	= []
	exp_2_lr	= []
	exp_3_lr	= []
else:
	Gsketch 	= []
	Csketch 	= []
	SRHTe 		= []
	SRFTe 		= []
	MGe			= []
	exp_1		= []
	exp_2		= []
	exp_3		= []




while sketchsize <= M:
	print(sketchsize)
	sindex.append(sketchsize)

	if show_lr_error:
		Gerror_lr	= []
		Cerror_lr 	= []	
		SRFTe_lr 	= []
		SRHTe_lr 	= []	
		MGe_lr		= []
		e1_lr		= []
		e2_lr		= []
		e3_lr		= []
	else:
		Gerror 		= []
		Cerror 		= []
		SRFTerr 	= []
		SRHTerr		= []
		MGerr		= []	
		e1			= []
		e2			= []
		e3			= []
	
	for i in range (repeat):
		print('repeat =',i)
		Gclass 		= GaussianSketch(inputdim, sketchsize)
		matrixC 	= Gclass.Apply(matrixA, applyLeft)
		if show_lr_error:
			Gerror_lr.append(get_LR_error(matrixA, matrixC,low_rank))
		else:
			Gerror.append(get_distortion(matrixA, matrixC))


		Cclass 		= countSketch(inputdim, sketchsize)
		matrixC 	= Cclass.Apply(matrixA, applyLeft)
		if show_lr_error:
			Cerror_lr.append(get_LR_error(matrixA, matrixC,low_rank))
		else:
			Cerror.append(get_distortion(matrixA, matrixC))
	
		SRFTclass	= SRFT(inputdim, sketchsize)
		matrixC 	= SRFTclass.Apply(matrixA, applyLeft)
		if show_lr_error:
			SRFTe_lr.append(get_LR_error(matrixA, matrixC,low_rank))
		else:
			SRFTerr.append(get_distortion(matrixA, matrixC))
			
		SRHTclass	= SRHT(inputdim, sketchsize)
		matrixC 	= SRHTclass.Apply(matrixA, applyLeft)
		if show_lr_error:
			SRHTe_lr.append(get_LR_error(matrixA, matrixC,low_rank))
		else:
			SRHTerr.append(get_distortion(matrixA, matrixC))
			
		MGclass		= MagicGraph(inputdim, sketchsize)
		matrixC 	= MGclass.Apply(matrixA, applyLeft)
		if show_lr_error:
			MGe_lr.append(get_LR_error(matrixA, matrixC,low_rank))
		else:
			MGerr.append(get_distortion(matrixA, matrixC))
			
		expander1	= ExpanderGraph(inputdim, sketchsize)
		matrixC		= expander1.Apply(matrixA, applyLeft, sparsity = 1)
		if show_lr_error:
			e1_lr.append(get_LR_error(matrixA, matrixC,low_rank))
		else:
			e1.append(get_distortion(matrixA, matrixC))

		expander2	= ExpanderGraph(inputdim, sketchsize)
		matrixC		= expander2.Apply(matrixA, applyLeft, sparsity = 2)
		if show_lr_error:
			e2_lr.append(get_LR_error(matrixA, matrixC,low_rank))
		else:
			e2.append(get_distortion(matrixA, matrixC))

		expander3	= ExpanderGraph(inputdim, sketchsize)
		matrixC		= expander3.Apply(matrixA, applyLeft, sparsity = 3)
		if show_lr_error:
			e3_lr.append(get_LR_error(matrixA, matrixC,low_rank))
		else:
			e3.append(get_distortion(matrixA, matrixC))


	
	if show_lr_error:	
		Gsketch_lr.append(np.mean(Gerror_lr))
		Csketch_lr.append(np.mean(Cerror_lr))
		SRFT_lr.append(np.mean(SRFTe_lr))
		SRHT_lr.append(np.mean(SRHTe_lr))
		MG_lr.append(np.mean(MGe_lr))
		exp_1_lr.append(np.mean(e1_lr))
		exp_2_lr.append(np.mean(e2_lr))
		exp_3_lr.append(np.mean(e3_lr))
	else:
		Gsketch.append(np.mean(Gerror))
		Csketch.append(np.mean(Cerror))
		SRFTe.append(np.mean(SRFTerr))
		SRHTe.append(np.mean(SRHTerr))
		MGe.append(np.mean(MGerr))
		exp_1.append(np.mean(e1))
		exp_2.append(np.mean(e2))
		exp_3.append(np.mean(e3))
	
	sketchsize+=increment

if show_lr_error == False:
	np.savez('SYN_error_distortion.npz',m=m, n=n,rank=rank,sindex=sindex, Gsketch=Gsketch,Csketch=Csketch,SRFTe=SRFTe,SRHTe=SRHTe,MGe=MGe,exp_1=exp_1,exp_2=exp_2,exp_3=exp_3)

	#plotting distortion
	plt.plot(sindex, Gsketch,marker='.', label='Gaussian Sketch')
	plt.plot(sindex, Csketch,marker='o', label='Count Sketch')
	plt.plot(sindex, SRFTe,marker='+', label='SRFT')
	plt.plot(sindex, SRHTe,marker='x', label='SRHT')
	plt.plot(sindex, SRHTe,marker='^', label='Magic Graph')
	plt.plot(sindex, exp_1,marker='v', label='expander with s=1')
	plt.plot(sindex, exp_2,marker='s', label='expander with s=2')
	plt.plot(sindex, exp_3,marker='h', label='expander with s=3')

	plt.legend(fontsize=14)
	plt.xlabel("sketching size s",fontsize=15)
	plt.ylabel(r'$||(A^TA)^{\dagger/2}\tilde{A}^T\tilde{A}(A^TA)^{\dagger/2}-P_{A^TA}||_2$',fontsize=15)
	#plt.ylabel(r'$\frac{||SAx||_2-||Ax||_2}{||Ax||_2}$',fontsize=15)
#	plt.title('Distortion for difsketching methods on Synthetic dataset(size:4096x1000)')
	#fig.savefig("plot.png")
	plt.show()

#plotting LR error
if show_lr_error:
	np.savez('SYN_error_lowrank.npz',m=m, n=n,rank=rank,sindex=sindex, Gsketch_lr=Gsketch_lr,Csketch_lr=Csketch_lr,SRFT_lr=SRFT_lr,exp_1_lr=exp_1_lr,exp_2_lr=exp_2_lr,exp_3_lr=exp_3_lr)
	plt.plot(sindex, Gsketch_lr/standard,marker='.', label='Gaussian Sketch')
	plt.plot(sindex, Csketch_lr/standard,marker='o', label='Count Sketch')
	plt.plot(sindex, SRFT_lr/standard,marker='+', label='SRFT')
	plt.plot(sindex, SRHT_lr/standard,marker='x', label='SRHT')
	plt.plot(sindex, MG_lr/standard,marker='^', label='Magic Graph')
	plt.plot(sindex, exp_1_lr/standard,marker='v', label='expander with s=1')
	plt.plot(sindex, exp_2_lr/standard,marker='s', label='expander with s=2')
	plt.plot(sindex, exp_3_lr/standard,marker='h', label='expander with s=3')
	#plt.plot(sindex, Magicsketch,marker='+', label='MagicGraph')
	plt.legend(fontsize=14)
	plt.xlabel("sketching size s",fontsize=15)
	plt.ylabel(r'$\frac{||A-{\tilde A}_r||_F}{||A-A_r||_F}$',fontsize=15)
	#plt.ylabel(r'$\frac{||SAx||_2-||Ax||_2}{||Ax||_2}$',fontsize=15)
#	plt.title('Low rank approximation error on Synthetic dataset(size:4096x1000)'+'r='+str(low_rank))
	#fig.savefig("plot.png")
	plt.show()
