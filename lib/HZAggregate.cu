/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Created by: Hang Zhang
 * ECE Department, Rutgers University
 * Email: zhang.hang@rutgers.edu
 * Copyright (c) 2016
 *
 * Feel free to reuse and distribute this software for research or 
 * non-profit purpose, subject to the following conditions:
 *  1. The code must retain the above copyright notice, this list of
 *     conditions.
 *  2. Original authors' names are not deleted.
 *  3. The authors' names are not used to endorse or promote products
 *      derived from this software 
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
__global__ void HZAggregate_Forward_kernel (
	THCDeviceTensor<real, 3> E,
	THCDeviceTensor<real, 3> A,
	THCDeviceTensor<real, 4> R)
{
  /* declarations of the variables */
  int b, k, d, i, N;
	real sum;
  /* Get the index and channels */ 
  b = blockIdx.z;
  d = blockIdx.x * blockDim.x + threadIdx.x;
  k = blockIdx.y * blockDim.y + threadIdx.y;
	N = A.getSize(1);
	/* boundary check for output */
	sum = 0;
	if (d >= E.getSize(2) || k >= E.getSize(1))	return;
	/* main operation */
	for(i=0; i<N; i++) {
		sum += A[b][i][k].ldg() * R[b][i][k][d].ldg();
	}
	E[b][k][d] = sum;
}

void HZAggregate_Forward(THCState *state, THCTensor *E_, THCTensor *A_,
							THCTensor *R_)
/*
 * mapping the image pixels based on the lookuptable
 */
{
	/* Check the GPU index */
	HZENCODING_assertSameGPU(state, 3, E_, A_, R_);
	/* Device tensors */
	THCDeviceTensor<real, 3> E = devicetensor<3>(state, E_);
	THCDeviceTensor<real, 3> A = devicetensor<3>(state, A_);
	THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(E.getSize(2)/16+1, E.getSize(1)/16+1, 
							E.getSize(0));
	HZAggregate_Forward_kernel<<<blocks, threads, 0, stream>>>(E, A, R);
	THCudaCheck(cudaGetLastError());
}

__global__ void HZAggregate_BackwardA_kernel (
	THCDeviceTensor<real, 3> G,
	THCDeviceTensor<real, 3> L,
	THCDeviceTensor<real, 4> R)
{
  /* declarations of the variables */
  int b, k, d, i, D;
	real sum;
  /* Get the index and channels */ 
  b = blockIdx.z;
  k = blockIdx.x * blockDim.x + threadIdx.x;
  i = blockIdx.y * blockDim.y + threadIdx.y;
	D = L.getSize(2);
	/* boundary check for output */
	if (k >= G.getSize(2) || i >= G.getSize(1))	return;
	/* main operation */
	sum = 0;
	for(d=0; d<D; d++) {
		sum += L[b][k][d].ldg() * R[b][i][k][d].ldg();
	}
	G[b][i][k] = sum;
}

void HZAggregate_BackwardA(THCState *state, THCTensor *G_, THCTensor *L_,
							THCTensor *R_)
/*
 * mapping the image pixels based on the lookuptable
 */
{
	/* Check the GPU index */
	HZENCODING_assertSameGPU(state, 3, G_, L_, R_);
	/* Device tensors */
	THCDeviceTensor<real, 3> G = devicetensor<3>(state, G_);
	THCDeviceTensor<real, 3> L = devicetensor<3>(state, L_);
	THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(G.getSize(2)/16+1, G.getSize(1)/16+1, 
							G.getSize(0));
	HZAggregate_BackwardA_kernel<<<blocks, threads, 0, stream>>>(G, L, R);
	THCudaCheck(cudaGetLastError());
}
