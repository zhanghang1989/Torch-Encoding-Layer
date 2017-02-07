/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Created by: Hang Zhang
 * ECE Department, Rutgers University
 * Email: zhang.hang@rutgers.edu
 * Copyright (c) 2016
 *
 * Free to reuse and distribute this software for research or 
 * non-profit purpose, subject to the following conditions:
 *  1. The code must retain the above copyright notice, this list of
 *     conditions.
 *  2. Original authors' names are not deleted.
 *  3. The authors' names are not used to endorse or promote products
 *      derived from this software 
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
__global__ void HZWeighting_UpdateParams_kernel (
	THCDeviceTensor<real, 2> G,
	THCDeviceTensor<real, 3> L,
	THCDeviceTensor<real, 3> D)
{
  /* declarations of the variables */
  int b, k, i, N;
	real sum;
  /* Get the index and channels */ 
  b = blockIdx.y;
  k = blockIdx.x * blockDim.x + threadIdx.x;
	N = L.getSize(1);
	/* boundary check for output */
	if (k >= G.getSize(1))	return;
	/* main operation */
	sum = 0;
	for(i=0; i<N; i++) {
		sum += L[b][i][k].ldg() * D[b][i][k].ldg();
	}
	G[b][k] = isnan(sum) ? 1e-6 : sum;
}

void HZWeighting_UpdateParams(THCState *state, THCTensor *G_, THCTensor *L_,
							THCTensor *D_)
/*
 * mapping the image pixels based on the lookuptable
 */
{
	/* Check the GPU index */
	HZENCODING_assertSameGPU(state, 3, G_, L_, D_);
	/* Device tensors */
	THCDeviceTensor<real, 2> G = devicetensor<2>(state, G_);
	THCDeviceTensor<real, 3> L = devicetensor<3>(state, L_);
	THCDeviceTensor<real, 3> D = devicetensor<3>(state, D_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16);
	dim3 blocks(G.getSize(1)/16+1,G.getSize(0));
	HZWeighting_UpdateParams_kernel<<<blocks, threads, 0, stream>>>(G, L, D);
	THCudaCheck(cudaGetLastError());
}

__global__ void HZWeighting_BatchRowWeighing_kernel (
	THCDeviceTensor<real, 3> G,
	THCDeviceTensor<real, 2> W,
	THCDeviceTensor<real, 3> L)
{
  /* declarations of the variables */
  int b, k, d;
	real output;
  /* Get the index and channels */ 
  b = blockIdx.z;
  d = blockIdx.x * blockDim.x + threadIdx.x;
  k = blockIdx.y * blockDim.y + threadIdx.y;
	/* boundary check for output */
	if (k >= G.getSize(1) || d >= G.getSize(2))	return;
	/* main operation */
	output = L[b][k][d].ldg() * W[b][k].ldg();
	G[b][k][d] = isnan(output) ? 1e-16: output;
}

void HZWeighting_BatchRowWeighting(THCState *state, THCTensor *G_, THCTensor *W_,
							THCTensor *L_)
/*
 * mapping the image pixels based on the lookuptable
 */
{
	/* Check the GPU index */
	HZENCODING_assertSameGPU(state, 3, G_, W_, L_);
	/* Device tensors */
	THCDeviceTensor<real, 3> G = devicetensor<3>(state, G_);
	THCDeviceTensor<real, 2> W = devicetensor<2>(state, W_);
	THCDeviceTensor<real, 3> L = devicetensor<3>(state, L_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16,16);
	dim3 blocks(G.getSize(2)/16+1, G.getSize(1)/16+1, G.getSize(0));
	HZWeighting_BatchRowWeighing_kernel<<<blocks, threads, 0, stream>>>(G, W, L);
	THCudaCheck(cudaGetLastError());
}

