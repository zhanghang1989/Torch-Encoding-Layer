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
__global__ void HZEncoding_ForwardF_kernel (
	THCDeviceTensor<real, 3> F,
	THCDeviceTensor<real, 2> C,
	THCDeviceTensor<real, 1> s,
	THCDeviceTensor<real, 3> X)
{
  /* declarations of the variables */
  int b, k, i, d, D;
	real sum;
  /* Get the index and channels */ 
  b = blockIdx.z;
  k = blockIdx.x * blockDim.x + threadIdx.x;
  i = blockIdx.y * blockDim.y + threadIdx.y;
	D = C.getSize(1);
	/* boundary check for output */
	if (k >= F.getSize(2) || i >= F.getSize(1))	return;
	/* main operation */
	sum = 0;
	for (d=0; d<D; d++) {
		sum += (X[b][i][d].ldg() - C[k][d].ldg()) * (X[b][i][d].ldg() - C[k][d].ldg());
	}
	F[b][i][k] = exp(-s[k] * sum);
}

void HZEncoding_ForwardF(THCState *state, THCTensor *F_, THCTensor *C_,
							THCTensor *s_, THCTensor *X_)
/*
 * mapping the image pixels based on the lookuptable
 */
{
	/* Check the GPU index */
	HZENCODING_assertSameGPU(state, 3, F_, C_, s_, X_);
	/* Device tensors */
	THCDeviceTensor<real, 3> F = devicetensor<3>(state, F_);
	THCDeviceTensor<real, 2> C = devicetensor<2>(state, C_);
	THCDeviceTensor<real, 1> s = devicetensor<1>(state, s_);
	THCDeviceTensor<real, 3> X = devicetensor<3>(state, X_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(F.getSize(2)/16+1, F.getSize(1)/16+1, F.getSize(0));
	HZEncoding_ForwardF_kernel<<<blocks, threads, 0, stream>>>(F, C, s, X);
	THCudaCheck(cudaGetLastError());
}

