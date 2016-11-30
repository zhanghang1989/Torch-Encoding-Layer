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
template <int Dim>
THCDeviceTensor<float, Dim> devicetensor(THCState *state, THCudaTensor *t) {
  if (!t) {
    return THCDeviceTensor<float, Dim>();
  }

  int inDim = THCudaTensor_nDimension(state, t);
  if (inDim == Dim) {
    return toDeviceTensor<float, Dim>(state, t);
  }

  // View in which the last dimensions are collapsed or expanded as needed
  THAssert(THCudaTensor_isContiguous(state, t));
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = t->size[i];
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= t->size[i];
    }
  }
  return THCDeviceTensor<float, Dim>(THCudaTensor_data(state, t), size);
}

