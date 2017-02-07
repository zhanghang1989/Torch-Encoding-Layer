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
#include "TH.h"
#include "luaT.h"
#include <THC/THC.h>
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "common.h"

/* extern function in cutorch */
struct THCState;
#ifdef __cplusplus
extern "C" struct THCState* cutorch_getstate(lua_State* L);
#else
extern struct THCState* cutorch_getstate(lua_State* L);
#endif

#define torch_(NAME)     TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor     TH_CONCAT_STRING_3(torch., Real, Tensor)
#define THCTensor        TH_CONCAT_3(TH,CReal,Tensor)
#define THCTensor_(NAME) TH_CONCAT_4(TH,CReal,Tensor_,NAME)
#define THC_Tensor       TH_CONCAT_STRING_3(torch., CReal, Tensor)
#define encoding_(NAME)    TH_CONCAT_3(encoding_, Real, NAME)

#ifdef __cplusplus
extern "C" {
#endif

#include "lib/HZENCODING.c"
#include "THCGenerateFloatType.h"

#include "generic/hzencoding.c"
#include "THCGenerateFloatType.h"

#ifdef __cplusplus
}
#endif
