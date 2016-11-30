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
static int encoding_(Main_Encoding_ForwardF)(lua_State *L)
/*
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 3)
    luaL_error(L,  "Encoding: Incorrect number of arguments.\n");
	THCTensor* F_ = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
	THCTensor* C_  = *(THCTensor**)luaL_checkudata(L, 2, 
												THC_Tensor);
	THCTensor* s_  = *(THCTensor**)luaL_checkudata(L, 2, 
												THC_Tensor);
	THCTensor* X_ = *(THCTensor**)luaL_checkudata(L, 3, 
												THC_Tensor);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, F_) != 3 ||
			THCTensor_(nDimension)(state, C_) != 2 ||
			THCTensor_(nDimension)(state, s_) != 1 ||
			THCTensor_(nDimension)(state, X_) != 3)
		luaL_error(L, "Encoding: incorrect input dims. \n");

	HZEncoding_ForwardF(state, F_, C_, s_, X_);
	/* C function return number of the outputs */
	return 0;
}

