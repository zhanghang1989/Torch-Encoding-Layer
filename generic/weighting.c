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
static int encoding_(Main_Weighting_UpdateParams)(lua_State *L)
/*
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 3)
    luaL_error(L,  "Encoding: Incorrect number of arguments.\n");
	THCTensor* G_ = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
	THCTensor* L_  = *(THCTensor**)luaL_checkudata(L, 2, 
												THC_Tensor);
	THCTensor* D_ = *(THCTensor**)luaL_checkudata(L, 3, 
												THC_Tensor);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, G_) != 2 ||
			THCTensor_(nDimension)(state, L_) != 3 ||
			THCTensor_(nDimension)(state, D_) != 3)
		luaL_error(L, "Encoding: incorrect input dims. \n");

	HZWeighting_UpdateParams(state, G_, L_, D_);
	/* C function return number of the outputs */
	return 0;
}

static int encoding_(Main_Weighting_BatchRow)(lua_State *L)
/*
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 3)
    luaL_error(L,  "Encoding: Incorrect number of arguments.\n");
	THCTensor* G_ = *(THCTensor**)luaL_checkudata(L, 3, 
												THC_Tensor);
	THCTensor* W_  = *(THCTensor**)luaL_checkudata(L, 2, 
												THC_Tensor);
	THCTensor* L_ = *(THCTensor**)luaL_checkudata(L, 3, 
												THC_Tensor);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, G_) != 3 ||
			THCTensor_(nDimension)(state, W_) != 2 ||
			THCTensor_(nDimension)(state, L_) != 3)
		luaL_error(L, "Encoding: incorrect input dims. \n");

	HZWeighting_BatchRowWeighting(state, G_, W_, L_);
	/* C function return number of the outputs */
	return 0;
}

