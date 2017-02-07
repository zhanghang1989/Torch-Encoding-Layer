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
static int encoding_(Main_Aggregate_Forward)(lua_State *L)
/*
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 3)
    luaL_error(L,  "Encoding: Incorrect number of arguments.\n");
	THCTensor* E_ = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
	THCTensor* A_  = *(THCTensor**)luaL_checkudata(L, 2, 
												THC_Tensor);
	THCTensor* R_ = *(THCTensor**)luaL_checkudata(L, 3, 
												THC_Tensor);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, E_) != 3 ||
			THCTensor_(nDimension)(state, A_) != 3 ||
			THCTensor_(nDimension)(state, R_) != 4)
		luaL_error(L, "Encoding: incorrect input dims. \n");

	HZAggregate_Forward(state, E_, A_, R_);
	/* C function return number of the outputs */
	return 0;
}

static int encoding_(Main_Aggregate_BackwardA)(lua_State *L)
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
	THCTensor* R_ = *(THCTensor**)luaL_checkudata(L, 3, 
												THC_Tensor);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, G_) != 3 ||
			THCTensor_(nDimension)(state, L_) != 3 ||
			THCTensor_(nDimension)(state, R_) != 4)
		luaL_error(L, "Encoding: incorrect input dims. \n");

	HZAggregate_BackwardA(state, G_, L_, R_);
	/* C function return number of the outputs */
	return 0;
}
