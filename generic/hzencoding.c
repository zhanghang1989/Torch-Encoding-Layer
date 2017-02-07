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
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/hzencoding.c"
#else

/* load the implementation detail */
#include "aggregate.c"
#include "weighting.c"
#include "encoding.c"

/* register the functions */
static const struct luaL_Reg encoding_(Aggregate) [] = 
{
	{"Forward",     encoding_(Main_Aggregate_Forward)},
	{"BackwardA",   encoding_(Main_Aggregate_BackwardA)},
	/* end */
	{NULL, NULL}
};

static const struct luaL_Reg encoding_(Weighting) [] = 
{
	{"UpdateParams", encoding_(Main_Weighting_UpdateParams)},
	{"BatchRowScale",     encoding_(Main_Weighting_BatchRow)},
	/* end */
	{NULL, NULL}
};

static const struct luaL_Reg encoding_(Encoding) [] = 
{
	{"ForwardF",     encoding_(Main_Encoding_ForwardF)},
	/* end */
	{NULL, NULL}
};

DLL_EXPORT int luaopen_libencoding(lua_State *L) {
	lua_newtable(L);
	lua_pushvalue(L, -1);
	lua_setglobal(L, "HZENCODING");

	lua_newtable(L);
	luaT_setfuncs(L, encoding_(Aggregate), 0);
	lua_setfield(L, -2, "Aggregate");

	lua_newtable(L);
	luaT_setfuncs(L, encoding_(Weighting), 0);
	lua_setfield(L, -2, "Weighting");

	lua_newtable(L);
	luaT_setfuncs(L, encoding_(Encoding), 0);
	lua_setfield(L, -2, "Encoding");

	return 1;
}

#endif // THC_GENERIC_FILE
