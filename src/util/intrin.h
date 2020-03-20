/* intrin.h -  */

/* Copyright (c) 2019, MARVELL International LTD. */

/*
modification history
--------------------
01a,03jun16,rnp written
02a,20aug16,abd neon
*/

#ifndef __INTRIN_H__
#define __INTRIN_H__

#ifdef USE_SCALAR
typedef struct {
	long one;
	long two;
} __m128i;
#endif

#ifdef USE_NEON
#include <arm_neon.h>
typedef int8x16_t __m128i;
#endif

#endif /* __INTRIN_H__ */