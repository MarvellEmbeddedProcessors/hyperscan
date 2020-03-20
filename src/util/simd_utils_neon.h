/*
 * Copyright (c) 2015-2017, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of Intel Corporation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** \file
 * \brief SIMD types and primitive operations.
 */

#ifndef SIMD_UTILS_NEON
#define SIMD_UTILS_NEON

#include <arm_neon.h>

#if 0
typedef int8x16_t __m128i;
typedef __m128i m128;
#else
/* Just for now; until the last of the _mm* usage below are removed. */
#include "scalar.h"
#endif


static really_inline m128 ones128(void) {
    return (m128)vdupq_n_s8 ((int8_t)-1);
}

static really_inline m128 zeroes128(void) {
    return (m128)vdupq_n_u8(0);
}

/** \brief Bitwise not for m128*/
static really_inline m128 not128(m128 a) {
    return (m128)vmvnq_s8((int8x16_t)a);
}

/** \brief Return 1 if a and b are different otherwise 0 */
static really_inline int diff128(m128 a, m128 b) {
    m128 t = (m128)vceqq_s8((int8x16_t)a, (int8x16_t)b);
    uint8_t t1 = vaddvq_u8((uint8x16_t)t);
    return t1 != (uint8_t)-16;
}

static really_inline int isnonzero128(m128 a) {
    return !!diff128(a, zeroes128());
}

/* Create a bitmask of which "32bit" lanes have their upper bit set.  */
static really_inline int upperbitmask128_32(m128 a)
{
    static const uint32x4_t movemask = { 1, 2, 4, 8 };
    static const uint32x4_t highbit = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
    uint32x4_t t1 = vtstq_u32((uint32x4_t)a, highbit);
    uint32x4_t t2 = vandq_u32(t1, movemask);
    /* Each of the 32bit lanes contain unique bits set so adding across the lanes is the same as oring across the langes. */
    return vaddvq_u32(t2);
}

/* Create a bitmask of which "32bit" lanes are the same. */
static really_inline int bitmaskequals128_32(m128 a, m128 b)
{
    static const uint32x4_t movemask = { 1, 2, 4, 8 };
    uint32x4_t t1 = vceqq_s32((int32x4_t)a, (int32x4_t)b);
    uint32x4_t t2 = vandq_u32(t1, movemask);
    /* Each of the 32bit lanes contain unique bits set so adding across the lanes is the same as oring across the langes. */
    return vaddvq_u32(t2);
}

/**
 * "Rich" version of diff128(). Takes two vectors a and b and returns a 4-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich128(m128 a, m128 b) {
    return ~(bitmaskequals128_32(a, b)) & 0xf;
}

/* Create a bitmask of which "64bit" lanes are the same. */
static really_inline int bitmaskequals128_64(m128 a, m128 b)
{
    static const uint64x2_t movemask = { 1, 4 };
    uint64x2_t t1 = vceqq_s64((int64x2_t)a, (int64x2_t)b);
    uint64x2_t t2 = vandq_u64(t1, movemask);
    /* Each of the 32bit lanes contain unique bits set so adding across the lanes is the same as oring across the langes. */
    return vaddvq_u64(t2);
}
/**
 * "Rich" version of diff128(), 64-bit variant. Takes two vectors a and b and
 * returns a 4-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_128(m128 a, m128 b) {
    return ~(bitmaskequals128_64(a, b)) & 0x5;
}

static really_really_inline
m128 lshift64_m128(m128 a, unsigned b) {
    return (m128)vshlq_n_s64((int64x2_t)a, b);
}

#define rshift64_m128(a, b) _mm_srli_epi64((a), (b))
#define eq128(a, b)      _mm_cmpeq_epi8((a), (b))
#define movemask128(a)  ((u32)_mm_movemask_epi8((a)))

static really_inline m128 set16x8(u8 c) {
    return _mm_set1_epi8(c);
}

static really_inline m128 set4x32(u32 c) {
    return _mm_set1_epi32(c);
}

static really_inline u32 movd(const m128 in) {
    return _mm_cvtsi128_si32(in);
}

static really_inline u64a movq(const m128 in) {
#if defined(ARCH_X86_64)
    return _mm_cvtsi128_si64(in);
#elif defined(USE_SCALAR) || defined(USE_NEON)
    return _mm_cvtsi128_si64(in);
#else // 32-bit - this is horrific
    u32 lo = movd(in);
    u32 hi = movd(_mm_srli_epi64(in, 32));
    return (u64a)hi << 32 | lo;
#endif
}

/* another form of movq */
static really_inline
m128 load_m128_from_u64a(const u64a *p) {
    return _mm_set_epi64x(0LL, *p);
}

#define rshiftbyte_m128(a, count_immed) _mm_srli_si128(a, count_immed)
#define lshiftbyte_m128(a, count_immed) _mm_slli_si128(a, count_immed)

#if defined(HAVE_SSE41)
#define extract32from128(a, imm) _mm_extract_epi32(a, imm)
#define extract64from128(a, imm) _mm_extract_epi64(a, imm)
#else
#define extract32from128(a, imm) movd(_mm_srli_si128(a, imm << 2))
#define extract64from128(a, imm) movq(_mm_srli_si128(a, imm << 3))
#endif

// TODO: this entire file needs restructuring - this carveout is awful
#define extractlow64from256(a) movq(a.lo)
#define extractlow32from256(a) movd(a.lo)
#if defined(HAVE_SSE41)
#define extract32from256(a, imm) _mm_extract_epi32((imm >> 2) ? a.hi : a.lo, imm % 4)
#define extract64from256(a, imm) _mm_extract_epi64((imm >> 1) ? a.hi : a.lo, imm % 2)
#else
#define extract32from256(a, imm) movd(_mm_srli_si128((imm >> 2) ? a.hi : a.lo, (imm % 4) * 4))
#define extract64from256(a, imm) movq(_mm_srli_si128((imm >> 1) ? a.hi : a.lo, (imm % 2) * 8))
#endif

static really_inline m128 and128(m128 a, m128 b) {
    return (m128)vandq_s8((int8x16_t)a, (int8x16_t)b);
}

static really_inline m128 xor128(m128 a, m128 b) {
    return (m128)veorq_s8((int8x16_t)a, (int8x16_t)b);
}

static really_inline m128 or128(m128 a, m128 b) {
    return (m128)vorrq_s8(a, b);
}

static really_inline m128 andnot128(m128 a, m128 b) {
    return (m128)vandq_s8(vmvnq_s8((int8x16_t)a),(int8x16_t)b);
}

// aligned load
static really_inline m128 load128(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    ptr = assume_aligned(ptr, 16);
    return _mm_load_si128((const m128 *)ptr);
}

// aligned store
static really_inline void store128(void *ptr, m128 a) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    ptr = assume_aligned(ptr, 16);
    *(m128 *)ptr = a;
}

// unaligned load
static really_inline m128 loadu128(const void *ptr) {
    return _mm_loadu_si128((const m128 *)ptr);
}

// unaligned store
static really_inline void storeu128(void *ptr, m128 a) {
    _mm_storeu_si128 ((m128 *)ptr, a);
}

// packed unaligned store of first N bytes
static really_inline
void storebytes128(void *ptr, m128 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline
m128 loadbytes128(const void *ptr, unsigned int n) {
    m128 a = zeroes128();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

static really_inline
m128 mask1bit128(unsigned int n) {
    /* A load will most likely be faster than doing the calcuation. */
    assert(n < sizeof(m128) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu128(&simd_onebit_masks[mask_idx]);
}

// switches on bit N in the given vector.
static really_inline
void setbit128(m128 *ptr, unsigned int n) {
    *ptr = or128(mask1bit128(n), *ptr);
}

// switches off bit N in the given vector.
static really_inline
void clearbit128(m128 *ptr, unsigned int n) {
    *ptr = andnot128(mask1bit128(n), *ptr);
}

// tests bit N in the given vector.
static really_inline
char testbit128(m128 val, unsigned int n) {
    const m128 mask = mask1bit128(n);
    m128 t = and128(mask, val);
    /* Special case of isnonzero128 as we know that mask1bit128 produces 1 bitset so the add will either be 1 or 0. */
    uint8_t t1 = vaddvq_u8((uint8x16_t)t);
    return t1 != 0;
}

// offset must be an immediate
#define palignr(r, l, offset) _mm_alignr_epi8(r, l, offset)

static really_inline
m128 pshufb_m128(m128 a, m128 b) {
    /* On Intel, if bit 0x80 is set, then result is zero, otherwise which the lane it is &0xf.
       In NEON, if >=16, then the result is zero, otherwise it is that lane.
       btranslated is the version that is converted from Intel to NEON.  */
    int8x16_t btranslated = vandq_s8((int8x16_t)b,vdupq_n_s8(0x8f));
    return (m128)vqtbl1q_s8((int8x16_t)a, (uint8x16_t)btranslated);
}

static really_inline
m256 pshufb_m256(m256 a, m256 b) {
    m256 rv;
    /* NOTE pshufb does not cross between the 128bit boundaries, this is unlike tbl2 and vpermb (the AVX512BMI version) */
    rv.lo = pshufb_m128(a.lo, b.lo);
    rv.hi = pshufb_m128(a.hi, b.hi);
    return rv;
}

static really_inline
m128 variable_byte_shift_m128(m128 in, s32 amount) {
    assert(amount >= -16 && amount <= 16);
    m128 shift_mask = loadu128(vbs_mask_data + 16 - amount);
    /* vbs_mask_data already has been corrected so that the zeros ones
       are 0x80 and the normal ones are &0xf already. so just do the TBL
       instruction directly. */
    return (m128)vqtbl1q_s8((int8x16_t)in, (uint8x16_t)shift_mask);
}

static really_inline
m128 max_u8_m128(m128 a, m128 b) {
    return _mm_max_epu8(a, b);
}

static really_inline
m128 min_u8_m128(m128 a, m128 b) {
    return _mm_min_epu8(a, b);
}

static really_inline
m128 sadd_u8_m128(m128 a, m128 b) {
    return _mm_adds_epu8(a, b);
}

static really_inline
m128 sub_u8_m128(m128 a, m128 b) {
    return _mm_sub_epi8(a, b);
}

static really_inline
m128 set64x2(u64a hi, u64a lo) {
    return _mm_set_epi64x(hi, lo);
}

/****
 **** 256-bit Primitives
 ****/

static really_really_inline
m256 lshift64_m256(m256 a, int b) {
    m256 rv = a;
    rv.lo = lshift64_m128(rv.lo, b);
    rv.hi = lshift64_m128(rv.hi, b);
    return rv;
}

static really_inline
m256 rshift64_m256(m256 a, int b) {
    m256 rv = a;
    rv.lo = rshift64_m128(rv.lo, b);
    rv.hi = rshift64_m128(rv.hi, b);
    return rv;
}
static really_inline
m256 set32x8(u32 in) {
    m256 rv;
    rv.lo = set16x8((u8) in);
    rv.hi = rv.lo;
    return rv;
}

static really_inline
m256 eq256(m256 a, m256 b) {
    m256 rv;
    rv.lo = eq128(a.lo, b.lo);
    rv.hi = eq128(a.hi, b.hi);
    return rv;
}

static really_inline
u32 movemask256(m256 a) {
    u32 lo_mask = movemask128(a.lo);
    u32 hi_mask = movemask128(a.hi);
    return lo_mask | (hi_mask << 16);
}

static really_inline
m256 set2x128(m128 a) {
    m256 rv = {a, a};
    return rv;
}

static really_inline m256 zeroes256(void) {
    m256 rv = {zeroes128(), zeroes128()};
    return rv;
}

static really_inline m256 ones256(void) {
    m256 rv = {ones128(), ones128()};
    return rv;
}

static really_inline m256 and256(m256 a, m256 b) {
    m256 rv;
    rv.lo = and128(a.lo, b.lo);
    rv.hi = and128(a.hi, b.hi);
    return rv;
}

static really_inline m256 or256(m256 a, m256 b) {
    m256 rv;
    rv.lo = or128(a.lo, b.lo);
    rv.hi = or128(a.hi, b.hi);
    return rv;
}

static really_inline m256 xor256(m256 a, m256 b) {
    m256 rv;
    rv.lo = xor128(a.lo, b.lo);
    rv.hi = xor128(a.hi, b.hi);
    return rv;
}

static really_inline m256 not256(m256 a) {
    m256 rv;
    rv.lo = not128(a.lo);
    rv.hi = not128(a.hi);
    return rv;
}

static really_inline m256 andnot256(m256 a, m256 b) {
    m256 rv;
    rv.lo = andnot128(a.lo, b.lo);
    rv.hi = andnot128(a.hi, b.hi);
    return rv;
}

static really_inline int diff256(m256 a, m256 b) {
    return diff128(a.lo, b.lo) || diff128(a.hi, b.hi);
}

static really_inline int isnonzero256(m256 a) {
    return isnonzero128(or128(a.lo, a.hi));
}

/**
 * "Rich" version of diff256(). Takes two vectors a and b and returns an 8-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich256(m256 a, m256 b) {
    m128 z = zeroes128();
    a.lo = _mm_cmpeq_epi32(a.lo, b.lo);
    a.hi = _mm_cmpeq_epi32(a.hi, b.hi);
    m128 packed = _mm_packs_epi16(_mm_packs_epi32(a.lo, a.hi), z);
    return ~(_mm_movemask_epi8(packed)) & 0xff;
}

/**
 * "Rich" version of diff256(), 64-bit variant. Takes two vectors a and b and
 * returns an 8-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_256(m256 a, m256 b) {
    u32 d = diffrich256(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline m256 load256(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    m256 rv = { load128(ptr), load128((const char *)ptr + 16) };
    return rv;
}

// aligned load  of 128-bit value to low and high part of 256-bit value
static really_inline m256 load2x128(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    m256 rv;
    rv.hi = rv.lo = load128(ptr);
    return rv;
}

static really_inline m256 loadu2x128(const void *ptr) {
    return set2x128(loadu128(ptr));
}

// aligned store
static really_inline void store256(void *ptr, m256 a) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    ptr = assume_aligned(ptr, 16);
    *(m256 *)ptr = a;
}

// unaligned load
static really_inline m256 loadu256(const void *ptr) {
    m256 rv = { loadu128(ptr), loadu128((const char *)ptr + 16) };
    return rv;
}

// unaligned store
static really_inline void storeu256(void *ptr, m256 a) {
    storeu128(ptr, a.lo);
    storeu128((char *)ptr + 16, a.hi);
}

// packed unaligned store of first N bytes
static really_inline
void storebytes256(void *ptr, m256 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline
m256 loadbytes256(const void *ptr, unsigned int n) {
    m256 a = zeroes256();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

static really_inline
m256 mask1bit256(unsigned int n) {
    assert(n < sizeof(m256) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu256(&simd_onebit_masks[mask_idx]);
}

static really_inline
m256 set64x4(u64a hi_1, u64a hi_0, u64a lo_1, u64a lo_0) {
    m256 rv;
    rv.hi = set64x2(hi_1, hi_0);
    rv.lo = set64x2(lo_1, lo_0);
    return rv;
}

// switches on bit N in the given vector.
static really_inline
void setbit256(m256 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else {
        sub = &ptr->hi;
        n -= 128;
    }
    setbit128(sub, n);
}

// switches off bit N in the given vector.
static really_inline
void clearbit256(m256 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else {
        sub = &ptr->hi;
        n -= 128;
    }
    clearbit128(sub, n);
}

// tests bit N in the given vector.
static really_inline
char testbit256(m256 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo;
    } else {
        sub = val.hi;
        n -= 128;
    }
    return testbit128(sub, n);
}

static really_really_inline
m128 movdq_hi(m256 x) {
    return x.hi;
}

static really_really_inline
m128 movdq_lo(m256 x) {
    return x.lo;
}

static really_inline
m256 combine2x128(m128 hi, m128 lo) {
    m256 rv = {lo, hi};
    return rv;
}

/****
 **** 384-bit Primitives
 ****/

static really_inline m384 and384(m384 a, m384 b) {
    m384 rv;
    rv.lo = and128(a.lo, b.lo);
    rv.mid = and128(a.mid, b.mid);
    rv.hi = and128(a.hi, b.hi);
    return rv;
}

static really_inline m384 or384(m384 a, m384 b) {
    m384 rv;
    rv.lo = or128(a.lo, b.lo);
    rv.mid = or128(a.mid, b.mid);
    rv.hi = or128(a.hi, b.hi);
    return rv;
}

static really_inline m384 xor384(m384 a, m384 b) {
    m384 rv;
    rv.lo = xor128(a.lo, b.lo);
    rv.mid = xor128(a.mid, b.mid);
    rv.hi = xor128(a.hi, b.hi);
    return rv;
}
static really_inline m384 not384(m384 a) {
    m384 rv;
    rv.lo = not128(a.lo);
    rv.mid = not128(a.mid);
    rv.hi = not128(a.hi);
    return rv;
}
static really_inline m384 andnot384(m384 a, m384 b) {
    m384 rv;
    rv.lo = andnot128(a.lo, b.lo);
    rv.mid = andnot128(a.mid, b.mid);
    rv.hi = andnot128(a.hi, b.hi);
    return rv;
}

static really_really_inline
m384 lshift64_m384(m384 a, unsigned b) {
    m384 rv;
    rv.lo = lshift64_m128(a.lo, b);
    rv.mid = lshift64_m128(a.mid, b);
    rv.hi = lshift64_m128(a.hi, b);
    return rv;
}

static really_inline m384 zeroes384(void) {
    m384 rv = {zeroes128(), zeroes128(), zeroes128()};
    return rv;
}

static really_inline m384 ones384(void) {
    m384 rv = {ones128(), ones128(), ones128()};
    return rv;
}

static really_inline int diff384(m384 a, m384 b) {
    return diff128(a.lo, b.lo) || diff128(a.mid, b.mid) || diff128(a.hi, b.hi);
}

static really_inline int isnonzero384(m384 a) {
    return isnonzero128(or128(or128(a.lo, a.mid), a.hi));
}

/**
 * "Rich" version of diff384(). Takes two vectors a and b and returns a 12-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich384(m384 a, m384 b) {
    m128 z = zeroes128();
    a.lo = _mm_cmpeq_epi32(a.lo, b.lo);
    a.mid = _mm_cmpeq_epi32(a.mid, b.mid);
    a.hi = _mm_cmpeq_epi32(a.hi, b.hi);
    m128 packed = _mm_packs_epi16(_mm_packs_epi32(a.lo, a.mid),
                                  _mm_packs_epi32(a.hi, z));
    return ~(_mm_movemask_epi8(packed)) & 0xfff;
}

/**
 * "Rich" version of diff384(), 64-bit variant. Takes two vectors a and b and
 * returns a 12-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_384(m384 a, m384 b) {
    u32 d = diffrich384(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline m384 load384(const void *ptr) {
    assert(ISALIGNED_16(ptr));
    m384 rv = { load128(ptr), load128((const char *)ptr + 16),
                load128((const char *)ptr + 32) };
    return rv;
}

// aligned store
static really_inline void store384(void *ptr, m384 a) {
    assert(ISALIGNED_16(ptr));
    ptr = assume_aligned(ptr, 16);
    *(m384 *)ptr = a;
}

// unaligned load
static really_inline m384 loadu384(const void *ptr) {
    m384 rv = { loadu128(ptr), loadu128((const char *)ptr + 16),
                loadu128((const char *)ptr + 32)};
    return rv;
}

// packed unaligned store of first N bytes
static really_inline
void storebytes384(void *ptr, m384 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline
m384 loadbytes384(const void *ptr, unsigned int n) {
    m384 a = zeroes384();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

// switches on bit N in the given vector.
static really_inline
void setbit384(m384 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else if (n < 256) {
        sub = &ptr->mid;
    } else {
        sub = &ptr->hi;
    }
    setbit128(sub, n % 128);
}

// switches off bit N in the given vector.
static really_inline
void clearbit384(m384 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else if (n < 256) {
        sub = &ptr->mid;
    } else {
        sub = &ptr->hi;
    }
    clearbit128(sub, n % 128);
}

// tests bit N in the given vector.
static really_inline
char testbit384(m384 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo;
    } else if (n < 256) {
        sub = val.mid;
    } else {
        sub = val.hi;
    }
    return testbit128(sub, n % 128);
}

/****
 **** 512-bit Primitives
 ****/

#define eq512mask(a, b) _mm512_cmpeq_epi8_mask((a), (b))
#define masked_eq512mask(k, a, b) _mm512_mask_cmpeq_epi8_mask((k), (a), (b))

static really_inline
m512 zeroes512(void) {
    m512 rv = {zeroes256(), zeroes256()};
    return rv;
}

static really_inline
m512 ones512(void) {
    m512 rv = {ones256(), ones256()};
    return rv;
}

static really_inline
m512 and512(m512 a, m512 b) {
    m512 rv;
    rv.lo = and256(a.lo, b.lo);
    rv.hi = and256(a.hi, b.hi);
    return rv;
}

static really_inline
m512 or512(m512 a, m512 b) {
    m512 rv;
    rv.lo = or256(a.lo, b.lo);
    rv.hi = or256(a.hi, b.hi);
    return rv;
}

static really_inline
m512 xor512(m512 a, m512 b) {
    m512 rv;
    rv.lo = xor256(a.lo, b.lo);
    rv.hi = xor256(a.hi, b.hi);
    return rv;
}

static really_inline
m512 not512(m512 a) {
    m512 rv;
    rv.lo = not256(a.lo);
    rv.hi = not256(a.hi);
    return rv;
}

static really_inline
m512 andnot512(m512 a, m512 b) {
    m512 rv;
    rv.lo = andnot256(a.lo, b.lo);
    rv.hi = andnot256(a.hi, b.hi);
    return rv;
}

static really_really_inline
m512 lshift64_m512(m512 a, unsigned b) {
    m512 rv;
    rv.lo = lshift64_m256(a.lo, b);
    rv.hi = lshift64_m256(a.hi, b);
    return rv;
}

#if !defined(_MM_CMPINT_NE)
#define _MM_CMPINT_NE 0x4
#endif

static really_inline
int diff512(m512 a, m512 b) {
    return diff256(a.lo, b.lo) || diff256(a.hi, b.hi);
}

static really_inline
int isnonzero512(m512 a) {
    m128 x = or128(a.lo.lo, a.lo.hi);
    m128 y = or128(a.hi.lo, a.hi.hi);
    return isnonzero128(or128(x, y));
}

/**
 * "Rich" version of diff512(). Takes two vectors a and b and returns a 16-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline
u32 diffrich512(m512 a, m512 b) {
    a.lo.lo = _mm_cmpeq_epi32(a.lo.lo, b.lo.lo);
    a.lo.hi = _mm_cmpeq_epi32(a.lo.hi, b.lo.hi);
    a.hi.lo = _mm_cmpeq_epi32(a.hi.lo, b.hi.lo);
    a.hi.hi = _mm_cmpeq_epi32(a.hi.hi, b.hi.hi);
    m128 packed = _mm_packs_epi16(_mm_packs_epi32(a.lo.lo, a.lo.hi),
                                  _mm_packs_epi32(a.hi.lo, a.hi.hi));
    return ~(_mm_movemask_epi8(packed)) & 0xffff;
}

/**
 * "Rich" version of diffrich(), 64-bit variant. Takes two vectors a and b and
 * returns a 16-bit mask indicating which 64-bit words contain differences.
 */
static really_inline
u32 diffrich64_512(m512 a, m512 b) {
    //TODO: cmp_epi64?
    u32 d = diffrich512(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline
m512 load512(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    m512 rv = { load256(ptr), load256((const char *)ptr + 32) };
    return rv;
}

// aligned store
static really_inline
void store512(void *ptr, m512 a) {
    assert(ISALIGNED_N(ptr, alignof(m512)));
    ptr = assume_aligned(ptr, 16);
    *(m512 *)ptr = a;
}

// unaligned load
static really_inline
m512 loadu512(const void *ptr) {
    m512 rv = { loadu256(ptr), loadu256((const char *)ptr + 32) };
    return rv;
}

// packed unaligned store of first N bytes
static really_inline
void storebytes512(void *ptr, m512 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline
m512 loadbytes512(const void *ptr, unsigned int n) {
    m512 a = zeroes512();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

static really_inline
m512 mask1bit512(unsigned int n) {
    assert(n < sizeof(m512) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu512(&simd_onebit_masks[mask_idx]);
}

// switches on bit N in the given vector.
static really_inline
void setbit512(m512 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo.lo;
    } else if (n < 256) {
        sub = &ptr->lo.hi;
    } else if (n < 384) {
        sub = &ptr->hi.lo;
    } else {
        sub = &ptr->hi.hi;
    }
    setbit128(sub, n % 128);
}

// switches off bit N in the given vector.
static really_inline
void clearbit512(m512 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo.lo;
    } else if (n < 256) {
        sub = &ptr->lo.hi;
    } else if (n < 384) {
        sub = &ptr->hi.lo;
    } else {
        sub = &ptr->hi.hi;
    }
    clearbit128(sub, n % 128);
}

// tests bit N in the given vector.
static really_inline
char testbit512(m512 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo.lo;
    } else if (n < 256) {
        sub = val.lo.hi;
    } else if (n < 384) {
        sub = val.hi.lo;
    } else {
        sub = val.hi.hi;
    }
    return testbit128(sub, n % 128);
}

#endif
