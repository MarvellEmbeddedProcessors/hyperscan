/* scalar.h -  */

/* Copyright (c) 2019, MARVELL International LTD. */

/*
modification history
--------------------
01a,03jun16,rnp written
01a,16aug16,abd neon intrin added
*/

#ifndef __SCALAR_H__
#define __SCALAR_H__

#include "intrin.h"
typedef __m128i m128;

// int printf(const char * fmt, ...);

#ifdef USE_NEON

/* Disable for non-inline */
#define USE_INLINE

#ifdef USE_INLINE
#define RTN_DEFS
#endif

#endif

#ifdef RTN_DEFS
#ifdef RTN_DECLS
m128 _mm_shuffle_epi8(m128 a, m128 b);
m128 _mm_shuffle_epi32(m128 a, int b);
m128 _mm_alignr_epi8(m128 a, m128 b, int offset);
int _mm_movemask_epi8(m128 a);
int _mm_movemask_ps(m128 a);
m128 _mm_cmpeq_epi8(m128 a, m128 b);
m128 _mm_cmpeq_epi32(m128 a, m128 b);
m128 _mm_castsi128_ps(m128 a);
m128 _mm_cvtsi32_si128(int a);
int _mm_cvtsi128_si32(m128 a);
long _mm_cvtsi128_si64(m128 a);
m128 _mm_unpacklo_epi8(m128 a, m128 b);
m128 _mm_srli_epi64(m128 a, int count);
m128 _mm_srli_si128(m128 a, int imm);
m128 _mm_slli_si128(m128 a, int imm);
m128 _mm_slli_epi64(m128 a, int count);
m128 _mm_and_si128(m128 a, m128 b);
m128 _mm_andnot_si128(m128 a, m128 b);
m128 _mm_or_si128(m128 a, m128 b);
m128 _mm_xor_si128(m128 a, m128 b);
m128 _mm_packs_epi16(m128 a, m128 b);
m128 _mm_packs_epi32(m128 a, m128 b);
m128 _mm_load_si128(const m128 *ptr);
m128 _mm_loadu_si128(const m128 *ptr);
void _mm_storeu_si128(m128 *ptr, m128 a);
m128 _mm_setzero_si128(void);
m128 _mm_set1_epi8(char b);
m128 _mm_set1_epi64x(long long a);
m128 _mm_set_epi64x(long long a, long long b);
#endif /* RTN_DECLS */

#ifdef USE_NEON

/* Use SW alternatives for some neon intrinsics */
//#define SW_ALT

#ifdef USE_INLINE
#define really_inline inline __attribute__((always_inline, unused))
#else
#define static
#undef really_inline
#define really_inline
#endif

#define int8x16_to_8x8x2(v)                                                    \
    (int8x8x2_t) {                                                             \
        { vget_low_s8(v), vget_high_s8(v) }                                    \
    }

static really_inline m128 _mm_set1_epi32(int i) {
    return (m128)vdupq_n_s32((int32_t)i);
}

static really_inline m128 _mm_set_epi64(long long q1, long long q0) {
    m128 r =
        (m128)vcombine_s64((int64x1_t)vdup_n_s64(q0),
                           (int64x1_t)vdup_n_s64(q1)); // low,high /* fixed */
    return r;
}

static really_inline m128 _mm_max_epu8(m128 a, m128 b) {
    return (m128)vmaxq_u8((uint8x16_t)a, (uint8x16_t)b);
}

static really_inline m128 _mm_min_epu8(m128 a, m128 b) {
    return (m128)vminq_u8((uint8x16_t)a, (uint8x16_t)b);
}

static really_inline m128 _mm_adds_epu8(m128 a, m128 b) {
    return (m128)vqaddq_u8((uint8x16_t)a, (uint8x16_t)b);
}

static really_inline m128 _mm_sub_epi8(m128 a, m128 b) {
    return (m128)vsubq_s8((int8x16_t)a, (int8x16_t)b);
}

static really_inline m128 _mm_shuffle_epi8(m128 a, m128 b) {
#ifndef SW_ALT

    int8x16_t btranslated = vandq_s8((int8x16_t)b, vdupq_n_s8(0x8f));
    return (m128)vqtbl1q_s8((int8x16_t)a, (uint8x16_t)btranslated);

#define DUMMY_MASK(mask)                                                       \
    vorrq_s8((int8x16_t)VTST(mask),                                            \
             vdupq_n_s8(0x0F)) // remove upper nibble from mask
#define VTST(mask)                                                             \
    vtstq_s8((int8x16_t)mask, vdupq_n_s8(0x80)) //..if msb is clear
    m128 modified_mask = vandq_s8((int8x16_t)b, DUMMY_MASK(b));

    return (m128)vcombine_s8(
        vtbl2_s8(int8x16_to_8x8x2(a), vget_low_s8(modified_mask)),
        vtbl2_s8(int8x16_to_8x8x2(a), vget_high_s8(modified_mask)));
#else
    m128 r = {0, 0, 0, 0};
    char *_a = (char *)&a;
    char *_b = (char *)&b;
    char *_r = (char *)&r;
    unsigned int i = 0;
    for (i = 0; i < 16; i++) {
        if (_b[i] & 0x80) {
            _r[i] = 0;
        } else {
            _r[i] = _a[_b[i] & 0x0F];
        }
    }
    return r;
#endif
}

static really_inline m128 _mm_shuffle_epi32(m128 a, int b) /* not used in DP */
{
    /* SW implementation */
    m128 r = {0};
    char *_a = (char *)&a;
    char *_r = (char *)&r;
    _r[0] = _a[(b >> 0) & 0x3];
    _r[1] = _a[(b >> 2) & 0x3];
    _r[2] = _a[(b >> 4) & 0x3];
    _r[3] = _a[(b >> 6) & 0x3];
    return r;
}

static really_inline m128 _mm_alignr_epi8(m128 a, m128 b, int offset) {
#ifndef SW_ALT
    return (m128)vextq_s8((int8x16_t)b, (int8x16_t)a,
                          offset); /* caution reverse */
#else
    m128 r = vdupq_n_s8(0);
    char *_a = (char *)&a;
    char *_b = (char *)&b;
    char *_r = (char *)&r;
    unsigned int i = 0;
    if (offset < (int)sizeof(b))
        for (i = 0; i + offset < sizeof(r); i++)
            _r[i] = _b[i + offset];
    if (offset >= (int)sizeof(b))
        offset -= sizeof(b);
    for (; i + offset < sizeof(r); i++)
        _r[i] = _a[i + offset];
    return r;
#endif
}

static really_inline int _mm_movemask_epi8(m128 a) {
#ifdef SW_ALT
    int r = 0;
    char *_a = (char *)&a;
    unsigned int i = 0;
    for (i = 0; i < 16; i++) {
        r |= ((_a[i] >> 7) & 0x1) << i;
    }
    return r;
#else
#if 1
    uint8x16_t mask_shift = {
        (unsigned char)-7, (unsigned char)-6, (unsigned char)-5,
        (unsigned char)-4, (unsigned char)-3, (unsigned char)-2,
        (unsigned char)-1, (unsigned char)0,  (unsigned char)-7,
        (unsigned char)-6, (unsigned char)-5, (unsigned char)-4,
        (unsigned char)-3, (unsigned char)-2, (unsigned char)-1,
        (unsigned char)0};
    uint8x16_t x = (uint8x16_t)a;
    x = vandq_u8(x, vdupq_n_u8(0x80));
    x = vshlq_u8(x, vreinterpretq_s8_u8(mask_shift));
    uint8_t addl = vaddv_u8(vget_low_u8(x));
    uint8_t addh = vaddv_u8(vget_high_u8(x));
    return addh << 8 | addl;
#elif 1
    uint8x16_t mask_shift = {
        (unsigned char)-7, (unsigned char)-6, (unsigned char)-5,
        (unsigned char)-4, (unsigned char)-3, (unsigned char)-2,
        (unsigned char)-1, (unsigned char)0,  (unsigned char)-7,
        (unsigned char)-6, (unsigned char)-5, (unsigned char)-4,
        (unsigned char)-3, (unsigned char)-2, (unsigned char)-1,
        (unsigned char)0};
    uint8x16_t mask_and = vdupq_n_u8(0x80);
    uint8x16_t x = (uint8x16_t)a;
    x = vandq_u8(x, mask_and);
    x = vshlq_u8(x, vreinterpretq_s8_u8(mask_shift));

    x = vpaddq_u8(x, x);
    x = vpaddq_u8(x, x);
    x = vpaddq_u8(x, x);

    // return vgetq_lane_u8 (x, 0) | (vgetq_lane_u8 (x, 1) << 8);
    return vgetq_lane_u16((uint16x8_t)x, 0);
#else

    /* do for q,4 add instead of 6 (test perf)*/
    uint8x16_t input = (uint8x16_t)a;
    const int8_t __attribute__((aligned(16)))
    xr[8] = {-7, -6, -5, -4, -3, -2, -1, 0};
    uint8x8_t mask_and = vdup_n_u8(0x80);
    int8x8_t mask_shift = vld1_s8(xr);

    uint8x8_t lo = vget_low_u8(input);
    uint8x8_t hi = vget_high_u8(input);

    lo = vand_u8(lo, mask_and);
    lo = vshl_u8(lo, mask_shift);

    hi = vand_u8(hi, mask_and);
    hi = vshl_u8(hi, mask_shift);

    lo = vpadd_u8(lo, lo);
    lo = vpadd_u8(lo, lo);
    lo = vpadd_u8(lo, lo);

    hi = vpadd_u8(hi, hi);
    hi = vpadd_u8(hi, hi);
    hi = vpadd_u8(hi, hi);

    return ((hi[0] << 8) | (lo[0] & 0xFF));
#endif
#endif
}

static really_inline int _mm_movemask_ps(m128 a) /* not used in DP */
{
#ifdef SW_ALT
    int r = 0;
    int *_a = (int *)&a;
    unsigned int i = 0;
    for (i = 0; i < 4; i++)
        r |= ((_a[i] >> 31) & 0x1) << i;
    return r;
#else
    static const uint32x4_t movemask = {1, 2, 4, 8};
    static const uint32x4_t highbit = {0x80000000, 0x80000000, 0x80000000,
                                       0x80000000};
    uint32x4_t t0 = vreinterpretq_u32_f32((float32x4_t)a);
    uint32x4_t t1 = vtstq_u32(t0, highbit);
    uint32x4_t t2 = vandq_u32(t1, movemask);
    uint32x2_t t3 = vorr_u32(vget_low_u32(t2), vget_high_u32(t2));
    return vget_lane_u32(t3, 0) | vget_lane_u32(t3, 1);
#endif
}

static really_inline m128 _mm_cmpeq_epi8(m128 a, m128 b) {
#ifndef SW_ALT /* verify */
    return (m128)vceqq_s8((int8x16_t)a, (int8x16_t)b);
#else
    m128 r = (m128)vdupq_n_s8(0);
    char *_a = (char *)&a;
    char *_b = (char *)&b;
    char *_r = (char *)&r;
    unsigned int i = 0;
    for (i = 0; i < sizeof(r); i++) {
        _r[i] = (_a[i] == _b[i]) ? 0xff : 0x0;
    }
    return r;
#endif
}

static really_inline m128 _mm_cmpeq_epi32(m128 a, m128 b) {
    return (m128)vceqq_s32((int32x4_t)a,
                           (int32x4_t)b); /* verify correctness, not used? */
}

static really_inline m128 _mm_castsi128_ps(m128 a) /* not used in DP*/
{
    return *(const m128 *)&a;
}

static really_inline m128 _mm_cvtsi32_si128(int a) /* not used in DP*/
{
    m128 r = vdupq_n_s8(0);
    return (m128)vsetq_lane_s32(a, (int32x4_t)r, 0);
}

static really_inline int _mm_cvtsi128_si32(m128 a) /* not used in DP */
{
    return vgetq_lane_s32((int32x4_t)a, 0);
}

static really_inline long _mm_cvtsi128_si64(m128 a) {
    return vgetq_lane_s64((int64x2_t)a, 0);
}

static really_inline m128 _mm_unpacklo_epi8(m128 a, m128 b) {
    int8x8_t a1 = (int8x8_t)vget_low_s16((int16x8_t)a);
    int8x8_t b1 = (int8x8_t)vget_low_s16((int16x8_t)b);

    int8x8x2_t result = vzip_s8(a1, b1);

    return (__m128i)vcombine_s8(result.val[0], result.val[1]);
}

static really_inline m128 _mm_srli_epi64(m128 a, int count) {
    return (m128)vshrq_n_s64((int64x2_t)a, count);
}
static really_inline m128 _mm_srli_si128(m128 a, int imm) {
#ifndef SW_ALT
    return (m128)vextq_s8((int8x16_t)a, vdupq_n_s8(0), (imm)); /* fixed*/
#else
    m128 r = (m128)vdupq_n_s8(0);
    char *_a = (char *)&a;
    char *_r = (char *)&r;
    unsigned int i = 0;
    int offset = imm;
    for (i = 0; i + offset < 16; i++) {
        _r[i] = _a[i + offset];
    }
    return r;
#endif
}

static really_inline m128 _mm_slli_si128(m128 a, int imm) {
#ifndef SW_ALT
    if (imm) {
        return (m128)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 16 - (imm));
    } else {
        return a;
    }
#else
    m128 r = (m128)vdupq_n_s8(0);
    char *_a = (char *)&a;
    char *_r = (char *)&r;
    unsigned int i = 0;
    int offset = imm;
    for (i = 0; i + offset < 16; i++) {
        _r[i + offset] = _a[i];
    }
    return r;
#endif
}

static really_inline m128 _mm_slli_epi64(m128 a, int count) {
    return (m128)vshlq_n_s64((int64x2_t)a, count);
}

static really_inline m128 _mm_and_si128(m128 a, m128 b) {
    return (m128)vandq_s8((int8x16_t)a, (int8x16_t)b);
}

static really_inline m128 _mm_andnot_si128(m128 a, m128 b) {
    return (m128)vandq_s8(vmvnq_s8((int8x16_t)a), (int8x16_t)b); /*TODO*/
}

static really_inline m128 _mm_or_si128(m128 a, m128 b) {
    return (m128)vorrq_s8(a, b);
}

static really_inline m128 _mm_xor_si128(m128 a, m128 b) {
    return (m128)veorq_s8(a, b);
}

static really_inline m128 _mm_packs_epi16(m128 a, m128 b) /* not used */
{
    return (m128)vqmovn_high_s16(vqmovn_s16((int16x8_t)a), (int16x8_t)b);
}

static really_inline m128 _mm_packs_epi32(m128 a, m128 b) /* not used */
{
    return (m128)vqmovn_high_s32(vqmovn_s32((int32x4_t)a), (int32x4_t)b);
}

static really_inline m128 _mm_load_si128(const m128 *ptr) {
#ifdef SW_ALT
    // m128 r = vdupq_n_s8(0);
    uint16x8_t r = {0};
    memcpy(&r, ptr, sizeof(r));
    return (m128)r;
#else
    return (m128)vld1q_s8((const int8_t *)ptr);
#endif
}

static really_inline m128 _mm_loadu_si128(const m128 *ptr) {
    return (m128)vld1q_s8((const int8_t *)ptr);
}

static really_inline void _mm_storeu_si128(m128 *ptr, m128 a) {
    vst1q_s8((int8_t *)ptr, a); /* unaligned? */
}

static really_inline m128 _mm_setzero_si128(void) {
    return (m128)vdupq_n_s8(0);
}

static really_inline m128 _mm_set1_epi8(char b) {
    return (m128)vdupq_n_s8((int8_t)b);
}

static really_inline m128 _mm_set1_epi64x(long long a) /* not used */
{
    m128 r = (m128)vdupq_n_s64(a);
    return r;
}

static really_inline m128 _mm_set_epi64x(long long a, long long b) /*not used */
{
    m128 r =
        (m128)vcombine_s64((int64x1_t)vdup_n_s64(b),
                           (int64x1_t)vdup_n_s64(a)); // low,high /* fixed*/
    return r;
}

#elif defined(USE_SCALAR)

#define really_inline inline __attribute__((always_inline, unused))
static really_inline void hexdump(const char *p, int s, const char *msg) {
    int _i = 0;
    int _len = s;
    const unsigned char *_data = (const unsigned char *)p;
    printf("%s", msg);
    for (_i = 0; _i < _len; _i++)
        printf("%02x%s", _data[_i],
               (_i + 1) % 4 ? "" : ((_i + 1) % 16 ? " " : ""));
    // printf("\n");
}

#define static
#undef really_inline
#define really_inline

static really_inline m128 _mm_shuffle_epi8(m128 a, m128 b) {
    /*
        for (i = 0; i < 16; i++){
            if (b[i] & 0x80){
                r[i] =  0;
            }
            else {
                r[i] = a[b[i] & 0x0F];
            }
        }
    */
    m128 r = {0, 0};
    char *_a = (char *)&a;
    char *_b = (char *)&b;
    char *_r = (char *)&r;
    unsigned int i = 0;
    for (i = 0; i < 16; i++) {
        if (_b[i] & 0x80) {
            _r[i] = 0;
        } else {
            _r[i] = _a[_b[i] & 0x0F];
        }
    }
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_shuffle_epi8(");
    // hexdump((char *)&b, sizeof(b), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline m128 _mm_shuffle_epi32(m128 a, int b) {
    /*
        DEST[31:0] = (SRC >> (ORDER[1:0] * 32))[31:0];
        DEST[63:32] = (SRC >> (ORDER[3:2] * 32))[31:0];
        DEST[95:64] = (SRC >> (ORDER[5:4] * 32))[31:0];
        DEST[127:96] = (SRC >> (ORDER[7:6] * 32))[31:0];
    */
    m128 r = {0, 0};
    char *_a = (char *)&a;
    char *_r = (char *)&r;
    _r[0] = _a[(b >> 0) & 0x3];
    _r[1] = _a[(b >> 2) & 0x3];
    _r[2] = _a[(b >> 4) & 0x3];
    _r[3] = _a[(b >> 6) & 0x3];
    return r;
}
static really_inline m128 _mm_alignr_epi8(m128 a, m128 b, int offset) {
    /*
        t1[255:128] = a;
        t1[127:0] = b;
        t1[255:0] = t1[255:0] >> (8 * offset); // unsigned shift
        r[127:0] = t1[127:0];
    */
    m128 r = {0, 0};
    char *_a = (char *)&a;
    char *_b = (char *)&b;
    char *_r = (char *)&r;
    unsigned int i = 0;
    if (offset < (int)sizeof(b))
        for (i = 0; i + offset < sizeof(r); i++)
            _r[i] = _b[i + offset];
    if (offset >= (int)sizeof(b))
        offset -= sizeof(b);
    for (; i + offset < sizeof(r); i++)
        _r[i] = _a[i + offset];
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_alignr_epi8(");
    // hexdump((char *)&b, sizeof(b), ", ");
    // hexdump((char *)&offset, sizeof(offset), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}

static really_inline int _mm_movemask_epi8(m128 a) {
    /* Creates a 16-bit mask from the most significant bits of the 16
     * signed or unsigned 8-bit integers in a and zero extends the upper
     * bits.

        R0 = a15[7] << 15 | a14[7] << 14 | ... a1[7] << 1 | a0[7]
     */
    int r = 0;
    char *_a = (char *)&a;
    unsigned int i = 0;
    for (i = 0; i < 16; i++) {
        r |= ((_a[i] >> 7) & 0x1) << i;
    }
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_movemask_epi8(");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline int _mm_movemask_ps(m128 a) {
    /*
    Creates a 4-bit mask from the most significant bits of the four,
    single-precision floating-point values of A, as follows:

    int i = move_mask(F32vec4 A)
    i := sign(a3)<<3 | sign(a2)<<2 | sign(a1)<<1 | sign(a0)<<0
    */
    int r = 0;
    int *_a = (int *)&a;
    unsigned int i = 0;
    for (i = 0; i < 4; i++)
        r |= ((_a[i] >> 31) & 0x1) << i;
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_movemask_ps(");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline m128 _mm_cmpeq_epi8(m128 a, m128 b) {
    /*
    Compares the 16 signed or unsigned 8-bit integers in a and the 16 signed or
    unsigned 8-bit integers in b for equality.

    R0             |R1                 |...  |R15
    (a0 == b0) ? 0xff : 0x0 |(a1 == b1) ? 0xff : 0x0     |...  |(a15 == b15) ?
    0xff : 0x0
    */
    m128 r = {0, 0};
    char *_a = (char *)&a;
    char *_b = (char *)&b;
    char *_r = (char *)&r;
    unsigned int i = 0;
    for (i = 0; i < sizeof(r); i++) {
        _r[i] = (_a[i] == _b[i]) ? 0xff : 0x0;
    }
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_cmpeq_epi8(");
    // hexdump((char *)&b, sizeof(b), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline int int_saturate(int x) {
    x = (x == -1) ? -1 : 0x0;
    return x;
}
static really_inline m128 _mm_cmpeq_epi32(m128 a, m128 b) {
    /*
    Compares the four signed or unsigned 32-bit integers in a and the four
    signed or unsigned 32-bit integers in b for equality.

    R0                 |R1                 |R2                 |R3
    (a0 == b0) ? 0xffffffff : 0x0     |(a1 == b1) ? 0xffffffff : 0x0     |(a2 ==
    b2) ? 0xffffffff : 0x0     |(a3 == b3) ? 0xffffffff : 0x0
    */
    m128 r = {0, 0};
    int *_a = (int *)&a;
    int *_b = (int *)&b;
    int *_r = (int *)&r;
    unsigned int i = 0;
    for (i = 0; i < sizeof(r) / sizeof(*_r); i++) {
        _r[i] = (_a[i] == _b[i]) ? 0xffffffff : 0x0;
    }
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_cmpeq_epi32(");
    // hexdump((char *)&b, sizeof(b), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline m128 _mm_castsi128_ps(m128 a) { return a; }
static really_inline m128 _mm_cvtsi32_si128(int a) {
    /* Moves 32-bit integer a to the least significant 32 bits of an
     * __m128i object. Zeroes the upper 96 bits of the __m128i object.
     */
    m128 r = {0, 0};
    r.one = a;
    return r;
}
static really_inline int _mm_cvtsi128_si32(m128 a) {
    /* Moves the least significant 32 bits of a to a 32-bit integer. */
    int t = 0;
    t = (int)a.one;
    return t;
}
static really_inline long _mm_cvtsi128_si64(m128 a) {
    /* Moves the least significant 64 bits of a to a 64-bit integer. */
    return a.one;
}
static really_inline m128 _mm_unpacklo_epi8(m128 a, m128 b) {
    /*
    Interleaves the lower eight signed or unsigned 8-bit integers in a with the
    lower eight signed or unsigned 8-bit integers in b.

    R0 R1 R2 R3 ...  R14 R15
    a0 b0 a1 b1 ...  a7  b7
    */
    m128 r = {0, 0};
    char *_a = (char *)&a;
    char *_b = (char *)&b;
    char *_r = (char *)&r;
    unsigned int i = 0;
    for (i = 0; i < 16; i++) {
        _r[i] = (i & 1) ? _b[i >> 1] : _a[i >> 1];
    }
    return r;
}

static really_inline m128 _mm_srli_epi64(m128 a, int count) {
    /*
    Shifts the two signed or unsigned 64-bit integers in a right by count bits
    while shifting in zeros.

    R0         | R1
    srl(a0, count)  | srl(a1, count)
    */
    m128 r = {0, 0};
    r.one = ((unsigned long)a.one) >> count;
    r.two = ((unsigned long)a.two) >> count;
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_srli_epi64(");
    // hexdump((char *)&count, sizeof(count), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline m128 _mm_srli_si128(m128 a, int imm) {
    /*
    Shifts the 128-bit value in a right by imm bytes while shifting in zeros.
    imm must be an immediate.

    R = srl(a, imm*8)
    */
    m128 r = {0, 0};
    char *_a = (char *)&a;
    char *_r = (char *)&r;
    unsigned int i = 0;
    int offset = imm;
    for (i = 0; i + offset < 16; i++) {
        _r[i] = _a[i + offset];
    }
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_srli_si128(");
    // hexdump((char *)&imm, sizeof(imm), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline m128 _mm_slli_si128(m128 a, int imm) {
    /*
    Shifts the 128-bit value in a left by imm bytes while shifting in zeros. imm
    must be an immediate.

    R = a << (imm * 8)
    */
    m128 r = {0, 0};
    char *_a = (char *)&a;
    char *_r = (char *)&r;
    unsigned int i = 0;
    int offset = imm;
    for (i = 0; i + offset < 16; i++) {
        _r[i + offset] = _a[i];
    }
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_slli_epi128(");
    // hexdump((char *)&imm, sizeof(imm), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline m128 _mm_slli_epi64(m128 a, int count) {
    /*
    Shifts the two signed or unsigned 64-bit integers in a left by count bits
    while shifting in zeros.

    R0         | R1
    a0 << count     | a1 << count
    */
    m128 r = {0, 0};
    r.one = a.one << count;
    r.two = a.two << count;
    return r;
}

static really_inline m128 _mm_and_si128(m128 a, m128 b) {
    /*
    Computes the bitwise AND of the 128-bit value in a and the 128-bit value in
    b.

    R0 = a & b
    */
    m128 r = {0, 0};
    r.one = a.one & b.one;
    r.two = a.two & b.two;
    return r;
}
static really_inline m128 _mm_andnot_si128(m128 a, m128 b) {
    /*
    Computes the bitwise AND of the 128-bit value in b and the bitwise NOT of
    the 128-bit value in a.

    R0 = (~a) & b
    */
    m128 r = {0, 0};
    r.one = (~a.one) & b.one;
    r.two = (~a.two) & b.two;
    return r;
}
static really_inline m128 _mm_or_si128(m128 a, m128 b) {
    /*
    Computes the bitwise OR of the 128-bit value in a and the 128-bit value in
    b.

    R0 = a | b
    */
    m128 r = {0, 0};
    r.one = a.one | b.one;
    r.two = a.two | b.two;
    return r;
}
static really_inline m128 _mm_xor_si128(m128 a, m128 b) {
    /*
    Computes the bitwise XOR of the 128-bit value in a and the 128-bit value in
    b.

    R0 = a ^ b
    */
    m128 r = {0, 0};
    r.one = a.one ^ b.one;
    r.two = a.two ^ b.two;
    return r;
}

static really_inline m128 _mm_packs_epi16(m128 a, m128 b) {
    /*
    Packs the 16 signed 16-bit integers from a and b into 8-bit integers and
    saturates.

    R0             |...  |R7            |R8             |...  |R15
    Signed Saturate(a0)     |...  |Signed Saturate(a7)     |Signed Saturate(b0)
    |...  |Signed Saturate(b7)
    */
    m128 r = {0, 0};
    short *_a = (short *)&a;
    short *_b = (short *)&b;
    signed char *_r = (signed char *)&r;
    unsigned int i = 0;
    for (i = 0; i < 8; i++) {
        _r[i] = (_a[i] > (signed char)_a[i]) ? 0x7f : _a[i];
        _r[i] = (_a[i] < _r[i]) ? 0x80 : _r[i];
    }
    for (i = 0; i < 8; i++) {
        _r[8 + i] = (_b[i] > (signed char)_b[i]) ? 0x7f : _b[i];
        _r[8 + i] = (_b[i] < _r[8 + i]) ? 0x80 : _r[8 + i];
    }
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_packs_epi16(");
    // hexdump((char *)&b, sizeof(b), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline m128 _mm_packs_epi32(m128 a, m128 b) {
    /*
    Packs the eight signed 32-bit integers from a and b into signed 16-bit
    integers and saturates.

    R0             |...  |R3             |R4             |...  |R7
    Signed Saturate(a0)     |...  |Signed Saturate(a3)     |Signed Saturate(b0)
    |...  |Signed Saturate(b3)
    */
    m128 r = {0, 0};
    int *_a = (int *)&a;
    int *_b = (int *)&b;
    short *_r = (short *)&r;
    _r[0] = (_a[0] > (short)_a[0]) ? 0x7fff : _a[0];
    _r[0] = (_a[0] < (short)_a[0]) ? 0x8000 : _r[0];
    _r[1] = (_a[1] > (short)_a[1]) ? 0x7fff : _a[1];
    _r[1] = (_a[1] < (short)_a[1]) ? 0x8000 : _r[1];
    _r[2] = (_a[2] > (short)_a[2]) ? 0x7fff : _a[2];
    _r[2] = (_a[2] < (short)_a[2]) ? 0x8000 : _r[2];
    _r[3] = (_a[3] > (short)_a[3]) ? 0x7fff : _a[3];
    _r[3] = (_a[3] < (short)_a[3]) ? 0x8000 : _r[3];

    _r[4] = (_b[0] > (short)_b[0]) ? 0x7fff : _b[0];
    _r[4] = (_b[0] < (short)_b[0]) ? 0x8000 : _r[4];
    _r[5] = (_b[1] > (short)_b[1]) ? 0x7fff : _b[1];
    _r[5] = (_b[1] < (short)_b[1]) ? 0x8000 : _r[5];
    _r[6] = (_b[2] > (short)_b[2]) ? 0x7fff : _b[2];
    _r[6] = (_b[2] < (short)_b[2]) ? 0x8000 : _r[6];
    _r[7] = (_b[3] > (short)_b[3]) ? 0x7fff : _b[3];
    _r[7] = (_b[3] < (short)_b[3]) ? 0x8000 : _r[7];
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((char *)&a, sizeof(a), ") = _mm_packs_epi32(");
    // hexdump((char *)&b, sizeof(b), ", ");
    // hexdump((char *)0, 0, ")\n");
    return r;
}

static really_inline m128 _mm_load_si128(const m128 *ptr) {
    /*
    Loads 128-bit value. Address p must be 16-byte aligned.

    R = *p
    */
    m128 r = {0, 0};
    memcpy(&r, ptr, sizeof(r));
    return r;
}
static really_inline m128 _mm_loadu_si128(const m128 *ptr) {
    /*
    Loads 128-bit value. Address p not need be 16-byte aligned.

    R = *p
    */
    m128 r = {0, 0};
    memcpy(&r, ptr, sizeof(r));
    // hexdump((char *)&r, sizeof(r), "r(");
    // hexdump((const char *)ptr, sizeof(r), ") = _mm_loadu_si128(");
    // hexdump((char *)0, 0, ")\n");
    return r;
}
static really_inline void _mm_storeu_si128(m128 *ptr, m128 a) {
    /*
    Stores 128-bit value. Address p need not be 16-byte aligned.

    *p = a
    */
    memcpy(ptr, &a, sizeof(a));
    return;
}

static really_inline m128 _mm_setzero_si128(void) {
    /* Sets the 128-bit value to zero. */
    m128 r = {0, 0};
    return r;
}
static really_inline m128 _mm_set1_epi8(char b) {
    /*
    Sets the 16 signed 8-bit integer values to b.

    R0     R1     ...      R15
    b     b     b     b
    */
    m128 r = {0, 0};
    memset(&r, b, sizeof(r));
    return r;
}
static really_inline m128 _mm_set1_epi64x(long long a) {
    /*
    Sets the two 64-bit integer values to a.

    R0     R1
    a     a
    */
    m128 r = {0, 0};
    r.one = r.two = a;
    return r;
}
static really_inline m128 _mm_set_epi64x(long long a, long long b) {
    /*
    Sets the two 64-bit integer values.

    R0    R1
    a     b
    */
    m128 r = {0, 0};
    r.one = b;
    r.two = a;
    return r;
}

#endif
#else /* RTN_DEFS */
#ifdef __cplusplus
extern "C" {
#endif
m128 _mm_shuffle_epi8(m128 a, m128 b);
m128 _mm_shuffle_epi32(m128 a, int b);
m128 _mm_alignr_epi8(m128 a, m128 b, int offset);
int _mm_movemask_epi8(m128 a);
int _mm_movemask_ps(m128 a);
m128 _mm_cmpeq_epi8(m128 a, m128 b);
m128 _mm_cmpeq_epi32(m128 a, m128 b);
m128 _mm_castsi128_ps(m128 a);
m128 _mm_cvtsi32_si128(int a);
int _mm_cvtsi128_si32(m128 a);
long _mm_cvtsi128_si64(m128 a);
m128 _mm_unpacklo_epi8(m128 a, m128 b);
m128 _mm_srli_epi64(m128 a, int count);
m128 _mm_srli_si128(m128 a, int imm);
m128 _mm_slli_si128(m128 a, int imm);
m128 _mm_slli_epi64(m128 a, int count);
m128 _mm_and_si128(m128 a, m128 b);
m128 _mm_andnot_si128(m128 a, m128 b);
m128 _mm_or_si128(m128 a, m128 b);
m128 _mm_xor_si128(m128 a, m128 b);
m128 _mm_packs_epi16(m128 a, m128 b);
m128 _mm_packs_epi32(m128 a, m128 b);
m128 _mm_load_si128(const m128 *ptr);
m128 _mm_loadu_si128(const m128 *ptr);
void _mm_storeu_si128(m128 *ptr, m128 a);
m128 _mm_setzero_si128(void);
m128 _mm_set1_epi8(char b);
m128 _mm_set1_epi64x(long long a);
m128 _mm_set_epi64x(long long a, long long b);
#ifdef __cplusplus
}
#endif
#endif

#endif /* __SCALAR_H__ */
