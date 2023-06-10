#ifndef __MINER_H__
#define __MINER_H__

extern "C" {

#include <stdbool.h>
#include <inttypes.h>
#include <sys/time.h>
#include <pthread.h>
#include <jansson.h>
#include <curl/curl.h>

#ifdef STDC_HEADERS
# include <stdlib.h>
# include <stddef.h>
#else
# ifdef HAVE_STDLIB_H
#  include <stdlib.h>
# endif
#endif

#ifdef __INTELLISENSE__
/* should be in stdint.h but... */
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef __int16 int8_t;
typedef unsigned __int16 uint8_t;

typedef unsigned __int32 time_t;
typedef char *  va_list;
#endif

# define _ALIGN(x) __align__(x)
typedef unsigned char uchar;

#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifndef max
# define max(a, b)  ((a) > (b) ? (a) : (b))
#endif
#ifndef min
# define min(a, b)  ((a) < (b) ? (a) : (b))
#endif

#ifndef UINT32_MAX
/* for gcc 4.4 */
#define UINT32_MAX UINT_MAX
#endif

static inline bool is_windows(void) {
#ifdef WIN32
        return 1;
#else
        return 0;
#endif
}

static inline bool is_x64(void) {
#if defined(__x86_64__) || defined(_WIN64) || defined(__aarch64__)
	return 1;
#elif defined(__amd64__) || defined(__amd64) || defined(_M_X64) || defined(_M_IA64)
	return 1;
#else
	return 0;
#endif
}

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define WANT_BUILTIN_BSWAP
#else
#define bswap_32(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
                   | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#define bswap_64(x) (((uint64_t) bswap_32((uint32_t)((x) & 0xffffffffu)) << 32) \
                   | (uint64_t) bswap_32((uint32_t)((x) >> 32)))
#endif

static inline uint32_t swab32(uint32_t v)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap32(v);
#else
	return bswap_32(v);
#endif
}

static inline uint64_t swab64(uint64_t v)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap64(v);
#else
	return bswap_64(v);
#endif
}

static inline void swab256(void *dest_p, const void *src_p)
{
	uint32_t *dest = (uint32_t *) dest_p;
	const uint32_t *src = (const uint32_t *) src_p;

	dest[0] = swab32(src[7]);
	dest[1] = swab32(src[6]);
	dest[2] = swab32(src[5]);
	dest[3] = swab32(src[4]);
	dest[4] = swab32(src[3]);
	dest[5] = swab32(src[2]);
	dest[6] = swab32(src[1]);
	dest[7] = swab32(src[0]);
}

#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif

#if !HAVE_DECL_BE32DEC
static inline uint32_t be32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
	    ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}
#endif

#if !HAVE_DECL_LE32DEC
static inline uint32_t le32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
	    ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}
#endif

#if !HAVE_DECL_BE32ENC
static inline void be32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[3] = x & 0xff;
	p[2] = (x >> 8) & 0xff;
	p[1] = (x >> 16) & 0xff;
	p[0] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_LE32ENC
static inline void le32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
	p[2] = (x >> 16) & 0xff;
	p[3] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_BE16DEC
static inline uint16_t be16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[1]) + ((uint16_t)(p[0]) << 8));
}
#endif

#if !HAVE_DECL_BE16ENC
static inline void be16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[1] = x & 0xff;
	p[0] = (x >> 8) & 0xff;
}
#endif

#if !HAVE_DECL_LE16DEC
static inline uint16_t le16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[0]) + ((uint16_t)(p[1]) << 8));
}
#endif

#if !HAVE_DECL_LE16ENC
static inline void le16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
}
#endif

/* used for struct work */
void *aligned_calloc(int size);
void aligned_free(void *ptr);

#define MAX_NONCES 2
struct work {
	uint32_t data[48];
	uint32_t target[8];
	uint32_t maxvote;

	char job_id[128];
	size_t xnonce2_len;
	uchar xnonce2[32];

	union {
		uint32_t u32[2];
		uint64_t u64[1];
	} noncerange;

	uint8_t pooln;
	uint8_t valid_nonces;
	uint8_t submit_nonce_id;
	uint8_t job_nonce_id;

	uint32_t nonces[MAX_NONCES];
	double sharediff[MAX_NONCES];
	double shareratio[MAX_NONCES];
	double targetdiff;

	uint32_t height;

	uint32_t scanned_from;
	uint32_t scanned_to;

	// zec solution
	uint8_t extra[1388];
};

// blockbench define

static int opt_n_threads;
static bool opt_quite;

// #define RANDOM_GEN
static uint32_t* global_memory;

extern long device_sm;
extern short device_mpcount;
extern int device_bfactor;
extern int cryptonight_fork;

// sph/sha2

void sha256_init(uint32_t *state);
void sha256_transform(uint32_t *state, const uint32_t *block, int swap);
void sha256d(unsigned char *hash, const unsigned char *data, int len);

// util
extern bool fulltest(const uint32_t *hash, const uint32_t *target);
extern void gpulog(int prio, const char *fmt, ...);
extern int cuda_available_memory();


// sha256

extern int scanhash_sha256d(struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_sha256t(struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_sha256q(struct work* work, uint32_t max_nonce, unsigned long *hashes_done);

extern void free_sha256d();
extern void free_sha256t();
extern void free_sha256q();

// JHA

extern int scanhash_jackpot(struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_jha(struct work *work, uint32_t max_nonce, unsigned long *hashes_done);

extern void free_jackpot();
extern void free_jha();

// quark

extern int scanhash_nist5(struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern void free_nist5();

// x11

extern int scanhash_x11(struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern void free_x11();

// crypto

extern int scanhash_cryptonight(struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int variant);
extern void free_cryptonight();

}

extern void test_scanhash(struct work* work,
		void (*scanhash)(uint32_t threads, uint32_t startNounce, uint32_t *d_hash),
		void (*init)(uint32_t threads));

extern void test_scanhash(struct work* work,
		void (*scanhash)(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order),
		void (*init)(uint32_t threads));

extern void test_scanhash(struct work* work,
		void (*scanhash)(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order),
		int (*init)(uint32_t threads));

#endif /* __MINER_H__ */
