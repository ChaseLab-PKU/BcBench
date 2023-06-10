// Equihash solver
// Copyright (c) 2016 John Tromp

// Equihash presents the following problem
//
// Fix N, K, such that N is a multiple of K+1
// Let integer n = N/(K+1), and view N-bit words
// as having K+1 "digits" of n bits each
// Fix M = 2^{n+1} N-bit hashes H_0, ... , H_{M-1}
// as outputs of a hash function applied to an (n+1)-bit index
//
// Problem: find a binary tree on 2^K distinct indices,
// for which the exclusive-or of leaf hashes is all 0s
// Additionally, it should satisfy the Wagner conditions:
// 1) for each height i subtree, the exclusive-or
// of its 2^i leaf hashes starts with i*n 0 bits,
// 2) the leftmost leaf of any left subtree is less
// than the leftmost leaf of the corresponding right subtree
//
// The algorithm below solves this by storing trees
// as a directed acyclic graph of K layers
// The n digit bits are split into
// n-RESTBITS bucket bits and RESTBITS leftover bits
// Each layer i, consisting of height i subtrees
// whose xor starts with i*n 0s, is partitioned into
// 2^{n-RESTBITS} buckets according to the next n-RESTBITS
// in the xor
// Within each bucket, trees whose xor match in the
// next RESTBITS bits are combined to produce trees
// in the next layer
// To eliminate trees with duplicated indices,
// we simply test if the last 32 bits of the xor are 0,
// and if so, assume that this is due to index duplication
// In practice this works very well to avoid bucket overflow
// and produces negligible false positives

#include "equi.h"
#include <stdio.h>
#include <pthread.h>
#include <assert.h>

#include "blake2-avx2/blake2bip.h"

#if defined __builtin_bswap32 && defined __LITTLE_ENDIAN
#undef htobe32
#define htobe32(x) __builtin_bswap32(x)
#elif defined __APPLE__
#undef htobe32
#define htobe32(x) OSSwapHostToBigInt32(x)
#endif

// u32 already defined in equi.h
typedef uint16_t u16;
typedef uint64_t u64;

// required for avoiding multio-threading race conflicts
#ifdef ATOMIC
#include <atomic>
typedef std::atomic<u32> au32;
#else
typedef u32 au32;
#endif

#ifndef RESTBITS
#define CANTOR
#define RESTBITS	10
#endif

// 2_log of number of buckets
#define BUCKBITS (DIGITBITS-RESTBITS)

// by default buckets have a capacity of twice their expected size
// but this factor reduced it accordingly
#ifndef SAVEMEM
#if RESTBITS == 4
// can't save memory in such small buckets
#define SAVEMEM 1
#elif RESTBITS >= 8
// an expected size of at least 512 has such relatively small
// standard deviation that we can reduce capacity with negligible discarding
// this value reduces (200,9) memory to under 144MB
// must be under sqrt(2)/2 with -DCANTOR
#define SAVEMEM 9/14
#endif
#endif

static const u32 NBUCKETS = 1<<BUCKBITS;    // number of buckets
static const u32 BUCKMASK = NBUCKETS-1;     // corresponding bucket mask
static const u32 SLOTBITS = RESTBITS+1+1;   // 2_log of number of slots per bucket
static const u32 SLOTRANGE = 1<<SLOTBITS;   // default bucket capacity
static const u32 SLOTMASK = SLOTRANGE-1;    // corresponding SLOTBITS mask
static const u32 SLOTMSB = 1<<(SLOTBITS-1); // most significat bit in SLOTMASK
static const u32 NSLOTS = SLOTRANGE * SAVEMEM; // number of slots per bucket
static const u32 NRESTS = 1<<RESTBITS;      // number of possible values of RESTBITS bits
static const u32 MAXSOLS = 8;               // more than 8 solutions are rare

// tree node identifying its children as two different slots in
// a bucket on previous layer with matching rest bits (x-tra hash)
struct tree {
  // formerly i had these bitfields
  // unsigned bucketid : BUCKBITS;
  // unsigned slotid0  : SLOTBITS;
  // unsigned slotid1 : SLOTBITS;
  // but these were poorly optimized by the compiler
  // so now we do things "manually"
  u32 bid_s0_s1;

#ifdef CANTOR
  static const u32 CANTORBITS = 2*SLOTBITS-2;
  static const u32 CANTORMASK = (1<<CANTORBITS) - 1;
  static const u32 CANTORMAXSQRT = 2 * NSLOTS;
  static const u32 NSLOTPAIRS = (NSLOTS-1) * (NSLOTS+2) / 2;
  static_assert(NSLOTPAIRS <= 1<<CANTORBITS, "cantor throws a fit");
  static_assert(BUCKBITS + CANTORBITS <= 32, "cantor throws a fit");
#else
  static_assert(BUCKBITS + 2 * SLOTBITS <= 32, "cantor throws a fit");
#endif

  // constructor for height 0 trees stores index instead
  tree(const u32 idx) {
    bid_s0_s1 = idx;
  }
  static u32 cantor(u32 s0, u32 s1) {
    return s1*(s1+1)/2 + s0;
  }
  tree(const u32 bid, const u32 s0, const u32 s1) {
// CANTOR saves 2 bits by Cantor pairing
#ifdef CANTOR
    bid_s0_s1 = (bid << CANTORBITS) | cantor(s0,s1);
#else
    bid_s0_s1 = (((bid << SLOTBITS) | s0) << SLOTBITS) | s1;
#endif
  }
  // retrieve hash index from tree(const u32 idx) constructor
  u32 getindex() const {
    return bid_s0_s1;
  }
  // retrieve bucket index
  u32 bucketid() const {
#ifdef CANTOR
    return bid_s0_s1 >> (2*SLOTBITS - 2);
#else
    return bid_s0_s1 >> (2*SLOTBITS);
#endif
  }
  // retrieve first slot index
#ifdef CANTOR
  u32 slotid0(u32 s1) const {
    return (bid_s0_s1 & CANTORMASK) - cantor(0,s1);
  }
#else
  u32 slotid0() const {
    return (bid_s0_s1 >> SLOTBITS) & SLOTMASK;
  }
#endif
  // retrieve second slot index
  u32 slotid1() const {
#ifdef CANTOR
    u32 k, q, sqr = 8*(bid_s0_s1 & CANTORMASK)+1;;
    // this k=sqrt(sqr) computing loop averages 3.4 iterations out of maximum 9
    for (k = CANTORMAXSQRT; (q = sqr/k) < k; k = (k+q)/2) ;
    return (k-1) / 2;
#else
    return bid_s0_s1 & SLOTMASK;
#endif
  }
  bool prob_disjoint(const tree other) const {
#ifdef CANTOR
    if (bucketid() != other.bucketid())
      return true;
    u32 s1 = slotid1(), s0 = slotid0(s1);
    u32 os1 = other.slotid1(), os0 = other.slotid0(os1);
    return s1 != os1 && s0 != os0;
#else
    tree xort(bid_s0_s1 ^ other.bid_s0_s1);
    return xort.bucketid() || (xort.slotid0() && xort.slotid1());
    // next two tests catch much fewer cases and are therefore skipped
    // && slotid0() != other.slotid1() && slotid1() != other.slotid0()
#endif
  }
};

// each bucket slot occupies a variable number of hash/tree units,
// all but the last of which hold the xor over all leaf hashes,
// or what's left of it after stripping the initial i*n 0s
// the last unit holds the tree node itself
// the hash is sometimes accessed 32 bits at a time (word)
// and sometimes 8 bits at a time (bytes)
union htunit {
  tree tag;
  u32 word;
  uchar bytes[sizeof(u32)];
};

#define WORDS(bits)	((bits + 31) / 32)
#define HASHWORDS0 WORDS(WN - DIGITBITS + RESTBITS)
#define HASHWORDS1 WORDS(WN - 2*DIGITBITS + RESTBITS)

// A slot is up to HASHWORDS0 hash units followed by a tag
typedef htunit slot0[HASHWORDS0+1];
typedef htunit slot1[HASHWORDS1+1];
// a bucket is NSLOTS treenodes
typedef slot0 bucket0[NSLOTS];
typedef slot1 bucket1[NSLOTS];
// the N-bit hash consists of K+1 n-bit "digits"
// each of which corresponds to a layer of NBUCKETS buckets
typedef bucket0 digit0[NBUCKETS];
typedef bucket1 digit1[NBUCKETS];
typedef au32 bsizes[NBUCKETS];

// The algorithm proceeds in K+1 rounds, one for each digit
// All data is stored in two heaps,
// heap0 of type digit0, and heap1 of type digit1
// The following table shows the layout of these heaps
// in each round, which is an optimized version
// of xenoncat's fixed memory layout, avoiding any waste
// Each line shows only a single slot, which is actually
// replicated NSLOTS * NBUCKETS times
//
//             heap0         heap1
// round  hashes   tree   hashes tree
// 0      A A A A A A 0   . . . . . .
// 1      A A A A A A 0   B B B B B 1
// 2      C C C C C 2 0   B B B B B 1
// 3      C C C C C 2 0   D D D D 3 1
// 4      E E E E 4 2 0   D D D D 3 1
// 5      E E E E 4 2 0   F F F 5 3 1
// 6      G G 6 . 4 2 0   F F F 5 3 1
// 7      G G 6 . 4 2 0   H H 7 5 3 1
// 8      I 8 6 . 4 2 0   H H 7 5 3 1
//
// Round 0 generates hashes and stores them in the buckets
// of heap0 according to the initial n-RESTBITS bits
// These hashes are denoted A above and followed by the
// tree tag denoted 0
// In round 1 we combine each pair of slots in the same bucket
// with matching RESTBITS of digit 0 and store the resulting
// 1-tree in heap1 with its xor hash denoted B
// Upon finishing round 1, the A space is no longer needed,
// and is re-used in round 2 to store both the shorter C hashes,
// and their tree tags denoted 2
// Continuing in this manner, each round reads buckets from one
// heap, and writes buckets in the other heap.
// In the final round K, all pairs leading to 0 xors are identified
// and their leafs recovered through the DAG of tree nodes

// convenience function
u32 min(const u32 a, const u32 b) {
  return a < b ? a : b;
}

// size (in bytes) of hash in round 0 <= r < WK
u32 hashsize(const u32 r) {
  const u32 hashbits = WN - (r+1) * DIGITBITS + RESTBITS;
  return (hashbits + 7) / 8;
}

// convert bytes into words,rounding up
u32 hashwords(u32 bytes) {
  return (bytes + 3) / 4;
}

// manages hash and tree data
struct htalloc {
  bucket0 *heap0;
  bucket1 *heap1;
  u32 alloced;
  htalloc() {
    alloced = 0;
  }
  void alloctrees() {
    static_assert(DIGITBITS >= 16, "needed to ensure hashes shorten by 1 unit every 2 digits");
    heap0 = (bucket0 *)alloc(NBUCKETS, sizeof(bucket0));
    heap1 = (bucket1 *)alloc(NBUCKETS, sizeof(bucket1));
  }
  void dealloctrees() {
    free(heap0);
    free(heap1);
  }
  void *alloc(const u32 n, const u32 sz) {
    void *mem  = calloc(n, sz);
    assert(mem);
    alloced += n * sz;
    return mem;
  }
};

// main solver object, shared between all threads
struct equi {
  blake2b_state blake_ctx; // holds blake2b midstate after call to setheadernounce
  htalloc hta;             // holds allocated heaps
  bsizes *nslots;          // counts number of slots used in buckets
  proof *sols;             // store found solutions here (only first MAXSOLS)
  au32 nsols;              // number of solutions found
  u32 nthreads;
  u32 bfull;               // count number of times bucket can't fit new item
  u32 hfull;               // count number of xor-ed hash with last 32 bits zero
  pthread_barrier_t barry; // used to sync threads
  equi(const u32 n_threads) {
    static_assert(sizeof(htunit) == 4, "");
    static_assert(WK&1, "K assumed odd in candidate() calling indices1()");
    nthreads = n_threads;
    const int err = pthread_barrier_init(&barry, NULL, nthreads);
    assert(!err);
    hta.alloctrees();
    nslots = (bsizes *)hta.alloc(2 * NBUCKETS, sizeof(au32));
    sols   =  (proof *)hta.alloc(MAXSOLS, sizeof(proof));
  }
  ~equi() {
    hta.dealloctrees();
    free(nslots);
    free(sols);
  }
  // prepare blake2b midstate for new run and initialize counters
  void setheadernonce(const char *headernonce, const u32 len) {
    setheader(&blake_ctx, headernonce);
    nsols = bfull = hfull = 0;
  }
  // get heap0 bucket size in threadsafe manner
  u32 getslot0(const u32 bucketi) {
#ifdef ATOMIC
    return std::atomic_fetch_add_explicit(&nslots[0][bucketi], 1U, std::memory_order_relaxed);
#else
    return nslots[0][bucketi]++;
#endif
  }
  // get heap1 bucket size in threadsafe manner
  u32 getslot1(const u32 bucketi) {
#ifdef ATOMIC
    return std::atomic_fetch_add_explicit(&nslots[1][bucketi], 1U, std::memory_order_relaxed);
#else
    return nslots[1][bucketi]++;
#endif
  }
  // get old heap0 bucket size and clear it for next round
  u32 getnslots0(const u32 bid) {
    au32 &nslot = nslots[0][bid];
    const u32 n = min(nslot, NSLOTS);
    nslot = 0;
    return n;
  }
  // get old heap1 bucket size and clear it for next round
  u32 getnslots1(const u32 bid) {
    au32 &nslot = nslots[1][bid];
    const u32 n = min(nslot, NSLOTS);
    nslot = 0;
    return n;
  }
  // recognize most (but not all) remaining dupes while Wagner-ordering the indices
  bool orderindices(u32 *indices, u32 size) {
    if (indices[0] > indices[size]) {
      for (u32 i=0; i < size; i++) {
        const u32 tmp = indices[i];
        indices[i] = indices[size+i];
        indices[size+i] = tmp;
      }
    }
    return false;
  }
  // listindices combines index tree reconstruction with probably dupe test
  bool listindices0(u32 r, const tree t, u32 *indices) {
    if (r == 0) {
      *indices = t.getindex();
      return false;
    }
    const slot1 *buck = hta.heap1[t.bucketid()];
    const u32 size = 1 << --r;
    u32 tagi = hashwords(hashsize(r));
#ifdef CANTOR
    u32 s1 = t.slotid1(), s0 = t.slotid0(s1);
#else
    u32 s1 = t.slotid1(), s0 = t.slotid0();
#endif
    tree t0 = buck[s0][tagi].tag, t1 = buck[s1][tagi].tag;
    return !t0.prob_disjoint(t1)
      || listindices1(r, t0, indices) || listindices1(r, t1, indices+size)
      || orderindices(indices, size) || indices[0] == indices[size];
  }
  // need separate instance for accessing (differently typed) heap1
  bool listindices1(u32 r, const tree t, u32 *indices) {
    const slot0 *buck = hta.heap0[t.bucketid()];
    const u32 size = 1 << --r;
    u32 tagi = hashwords(hashsize(r));
#ifdef CANTOR
    u32 s1 = t.slotid1(), s0 = t.slotid0(s1);
#else
    u32 s1 = t.slotid1(), s0 = t.slotid0();
#endif
    tree t0 = buck[s0][tagi].tag, t1 = buck[s1][tagi].tag;
    return listindices0(r, t0, indices) || listindices0(r, t1, indices+size)
        || orderindices(indices, size) || indices[0] == indices[size];
  }
  // check a candidate that resulted in 0 xor
  // add as solution, with proper subtree ordering, if it has unique indices
  void candidate(const tree t) {
    proof prf;
    // listindices combines index tree reconstruction with probably dupe test
    if (listindices1(WK, t, prf) || duped(prf)) return; // assume WK odd
    // and now we have ourselves a genuine solution
#ifdef ATOMIC
    u32 soli = std::atomic_fetch_add_explicit(&nsols, 1U, std::memory_order_relaxed);
#else
    u32 soli = nsols++;
#endif
    // copy solution into final place
    if (soli < MAXSOLS) memcpy(sols[soli], prf, sizeof(proof));
  }
  // show bucket stats and, if desired, size distribution
  void showbsizes(u32 r) {
    printf(" b%d h%d\n", bfull, hfull);
    bfull = hfull = 0;
#if defined(HIST) || defined(SPARK) || defined(LOGSPARK)
    // group bucket sizes in 64 bins, from empty to full (ignoring SAVEMEM)
    u32 binsizes[65];
    memset(binsizes, 0, 65 * sizeof(u32));
    for (u32 bucketid = 0; bucketid < NBUCKETS; bucketid++) {
      u32 bsize = min(nslots[r&1][bucketid], NSLOTS) >> (SLOTBITS-6);
      binsizes[bsize]++;
    }
    for (u32 i=0; i < 65; i++) {
#ifdef HIST  // exact counts are useful for debugging
      printf(" %d:%d", i, binsizes[i]);
#else
#ifdef SPARK // everybody loves sparklines
      u32 sparks = binsizes[i] / SPARKSCALE;
#else
      u32 sparks = 0;
      for (u32 bs = binsizes[i]; bs; bs >>= 1) sparks++;
      sparks = sparks * 7 / SPARKSCALE;
#endif
      printf("\342\226%c", '\201' + sparks);
#endif
    }
    printf("\n");
#endif
    printf("Digit %d", r+1);
  }

  // thread-local object that precomputes various slot metrics for each round
  // facilitating access to various bits in the variable size slots
  struct htlayout {
    htalloc hta;
    u32 prevhtunits;
    u32 nexthtunits;
    u32 dunits;
    u32 prevbo;
  
    htlayout(equi *eq, u32 r): hta(eq->hta), prevhtunits(0), dunits(0) {
      u32 nexthashbytes = hashsize(r);        // number of bytes occupied by round r hash
      nexthtunits = hashwords(nexthashbytes); // number of 32bit words taken up by those bytes
      prevbo = 0;                  // byte offset for accessing hash form previous round
      if (r) {     // similar measure for previous round
        u32 prevhashbytes = hashsize(r-1);
        prevhtunits = hashwords(prevhashbytes);
        prevbo = prevhtunits * sizeof(htunit) - prevhashbytes; // 0-3
        dunits = prevhtunits - nexthtunits; // number of words by which hash shrinks
      }
    }
    // extract remaining bits in digit slots in same bucket still need to collide on
    u32 getxhash0(const htunit* slot) const {
#if WN == 200 && RESTBITS == 4
      return slot->bytes[prevbo] >> 4;
#elif WN == 200 && RESTBITS == 8
      return (slot->bytes[prevbo] & 0xf) << 4 | slot->bytes[prevbo+1] >> 4;
#elif WN == 200 && RESTBITS == 10
      return (slot->bytes[prevbo] & 0x3f) << 4 | slot->bytes[prevbo+1] >> 4;
#elif WN == 144 && RESTBITS == 4
      return slot->bytes[prevbo] & 0xf;
#else
#error non implemented
#endif
    }
    // similar but accounting for possible change in hashsize modulo 4 bits
    u32 getxhash1(const htunit* slot) const {
#if WN == 200 && RESTBITS == 4
      return slot->bytes[prevbo] & 0xf;
#elif WN == 200 && RESTBITS == 8
      return slot->bytes[prevbo];
#elif WN == 200 && RESTBITS == 10
      return (slot->bytes[prevbo] & 0x3) << 8 | slot->bytes[prevbo+1];
#elif WN == 144 && RESTBITS == 4
      return slot->bytes[prevbo] & 0xf;
#else
#error non implemented
#endif
    }
    // test whether two hashes match in last 32 bits
    bool equal(const htunit *hash0, const htunit *hash1) const {
      return hash0[prevhtunits-1].word == hash1[prevhtunits-1].word;
    }
  };

  // this thread-local object performs in-bucket collisions
  // by linking together slots that have identical rest bits
  // (which is in essense a 2nd stage bucket sort)
  struct collisiondata {
    // the bitmap is an early experiment in a bitmap encoding
    // that works only for at most 64 slots
    // it might as well be obsoleted as it performs worse even in that case
#ifdef XBITMAP
#if NSLOTS > 64
#error cant use XBITMAP with more than 64 slots
#endif
    u64 xhashmap[NRESTS];
    u64 xmap;
#else
    // This maintains NRESTS = 2^RESTBITS lists whose starting slot
    // are in xhashslots[] and where subsequent (next-lower-numbered)
    // slots in each list are found through nextxhashslot[]
    // since 0 is already a valid slot number, use ~0 as nil value
#if RESTBITS <= 6
    typedef uchar xslot;
#else
    typedef u16 xslot;
#endif
    static const xslot xnil = ~0;
    xslot xhashslots[NRESTS];
    xslot nextxhashslot[NSLOTS];
    xslot nextslot;
#endif
    u32 s0;

    void clear() {
#ifdef XBITMAP
      memset(xhashmap, 0, NRESTS * sizeof(u64));
#else
      memset(xhashslots, xnil, NRESTS * sizeof(xslot));
      memset(nextxhashslot, xnil, NSLOTS * sizeof(xslot));
#endif
    }
    void addslot(u32 s1, u32 xh) {
#ifdef XBITMAP
      xmap = xhashmap[xh];
      xhashmap[xh] |= (u64)1 << s1;
      s0 = -1;
#else
      nextslot = xhashslots[xh];
      nextxhashslot[s1] = nextslot;
      xhashslots[xh] = s1;
#endif
    }
    bool nextcollision() const {
#ifdef XBITMAP
      return xmap != 0;
#else
      return nextslot != xnil;
#endif
    }
    u32 slot() {
#ifdef XBITMAP
      const u32 ffs = __builtin_ffsll(xmap);
      s0 += ffs; xmap >>= ffs;
#else
      nextslot = nextxhashslot[s0 = nextslot];
#endif
      return s0;
    }
  };

#ifndef NBLAKES
#define NBLAKES 1
#endif

// number of hashes extracted from NBLAKES blake2b outputs
static const u32 HASHESPERBLOCK = NBLAKES*HASHESPERBLAKE;
// number of blocks of parallel blake2b calls
static const u32 NBLOCKS = (NHASHES+HASHESPERBLOCK-1)/HASHESPERBLOCK;

  void digit0(const u32 id) {
    htlayout htl(this, 0);
    const u32 hashbytes = hashsize(0);
    uchar hashes[NBLAKES * 64];
    blake2b_state state0 = blake_ctx;  // local copy on stack can be copied faster
    for (u32 block = id; block < NBLOCKS; block += nthreads) {
#if NBLAKES == 4
      blake2bx4_final(&state0, hashes, block);
#elif NBLAKES == 8
      blake2bx8_final(&state0, hashes, block);
#elif NBLAKES == 1
      blake2b_state state = state0;  // make another copy since blake2b_final modifies it
      u32 leb = htole32(block);
      blake2b_update(&state, (uchar *)&leb, sizeof(u32));
      blake2b_final(&state, hashes, HASHOUT);
#else
#error not implemented
#endif
      for (u32 i = 0; i<NBLAKES; i++) {
        for (u32 j = 0; j<HASHESPERBLAKE; j++) {
          const uchar *ph = hashes + i * 64 + j * WN/8;
          // figure out bucket for this hash by extracting leading BUCKBITS bits
#if BUCKBITS == 12 && RESTBITS == 8
          const u32 bucketid = ((u32)ph[0] << 4) | ph[1] >> 4;
#elif BUCKBITS == 10 && RESTBITS == 10
          const u32 bucketid = ((u32)ph[0] << 2) | ph[1] >> 6;
#elif BUCKBITS == 16 && RESTBITS == 4
          const u32 bucketid = ((u32)ph[0] << 8) | ph[1];
#elif BUCKBITS == 20 && RESTBITS == 4
          const u32 bucketid = ((((u32)ph[0] << 8) | ph[1]) << 4) | ph[2] >> 4;
#else
#error not implemented
#endif
          // grab next available slot in that bucket
          const u32 slot = getslot0(bucketid);
          if (slot >= NSLOTS) {
            bfull++; // this actually never seems to happen in round 0 due to uniformity
            continue;
          }
          // location for slot's tag
          htunit *s = hta.heap0[bucketid][slot] + htl.nexthtunits;
          // hash should end right before tag
          memcpy(s->bytes-hashbytes, ph+WN/8-hashbytes, hashbytes);
          // round 0 tags store hash-generating index
          s->tag = tree((block * NBLAKES + i) * HASHESPERBLAKE + j);
        }
      }
    }
  }
  
  void digitodd(const u32 r, const u32 id) {
    htlayout htl(this, r);
    collisiondata cd;
    // threads process buckets in round-robin fashion
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear(); // could have made this the constructor, and declare here
      slot0 *buck = htl.hta.heap0[bucketid]; // point to first slot of this bucket
      u32 bsize   = getnslots0(bucketid);    // grab and reset bucket size
      for (u32 s1 = 0; s1 < bsize; s1++) {   // loop over slots
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htl.getxhash0(slot1));// identify list of previous colliding slots 
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          if (htl.equal(slot0, slot1)) {     // expect difference in last 32 bits unless duped
            hfull++;                         // record discarding
            continue;
          }
          u32 xorbucketid;                   // determine bucket for s0 xor s1
          const uchar *bytes0 = slot0->bytes, *bytes1 = slot1->bytes;
#if WN == 200 && BUCKBITS == 12 && RESTBITS == 8
          xorbucketid = (((u32)(bytes0[htl.prevbo+1] ^ bytes1[htl.prevbo+1]) & 0xf) << 8)
                             | (bytes0[htl.prevbo+2] ^ bytes1[htl.prevbo+2]);
#elif WN == 200 && BUCKBITS == 10 && RESTBITS == 10
          xorbucketid = (((u32)(bytes0[htl.prevbo+1] ^ bytes1[htl.prevbo+1]) & 0xf) << 6)
                             | (bytes0[htl.prevbo+2] ^ bytes1[htl.prevbo+2]) >> 2;
#elif WN == 144 && BUCKBITS == 20 && RESTBITS == 4
          xorbucketid = ((((u32)(bytes0[htl.prevbo+1] ^ bytes1[htl.prevbo+1]) << 8)
                              | (bytes0[htl.prevbo+2] ^ bytes1[htl.prevbo+2])) << 4)
                              | (bytes0[htl.prevbo+3] ^ bytes1[htl.prevbo+3]) >> 4;
#elif WN == 96 && BUCKBITS == 12 && RESTBITS == 4
          xorbucketid = ((u32)(bytes0[htl.prevbo+1] ^ bytes1[htl.prevbo+1]) << 4)
                            | (bytes0[htl.prevbo+2] ^ bytes1[htl.prevbo+2]) >> 4;
#else
#error not implemented
#endif
          // grab next available slot in that bucket
          const u32 xorslot = getslot1(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;    // SAVEMEM determines how often this happens
            continue;
          }
          // start of slot for s0 ^ s1
          htunit *xs = htl.hta.heap1[xorbucketid][xorslot];
          // store xor of hashes possibly minus initial 0 word due to collision
          for (u32 i=htl.dunits; i < htl.prevhtunits; i++)
            xs++->word = slot0[i].word ^ slot1[i].word;
          // store tree node right after hash
          xs->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  
  void digiteven(const u32 r, const u32 id) {
    htlayout htl(this, r);
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot1 *buck = htl.hta.heap1[bucketid];
      u32 bsize   = getnslots1(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htl.getxhash1(slot1));
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          if (htl.equal(slot0, slot1)) {
            hfull++;
            continue;
          }
          u32 xorbucketid;
          const uchar *bytes0 = slot0->bytes, *bytes1 = slot1->bytes;
#if WN == 200 && BUCKBITS == 12 && RESTBITS == 8
          xorbucketid = ((u32)(bytes0[htl.prevbo+1] ^ bytes1[htl.prevbo+1]) << 4)
                            | (bytes0[htl.prevbo+2] ^ bytes1[htl.prevbo+2]) >> 4;
#elif WN == 200 && BUCKBITS == 10 && RESTBITS == 10
          xorbucketid = ((u32)(bytes0[htl.prevbo+2] ^ bytes1[htl.prevbo+2]) << 2)
                            | (bytes0[htl.prevbo+3] ^ bytes1[htl.prevbo+3]) >> 6;
#elif WN == 144 && BUCKBITS == 20 && RESTBITS == 4
          xorbucketid = ((((u32)(bytes0[htl.prevbo+1] ^ bytes1[htl.prevbo+1]) << 8)
                              | (bytes0[htl.prevbo+2] ^ bytes1[htl.prevbo+2])) << 4)
                              | (bytes0[htl.prevbo+3] ^ bytes1[htl.prevbo+3]) >> 4;
#elif WN == 96 && BUCKBITS == 12 && RESTBITS == 4
          xorbucketid = ((u32)(bytes0[htl.prevbo+1] ^ bytes1[htl.prevbo+1]) << 4)
                            | (bytes0[htl.prevbo+2] ^ bytes1[htl.prevbo+2]) >> 4;
#else
#error not implemented
#endif
          const u32 xorslot = getslot0(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          htunit *xs = htl.hta.heap0[xorbucketid][xorslot];
          for (u32 i=htl.dunits; i < htl.prevhtunits; i++)
            xs++->word = slot0[i].word ^ slot1[i].word;
          xs->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  
  // functions digit1 through digit9 are unrolled versions specific to the
  // (N=200,K=9) parameters with 10 RESTBITS
  // and will be used with compile option -DUNROLL
  void digit1(const u32 id) {
    htalloc heaps = hta;
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot0 *buck = heaps.heap0[bucketid];
      u32 bsize   = getnslots0(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htobe32(slot1->word) >> 20 & 0x3ff);
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          if (slot0[5].word == slot1[5].word) {
            hfull++;
            continue;
          }
          u32 xorbucketid = htobe32(slot0->word ^ slot1->word) >> 10 & BUCKMASK;
          const u32 xorslot = getslot1(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          u64 *x  = (u64 *)heaps.heap1[xorbucketid][xorslot];
          u64 *x0 = (u64 *)slot0, *x1 = (u64 *)slot1;
          *x++ = x0[0] ^ x1[0];
          *x++ = x0[1] ^ x1[1];
          *x++ = x0[2] ^ x1[2];
          ((htunit *)x)->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  void digit2(const u32 id) {
    htalloc heaps = hta;
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot1 *buck = heaps.heap1[bucketid];
      u32 bsize   = getnslots1(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htobe32(slot1->word) & 0x3ff);
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          if (slot0[5].word == slot1[5].word) {
            hfull++;
            continue;
          }
          u32 xor1 = slot0[1].word ^ slot1[1].word;
          u32 xorbucketid = htobe32(xor1) >> 22;
          const u32 xorslot = getslot0(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          htunit *xs = heaps.heap0[xorbucketid][xorslot];
          xs++->word = xor1;
          u64 *x = (u64 *)xs, *x0 = (u64 *)slot0, *x1 = (u64 *)slot1;
          *x++ = x0[1] ^ x1[1];
          *x++ = x0[2] ^ x1[2];
          ((htunit *)x)->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  void digit3(const u32 id) {
    htalloc heaps = hta;
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot0 *buck = heaps.heap0[bucketid];
      u32 bsize   = getnslots0(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htobe32(slot1->word) >> 12 & 0x3ff);
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          if (slot0[4].word == slot1[4].word) {
            hfull++;
            continue;
          }
          u32 xor0 = slot0->word ^ slot1->word;
          u32 xorbucketid = htobe32(xor0) >> 2 & BUCKMASK;
          const u32 xorslot = getslot1(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          htunit *xs = heaps.heap1[xorbucketid][xorslot];
          xs++->word = xor0;
          u64 *x = (u64 *)xs, *x0 = (u64 *)(slot0+1), *x1 = (u64 *)(slot1+1);
          *x++ = x0[0] ^ x1[0];
          *x++ = x0[1] ^ x1[1];
          ((htunit *)x)->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  void digit4(const u32 id) {
    htalloc heaps = hta;
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot1 *buck = heaps.heap1[bucketid];
      u32 bsize   = getnslots1(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, (slot1->bytes[3] & 0x3) << 8 | slot1->bytes[4]);
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          if (slot0[4].word == slot1[4].word) {
            hfull++;
            continue;
          }
          u32 xorbucketid = htobe32(slot0[1].word ^ slot1[1].word) >> 14 & BUCKMASK;
          const u32 xorslot = getslot0(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          u64 *x  = (u64 *)heaps.heap0[xorbucketid][xorslot];
          u64 *x0 = (u64 *)(slot0+1), *x1 = (u64 *)(slot1+1);
          *x++ = x0[0] ^ x1[0];
          *x++ = x0[1] ^ x1[1];
          ((htunit *)x)->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  void digit5(const u32 id) {
    htalloc heaps = hta;
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot0 *buck = heaps.heap0[bucketid];
      u32 bsize   = getnslots0(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htobe32(slot1->word) >> 4 & 0x3ff);
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          if (slot0[3].word == slot1[3].word) {
            hfull++;
            continue;
          }
          u32 xor1 = slot0[1].word ^ slot1[1].word;
          u32 xorbucketid = (((u32)(slot0->bytes[3] ^ slot1->bytes[3]) & 0xf)
                               << 6) | (xor1 >> 2  & 0x3f);
          const u32 xorslot = getslot1(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          htunit *xs = heaps.heap1[xorbucketid][xorslot];
          xs++->word = xor1;
          u64 *x = (u64 *)xs, *x0 = (u64 *)slot0, *x1 = (u64 *)slot1;
          *x++ = x0[1] ^ x1[1];
          ((htunit *)x)->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  void digit6(const u32 id) {
    htalloc heaps = hta;
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot1 *buck = heaps.heap1[bucketid];
      u32 bsize   = getnslots1(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htobe32(slot1->word) >> 16 & 0x3ff);
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          if (slot0[2].word == slot1[2].word) {
            hfull++;
            continue;
          }
          u32 xor0 = slot0->word ^ slot1->word;
          u32 xorbucketid = htobe32(xor0) >> 6 & BUCKMASK;
          const u32 xorslot = getslot0(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          htunit *xs = heaps.heap0[xorbucketid][xorslot];
          xs++->word = xor0;
          u64 *x = (u64 *)xs, *x0 = (u64 *)(slot0+1), *x1 = (u64 *)(slot1+1);
          *x++ = x0[0] ^ x1[0];
          ((htunit *)x)->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  void digit7(const u32 id) {
    htalloc heaps = hta;
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot0 *buck = heaps.heap0[bucketid];
      u32 bsize   = getnslots0(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, (slot1->bytes[3] & 0x3f) << 4 | slot1->bytes[4] >> 4);
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          u32 xor2 = slot0[2].word ^ slot1[2].word;
          if (!xor2) {
            hfull++;
            continue;
          }
          u32 xor1 = slot0[1].word ^ slot1[1].word;
          u32 xorbucketid = htobe32(xor1) >> 18 & BUCKMASK;
          const u32 xorslot = getslot1(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          htunit *xs = heaps.heap1[xorbucketid][xorslot];
          xs++->word = xor1;
          xs++->word = xor2;
          xs->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  void digit8(const u32 id) {
    htalloc heaps = hta;
    collisiondata cd;
    for (u32 bucketid=id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot1 *buck = heaps.heap1[bucketid];
      u32 bsize   = getnslots1(bucketid);
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htobe32(slot1->word) >> 8 & 0x3ff);
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          u32 xor1 = slot0[1].word ^ slot1[1].word;
          if (!xor1) {
            hfull++;
            continue;
          }
          u32 xorbucketid = ((u32)(slot0->bytes[3] ^ slot1->bytes[3]) << 2)
                          | (xor1 >> 6 & 0x3);
          const u32 xorslot = getslot0(xorbucketid);
          if (xorslot >= NSLOTS) {
            bfull++;
            continue;
          }
          htunit *xs = heaps.heap0[xorbucketid][xorslot];
          xs++->word = xor1;
          xs->tag = tree(bucketid, s0, s1);
        }
      }
    }
  }
  
  // final round looks simpler
  void digitK(const u32 id) {
    collisiondata cd;
    htlayout htl(this, WK);
    u32 nc = 0;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += nthreads) {
      cd.clear();
      slot0 *buck = htl.hta.heap0[bucketid];   // assume WK odd
      u32 bsize   = getnslots0(bucketid);      // assume WK odd
      for (u32 s1 = 0; s1 < bsize; s1++) {
        const htunit *slot1 = buck[s1];
        cd.addslot(s1, htl.getxhash0(slot1));  // assume WK odd
        for (; cd.nextcollision(); ) {
          const u32 s0 = cd.slot();
          const htunit *slot0 = buck[s0];
          // there is only 1 word of hash left
          if (htl.equal(slot0, slot1) && slot0[1].tag.prob_disjoint(slot1[1].tag)) {
            candidate(tree(bucketid, s0, s1)); // so a match gives a solution candidate
            nc++;
          }
        }
      }
    }
    // printf(" %d candidates ", nc);  // this gets uncommented a lot for debugging
  }
};

typedef struct {
  u32 id;
  pthread_t thread;
  equi *eq;
} thread_ctx;

void barrier(pthread_barrier_t *barry) {
  const int rc = pthread_barrier_wait(barry);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier\n");
    pthread_exit(NULL);
  }
}

// do all rounds for each thread
void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  equi *eq = tp->eq;

  if (tp->id == 0) printf("Digit 0");
  eq->digit0(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(0);
  barrier(&eq->barry);
#if WN == 200 && WK == 9 && RESTBITS == 10 && defined UNROLL
  eq->digit1(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(1);
  barrier(&eq->barry);
  eq->digit2(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(2);
  barrier(&eq->barry);
  eq->digit3(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(3);
  barrier(&eq->barry);
  eq->digit4(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(4);
  barrier(&eq->barry);
  eq->digit5(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(5);
  barrier(&eq->barry);
  eq->digit6(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(6);
  barrier(&eq->barry);
  eq->digit7(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(7);
  barrier(&eq->barry);
  eq->digit8(tp->id);
  barrier(&eq->barry);
  if (tp->id == 0) eq->showbsizes(8);
  barrier(&eq->barry);
#else
  for (u32 r = 1; r < WK; r++) {
    r&1 ? eq->digitodd(r, tp->id) : eq->digiteven(r, tp->id);
    barrier(&eq->barry);
    if (tp->id == 0) eq->showbsizes(r);
    barrier(&eq->barry);
  }
#endif
  eq->digitK(tp->id);
  pthread_exit(NULL);
  return 0;
}
