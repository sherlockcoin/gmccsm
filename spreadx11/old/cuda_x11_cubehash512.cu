
typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

static __device__ uint32_t cuda_swab32(uint32_t x)
{
	return __byte_perm(x, 0, 0x0123);
}

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

#define CUBEHASH_ROUNDS 16 /* this is r for CubeHashr/b */
#define CUBEHASH_BLOCKBYTES 32 /* this is b for CubeHashr/b */

typedef unsigned int uint32_t; /* must be exactly 32 bits */

static __device__ __forceinline__ uint32_t ROTL32( uint32_t x, const int n )
{
#if __CUDA_ARCH__ >= 320
    return __funnelshift_l(x, x, n);
#else
    return (x << n) | (x >> (32-n));
#endif
}

#define ROTATEUPWARDS7(a) ROTL32((a), 7)
#define ROTATEUPWARDS11(a) ROTL32((a), 11)
#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }

__constant__ uint32_t c_IV_512[32];
static const uint32_t h_IV_512[32] = {
	0x2AEA2A61, 0x50F494D4, 0x2D538B8B,
	0x4167D83E, 0x3FEE2313, 0xC701CF8C,
	0xCC39968E, 0x50AC5695, 0x4D42C787,
	0xA647A8B3, 0x97CF0BEF, 0x825B4537,
	0xEEF864D2, 0xF22090C4, 0xD0E5CD33,
	0xA23911AE, 0xFCD398D9, 0x148FE485,
	0x1B017BEF, 0xB6444532, 0x6A536159,
	0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
	0xD65C8A2B, 0xA5A70E75, 0xB1C62456,
	0xBC796576, 0x1921C8F7, 0xE7989AF1,
	0x7795D246, 0xD43E3B44
};

static __device__ void rrounds(uint32_t x[2][2][2][2][2])
{
    int r;
    int j;
    int k;
    int l;
    int m;

//#pragma unroll 16
    for (r = 0;r < CUBEHASH_ROUNDS;++r) {

        /* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[1][j][k][l][m] += x[0][j][k][l][m];

        /* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[0][j][k][l][m] = ROTATEUPWARDS7(x[0][j][k][l][m]);

        /* "swap x_00klm with x_01klm" */
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
            for (l = 0;l < 2;++l)
#pragma unroll 2
                for (m = 0;m < 2;++m)
                    SWAP(x[0][0][k][l][m],x[0][1][k][l][m])

        /* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[0][j][k][l][m] ^= x[1][j][k][l][m];

        /* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (m = 0;m < 2;++m)
                    SWAP(x[1][j][k][0][m],x[1][j][k][1][m])

        /* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[1][j][k][l][m] += x[0][j][k][l][m];

        /* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

        /* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (l = 0;l < 2;++l)
#pragma unroll 2
                for (m = 0;m < 2;++m)
                    SWAP(x[0][j][0][l][m],x[0][j][1][l][m])

        /* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[0][j][k][l][m] ^= x[1][j][k][l][m];

        /* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
                    SWAP(x[1][j][k][l][0],x[1][j][k][l][1])

    }
}


static __device__ void block_tox(uint32_t block[16], uint32_t x[2][2][2][2][2])
{
    int k;
    int l;
    int m;
    uint32_t *in = block;

#pragma unroll 2
    for (k = 0;k < 2;++k)
#pragma unroll 2
        for (l = 0;l < 2;++l)
#pragma unroll 2
            for (m = 0;m < 2;++m)
                x[0][0][k][l][m] ^= *in++;
}

static __device__ void block_toxC(uint32_t *in, uint32_t x[2][2][2][2][2], int threads, int thread, int offset)
{
    int k;
    int l;
    int m;
    int i;

#pragma unroll 2
    for (k = 0;k < 2;++k)
#pragma unroll 2
        for (l = 0;l < 2;++l)
#pragma unroll 2
            for (m = 0;m < 2;++m) {
                x[0][0][k][l][m] ^= in[threads*(i+offset)+thread];
                i++;
            }

}

static __device__ void hash_fromx(uint32_t hash[16], uint32_t x[2][2][2][2][2])
{
    int j;
    int k;
    int l;
    int m;
    uint32_t *out = hash;

#pragma unroll 2
    for (j = 0;j < 2;++j)
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
            for (l = 0;l < 2;++l)
#pragma unroll 2
                for (m = 0;m < 2;++m)
                    *out++ = x[0][j][k][l][m];
}

static __device__ void hash_fromxC(uint32_t *out, uint32_t x[2][2][2][2][2], int threads, int thread)
{
    int i;
    int j;
    int k;
    int l;
    int m;

#pragma unroll 2
    for (j = 0;j < 2;++j)
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
            for (l = 0;l < 2;++l)
#pragma unroll 2
                for (m = 0;m < 2;++m) {
                    out[i*threads+thread] = x[0][j][k][l][m];
                    i++;
                }
}

void __device__ Init(uint32_t x[2][2][2][2][2])
{
    int i,j,k,l,m;
#if 0
    /* "the first three state words x_00000, x_00001, x_00010" */
    /* "are set to the integers h/8, b, r respectively." */
    /* "the remaining state words are set to 0." */
#pragma unroll 2
    for (i = 0;i < 2;++i)
#pragma unroll 2
      for (j = 0;j < 2;++j)
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
          for (l = 0;l < 2;++l)
#pragma unroll 2
            for (m = 0;m < 2;++m)
              x[i][j][k][l][m] = 0;
    x[0][0][0][0][0] = 512/8;
    x[0][0][0][0][1] = CUBEHASH_BLOCKBYTES;
    x[0][0][0][1][0] = CUBEHASH_ROUNDS;

    /* "the state is then transformed invertibly through 10r identical rounds */
    for (i = 0;i < 10;++i) rrounds(x);
#else
    uint32_t *iv = c_IV_512;

#pragma unroll 2
    for (i = 0;i < 2;++i)
#pragma unroll 2
      for (j = 0;j < 2;++j)
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
          for (l = 0;l < 2;++l)
#pragma unroll 2
            for (m = 0;m < 2;++m)
              x[i][j][k][l][m] = *iv++;
#endif
}

void __device__ Update32(uint32_t x[2][2][2][2][2], const BitSequence *data)
{
    /* "xor the block into the first b bytes of the state" */
    /* "and then transform the state invertibly through r identical rounds" */
    block_tox((uint32_t*)data, x);
    rrounds(x);
}

void __device__ Update32c(uint32_t x[2][2][2][2][2], const BitSequence *data, int threads, int thread, int offset)
{
    /* "xor the block into the first b bytes of the state" */
    /* "and then transform the state invertibly through r identical rounds" */
    block_toxC((uint32_t*)data, x, threads, thread, offset);
    rrounds(x);
}

void __device__ Final(uint32_t x[2][2][2][2][2], BitSequence *hashval)
{
    int i;

    /* "the integer 1 is xored into the last state word x_11111" */
    x[1][1][1][1][1] ^= 1;

    /* "the state is then transformed invertibly through 10r identical rounds" */
//#pragma unroll 10
    for (i = 0;i < 10;++i) rrounds(x);

    /* "output the first h/8 bytes of the state" */
    hash_fromx((uint32_t*)hashval, x);
}

void __device__ FinalC(uint32_t x[2][2][2][2][2], const BitSequence *hashval, int threads, int thread)
{
    int i;

    /* "the integer 1 is xored into the last state word x_11111" */
    x[1][1][1][1][1] ^= 1;

    /* "the state is then transformed invertibly through 10r identical rounds" */
//#pragma unroll 10
    for (i = 0;i < 10;++i) rrounds(x);

    /* "output the first h/8 bytes of the state" */
    hash_fromxC((uint32_t*)hashval, x, threads, thread);
}


__global__ void x11_cubehash512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[8 * hashPosition];

        uint32_t x[2][2][2][2][2];
        Init(x);

        // erste H�lfte des Hashes (32 bytes)
        Update32(x, (const BitSequence*)Hash);

        // zweite H�lfte des Hashes (32 bytes)
        Update32(x, (const BitSequence*)(Hash+8));

        // Padding Block
        uint32_t last[8];
        last[0] = 0x80;
#pragma unroll 7
        for (int i=1; i < 8; i++) last[i] = 0;
        Update32(x, (const BitSequence*)last);

        Final(x, (BitSequence*)Hash);
    }
}

__global__ void x11_cubehash512c_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        uint32_t x[2][2][2][2][2];
        Init(x);

        // erste H�lfte des Hashes (32 bytes)
        Update32c(x, (const BitSequence*)g_hash, threads, thread, 0);

        // zweite H�lfte des Hashes (32 bytes)
        Update32c(x, (const BitSequence*)g_hash, threads, thread, 8);

        // Padding Block
        uint32_t last[8];
        last[0] = 0x80;
#pragma unroll 7
        for (int i=1; i < 8; i++) last[i] = 0;
        Update32(x, (const BitSequence*)last);

        FinalC(x, (BitSequence*)g_hash, threads, thread);
    }
}


// Setup-Funktionen
__host__ void x11_cubehash512_cpu_init(int thr_id, int threads)
{
    cudaMemcpyToSymbol( c_IV_512, h_IV_512, sizeof(h_IV_512), 0, cudaMemcpyHostToDevice);
}

__host__ void x11_cubehash512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
    const int threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Gr��e des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    x11_cubehash512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

//	cudaDeviceSynchronize();
}

__host__ void x11_cubehash512c_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
    const int threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Gr��e des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    x11_cubehash512c_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

//	cudaDeviceSynchronize();
}

