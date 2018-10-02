#include <stdint.h>
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_vector.h"

__constant__ uint2 c_PaddedMessage[32];

__constant__ const uint8_t c_sigma[16][16] = {
{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },{14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },{11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },{ 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
{ 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },{ 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },{12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },{13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
{ 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },{10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 },{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },{14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
{11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },{ 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },{ 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },{ 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9}
};
#define Gprecalc(a,b,c,d,idx1,idx2) { \
	v[a] += (block[idx2] ^ u512[idx1]) + v[b]; \
	v[d] = SWAPDWORDS2( v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 25); \
	v[a] += (block[idx1] ^ u512[idx2]) + v[b]; \
	v[d] = ROR16(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 11); \
	}

__constant__ const uint2 c_u512[16] = {
	{0x85a308d3,0x243f6a88},{0x03707344,0x13198a2e},{0x299f31d0,0xa4093822},{0xec4e6c89,0x082efa98},{0x38d01377,0x452821e6},{0x34e90c6c,0xbe5466cf},{0xc97c50dd,0xc0ac29b7},{0xb5470917,0x3f84d5b5},
	{0x8979fb1b,0x9216d5d9},{0x98dfb5ac,0xd1310ba6},{0xd01adfb7,0x2ffd72db},{0x6a267e96,0xb8e1afed},{0xf12c7f99,0xba7c9045},{0xb3916cf7,0x24a19947},{0x858efc16,0x0801f2e2},{0x71574e69,0x636920d8}
};

#define G(a,b,c,d,e) \
	x1=c_sigma[i][e];\
	x2=c_sigma[i][e+1];\
    v[a] += (m[x1] ^ c_u512[x2]) + v[b]; \
    v[d] = SWAPDWORDS2(v[d] ^ v[a]); \
    v[c] += v[d]; \
    v[b] = ROR2( v[b] ^ v[c],25); \
    v[a] += (m[x2] ^ c_u512[x1])+v[b]; \
    v[d] = ROR16(v[d] ^ v[a]); \
    v[c] += v[d]; \
    v[b] = ROR2( v[b] ^ v[c],11); \


static __device__ uint32_t cuda_swap32(uint32_t x)
{
	return __byte_perm(x, 0, 0x0123);
}

static __constant__ const uint2 d_IV[8] = {
	{ 0xf3bcc908UL, 0x6a09e667UL }, { 0x84caa73bUL, 0xbb67ae85UL },
	{ 0xfe94f82bUL, 0x3c6ef372UL },
	{ 0x5f1d36f1UL, 0xa54ff53aUL },
	{ 0xade682d1UL, 0x510e527fUL },
	{ 0x2b3e6c1fUL, 0x9b05688cUL },
	{ 0xfb41bd6bUL, 0x1f83d9abUL },
	{ 0x137e2179UL, 0x5be0cd19UL }
};

static __device__ __forceinline__ uint2 swap2(uint2 v) {
	uint2 result;
	result.x = v.y;
	result.y = v.x;
	return result;
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(192, 1)
#endif
void blake_gpu_hash_185(int threads, uint32_t startNonce, uint32_t *outputHash, uint32_t *g_signature,const uint2* g_hashwholeblock){
	
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		// bestimme den aktuellen Zähler
		uint32_t nonce = startNonce + thread;
		uint32_t idx64 = thread >> 6;
		uint2 h[8];
		uint2 m[16],v[16];
		uint8_t x1, x2;
		const uint2 *hashwholeblock = &g_hashwholeblock[idx64 << 2];
		uint64_t *signature = (uint64_t *)&g_signature[idx64 << 3];

		*(uint2x4*)&h[ 0] = *(uint2x4*)&d_IV[ 0];
		*(uint2x4*)&h[ 4] = *(uint2x4*)&d_IV[ 4];
		
		#pragma unroll 11
		for (int i = 0; i < 11; ++i) m[i] = c_PaddedMessage[i];

		m[10].x = cuda_swap32(nonce);
		
		*(uint2x4*)&m[11] = __ldg4((uint2x4*)&hashwholeblock[0]);

		#pragma unroll 4
		for (int i = 0; i < 4; ++i) m[i + 11] = swap2(m[i+11]);

		m[15] = c_PaddedMessage[15];

//		blake_compress( h, buf, 1024 );
		#pragma unroll 8
		for( int i = 0; i < 8; ++i )
			v[i] = h[i];

		*(uint2x4*)&v[ 8] = *(uint2x4*)&c_u512[ 0];
		*(uint2x4*)&v[12] = *(uint2x4*)&c_u512[ 4];
		v[12] = v[12] ^ 1024;
		v[13] = v[13] ^ 1024;

		for(int i = 0; i < 16; ++i ){
			/* column step */
			G( 0, 4, 8, 12, 0 );
			G( 1, 5, 9, 13, 2 );
			G( 2, 6, 10, 14, 4 );
			G( 3, 7, 11, 15, 6 );
			/* diagonal step */
			G( 0, 5, 10, 15, 8 );
			G( 1, 6, 11, 12, 10 );
			G( 2, 7, 8, 13, 12 );
			G( 3, 4, 9, 14, 14 );
		}

		#pragma unroll 8
		for( int i = 0; i < 8; ++i )  h[i] ^= v[i] ^ v[i+8];

		#pragma unroll 16
		for (int i=0; i < 16; ++i) m[i] = c_PaddedMessage[i+16];

		#pragma unroll 32
		for (int i=0; i < 32; ++i) ((unsigned char *)m)[25+i] = ((unsigned char *)signature)[i];//SWAP64(signature[i]);

		for (int i = 0; i < 16; ++i) m[i] = vectorizeswap(devectorize(m[i]));

//		blake_compress( h, buf, 1480 );
		#pragma unroll 8
		for( int i = 0; i < 8; ++i )
			v[i] = h[i];

		v[8] = c_u512[0];
		v[9] = c_u512[1];
		v[10] = c_u512[2];
		v[11] = c_u512[3];
		v[12] = c_u512[4] ^ 1480;
		v[13] = c_u512[5] ^ 1480;
		v[14] = c_u512[6];
		v[15] = c_u512[7];

		for(int i = 0; i < 16; ++i ){
			/* column step */
			G( 0, 4, 8, 12, 0 );
			G( 1, 5, 9, 13, 2 );
			G( 2, 6, 10, 14, 4 );
			G( 3, 7, 11, 15, 6 );
			/* diagonal step */
			G( 0, 5, 10, 15, 8 );
			G( 1, 6, 11, 12, 10 );
			G( 2, 7, 8, 13, 12 );
			G( 3, 4, 9, 14, 14 );
		}
		
		#pragma unroll 8
		for(int i=0; i<8;i++){
			const uint32_t tmp = cuda_swap32(h[ i].x ^ v[ i].x ^ v[i+8].x);
			
			h[ i].x = cuda_swap32(h[ i].y ^ v[ i].y ^ v[i+8].y);
			h[ i].y = tmp;
		}
		uint32_t *outHash = (uint32_t *)outputHash + 16 * thread;
		*(uint2x4*)&outHash[ 0] = *(uint2x4*)&h[ 0];
		*(uint2x4*)&outHash[ 8] = *(uint2x4*)&h[ 4];
		
//		#pragma unroll 8
//		for (int i=0; i < 8; ++i) {
//			outHash[2*i+0] = cuda_swap32( _HIWORD(h[i]) );
//			outHash[2*i+1] = cuda_swap32( _LOWORD(h[i]) );
//		}
	}
}

__host__ void blake_cpu_setBlock_185(void *pdata)
{
	unsigned char PaddedMessage[256];
	memcpy(PaddedMessage, pdata, 185);
	memset(PaddedMessage+185, 0, 71);
	PaddedMessage[185] = 0x80;
	PaddedMessage[239] = 1;
	PaddedMessage[254] = 0x05;
	PaddedMessage[255] = 0xC8;

    //for( int i = 0; i < 32; i++ ) ((uint64_t *)PaddedMessage)[i] = host_SWAP64(((uint64_t *)PaddedMessage)[i]);
	for(int i=0;i<32;i+=2){
		const uint32_t temp = cuda_swab32(*(uint32_t*)&PaddedMessage[(i+1)*4]);
		*(uint32_t*)&PaddedMessage[(i+1)*4] = cuda_swab32(*(uint32_t*)&PaddedMessage[i*4]);
		*(uint32_t*)&PaddedMessage[i*4] = temp;
	}
	cudaMemcpyToSymbol(c_PaddedMessage, PaddedMessage, 32*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__ void blake_cpu_hash_185( int thr_id, int threads, uint32_t startNonce, uint32_t *d_outputHash, uint32_t *d_signature,const uint2 *d_hashwholeblock )
{
	const int threadsperblock = 32;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	blake_gpu_hash_185<<<grid, block >>>(threads, startNonce, d_outputHash, d_signature, d_hashwholeblock);
}
