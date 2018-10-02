#include <stdint.h>
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h" 
#include "cuda_vector.h" 

static __constant__ uint64_t c_PaddedMessage80[2];
__constant__ uint2 precalcvalues[9];
__constant__ uint32_t sha256_endingTable[64];

static uint32_t *d_found[MAX_GPUS];

// Take a look at: https://www.schneier.com/skein1.3.pdf

#define SHL(x, n)			((x) << (n))
#define SHR(x, n)			((x) >> (n))

#define TPB52 512
#define TPB50 512

#include "skein_header.h"

uint32_t *d_nonce[MAX_GPUS];

/* ************************ */
__constant__ const uint2 buffer[152] = {
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC33,0xAE18A40B},
	{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC73,0x9E18A40B},{0x98173EC5,0xCAB2076D},
	{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC73,0x9E18A40B},{0x98173F04,0xCAB2076D},{0x749C51D0,0x4903ADFF},
	{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173F04,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF06,0x0D95DE39},
	{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79BD2,0x8FD19341},
	{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB6,0x9A255629},
	{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7B6,0x5DB62599},
	{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C3FB,0xEABE394C},
	{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B52B,0x991112C7},
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC3C,0xAE18A40B},
	{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC73,0x9E18A40B},{0x98173ece,0xcab2076d},
	{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC73,0x9E18A40B},{0x98173F04,0xCAB2076D},{0x749C51D9,0x4903ADFF},
	{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173F04,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF0F,0x0D95DE39},
	{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79BDB,0x8FD19341},
	{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CBF,0x9A255629},
	{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7BF,0x5DB62599},
	{0x660FCC33,0xAE18A40B},{0x98173ec4,0xcab2076d},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C404,0xEABE394C},
	{0x98173ec4,0xcab2076d},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B534,0x991112C7},
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC45,0xAE18A40B}
};

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52, 3)
#else
__launch_bounds__(TPB50, 3)
#endif
void quark_skein512_gpu_hash_64(uint32_t threads, uint64_t * const __restrict__ g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 p[8], h[9];

		uint64_t *Hash = &g_hash[thread<<3];

		uint2x4 *phash = (uint2x4*)Hash;
		*(uint2x4*)&p[0] = __ldg4(&phash[0]);
		*(uint2x4*)&p[4] = __ldg4(&phash[1]);
		
		h[0] = p[0];	h[1] = p[1];	h[2] = p[2];	h[3] = p[3];
		h[4] = p[4];	h[5] = p[5];	h[6] = p[6];	h[7] = p[7];

		p[0] += buffer[0];	p[1] += buffer[1];	p[2] += buffer[2];	p[3] += buffer[3];	p[4] += buffer[4];	p[5] += buffer[5];	p[6] += buffer[6];	p[7] += buffer[7];
		TFBIGMIX8e();
		p[0] += buffer[8];	p[1] += buffer[9];	p[2] += buffer[10];	p[3] += buffer[11];	p[4] += buffer[12];	p[5] += buffer[13];	p[6] += buffer[14];	p[7] += buffer[15];
		TFBIGMIX8o();
		p[0] += buffer[16];	p[1] += buffer[17];	p[2] += buffer[18];	p[3] += buffer[19];	p[4] += buffer[20];	p[5] += buffer[21];	p[6] += buffer[22];	p[7] += buffer[23];
		TFBIGMIX8e();
		p[0] += buffer[24];	p[1] += buffer[25];	p[2] += buffer[26];	p[3] += buffer[27];	p[4] += buffer[28];	p[5] += buffer[29];	p[6] += buffer[30];	p[7] += buffer[31];
		TFBIGMIX8o();
		p[0] += buffer[32];	p[1] += buffer[33];	p[2] += buffer[34];	p[3] += buffer[35];	p[4] += buffer[36];	p[5] += buffer[37];	p[6] += buffer[38];	p[7] += buffer[39];
		TFBIGMIX8e();
		p[0] += buffer[40];	p[1] += buffer[41];	p[2] += buffer[42];	p[3] += buffer[43];	p[4] += buffer[44];	p[5] += buffer[45];	p[6] += buffer[46];	p[7] += buffer[47];
		TFBIGMIX8o();
		p[0] += buffer[48];	p[1] += buffer[49];	p[2] += buffer[50];	p[3] += buffer[51];	p[4] += buffer[52];	p[5] += buffer[53];	p[6] += buffer[54];	p[7] += buffer[55];
		TFBIGMIX8e();
		p[0] += buffer[56];	p[1] += buffer[57];	p[2] += buffer[58];	p[3] += buffer[59];	p[4] += buffer[60];	p[5] += buffer[61];	p[6] += buffer[62];	p[7] += buffer[63];
		TFBIGMIX8o();
		p[0] += buffer[64];	p[1] += buffer[65];	p[2] += buffer[66];	p[3] += buffer[67];	p[4] += buffer[68];	p[5] += buffer[69];	p[6] += buffer[70];	p[7] += buffer[71];
		TFBIGMIX8e();
		p[0] += buffer[72];	p[1] += buffer[73];	p[2] += buffer[74];	p[3] += buffer[75];	p[4] += buffer[76];	p[5] += buffer[77];	p[6] += buffer[78];	p[7] += buffer[79];
		TFBIGMIX8o();
		p[0] += buffer[80];	p[1] += buffer[81];	p[2] += buffer[82];	p[3] += buffer[83];	p[4] += buffer[84];	p[5] += buffer[85];	p[6] += buffer[86];	p[7] += buffer[87];
		TFBIGMIX8e();
		p[0] += buffer[88];	p[1] += buffer[89];	p[2] += buffer[90];	p[3] += buffer[91];	p[4] += buffer[92];	p[5] += buffer[93];	p[6] += buffer[94];	p[7] += buffer[95];
		TFBIGMIX8o();
		p[0] += buffer[96];	p[1] += buffer[97];	p[2] += buffer[98];	p[3] += buffer[99];	p[4] += buffer[100];	p[5] += buffer[101];	p[6] += buffer[102];	p[7] += buffer[103];
		TFBIGMIX8e();
		p[0] += buffer[104];	p[1] += buffer[105];	p[2] += buffer[106];	p[3] += buffer[107];	p[4] += buffer[108];	p[5] += buffer[109];	p[6] += buffer[110];	p[7] += buffer[111];
		TFBIGMIX8o();
		p[0] += buffer[112];	p[1] += buffer[113];	p[2] += buffer[114];	p[3] += buffer[115];	p[4] += buffer[116];	p[5] += buffer[117];	p[6] += buffer[118];	p[7] += buffer[119];
		TFBIGMIX8e();
		p[0] += buffer[120];	p[1] += buffer[121];	p[2] += buffer[122];	p[3] += buffer[123];	p[4] += buffer[124];	p[5] += buffer[125];	p[6] += buffer[126];	p[7] += buffer[127];
		TFBIGMIX8o();
		p[0] += buffer[128];	p[1] += buffer[129];	p[2] += buffer[130];	p[3] += buffer[131];	p[4] += buffer[132];	p[5] += buffer[133];	p[6] += buffer[134];	p[7] += buffer[135];
		TFBIGMIX8e();
		p[0] += buffer[136];	p[1] += buffer[137];	p[2] += buffer[138];	p[3] += buffer[139];	p[4] += buffer[140];	p[5] += buffer[141];	p[6] += buffer[142];	p[7] += buffer[143];
		TFBIGMIX8o();
		p[0] += buffer[144];	p[1] += buffer[145];	p[2] += buffer[146];	p[3] += buffer[147];	p[4] += buffer[148];	p[5] += buffer[149];	p[6] += buffer[150];	p[7] += buffer[151];

		h[0]^= p[0];
		h[1]^= p[1];
		h[2]^= p[2];
		h[3]^= p[3];
		h[4]^= p[4];
		h[5]^= p[5];
		h[6]^= p[6];
		h[7]^= p[7];
		
		h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);

		uint32_t t0;
		uint2 t1,t2;
		t0 = 8;
		t1 = vectorize(0xFF00000000000000);
		t2 = t1+t0;

		p[5] = h[5] + 8U;

		p[0] = h[0] + h[1];
		p[1] = ROL2(h[1], 46) ^ p[0];
		p[2] = h[2] + h[3];
		p[3] = ROL2(h[3], 36) ^ p[2];
		p[4] = h[4] + p[5];
		p[5] = ROL2(p[5], 19) ^ p[4];
		p[6] = (h[6] + h[7] + t1);
		p[7] = ROL2(h[7], 37) ^ p[6];
		p[2]+= p[1];
		p[1] = ROL2(p[1], 33) ^ p[2];
		p[4]+= p[7];
		p[7] = ROL2(p[7], 27) ^ p[4];
		p[6]+= p[5];
		p[5] = ROL2(p[5], 14) ^ p[6];
		p[0]+= p[3];
		p[3] = ROL2(p[3], 42) ^ p[0];
		p[4]+= p[1];
		p[1] = ROL2(p[1], 17) ^ p[4];
		p[6]+= p[3];
		p[3] = ROL2(p[3], 49) ^ p[6];
		p[0]+= p[5];
		p[5] = ROL2(p[5], 36) ^ p[0];
		p[2]+= p[7];
		p[7] = ROL2(p[7], 39) ^ p[2];
		p[6]+= p[1];
		p[1] = ROL2(p[1], 44) ^ p[6];
		p[0]+= p[7];
		p[7] = ROL2(p[7], 9) ^ p[0];
		p[2]+= p[5];
		p[5] = ROL2(p[5], 54) ^ p[2];
		p[4]+= p[3];
		p[3] = ROR8(p[3]) ^ p[4];
		
		p[0]+= h[1];		p[1]+= h[2];
		p[2]+= h[3];		p[3]+= h[4];
		p[4]+= h[5];		p[5]+= h[6] + t1;
		p[6]+= h[7] + t2;	p[7]+= h[8] + 1U;
		TFBIGMIX8o();
		p[0]+= h[2];		p[1]+= h[3];
		p[2]+= h[4];		p[3]+= h[5];
		p[4]+= h[6];		p[5]+= h[7] + t2;
		p[6]+= h[8] + t0;	p[7]+= h[0] + 2U;
		TFBIGMIX8e();
		p[0]+= h[3];		p[1]+= h[4];
		p[2]+= h[5];		p[3]+= h[6];
		p[4]+= h[7];		p[5]+= h[8] + t0;
		p[6]+= h[0] + t1;	p[7]+= h[1] + 3U;
		TFBIGMIX8o();
		p[0]+= h[4];		p[1]+= h[5];
		p[2]+= h[6];		p[3]+= h[7];
		p[4]+= h[8];		p[5]+= h[0] + t1;
		p[6]+= h[1] + t2;	p[7]+= h[2] + 4U;
		TFBIGMIX8e();
		p[0]+= h[5];		p[1]+= h[6];
		p[2]+= h[7];		p[3]+= h[8];
		p[4]+= h[0];		p[5]+= h[1] + t2;
		p[6]+= h[2] + t0;	p[7]+= h[3] + 5U;
		TFBIGMIX8o();
		p[0]+= h[6];		p[1]+= h[7];
		p[2]+= h[8];		p[3]+= h[0];
		p[4]+= h[1];		p[5]+= h[2] + t0;
		p[6]+= h[3] + t1;	p[7]+= h[4] + 6U;
		TFBIGMIX8e();
		p[0]+= h[7];		p[1]+= h[8];
		p[2]+= h[0];		p[3]+= h[1];
		p[4]+= h[2];		p[5]+= h[3] + t1;
		p[6]+= h[4] + t2;	p[7]+= h[5] + 7U;
		TFBIGMIX8o();
		p[0]+= h[8];		p[1]+= h[0];
		p[2]+= h[1];		p[3]+= h[2];
		p[4]+= h[3];		p[5]+= h[4] + t2;
		p[6]+= h[5] + t0;	p[7]+= h[6] + 8U;
		TFBIGMIX8e();
		p[0]+= h[0];		p[1]+= h[1];
		p[2]+= h[2];		p[3]+= h[3];
		p[4]+= h[4];		p[5]+= h[5] + t0;
		p[6]+= h[6] + t1;	p[7]+= h[7] + 9U;
		TFBIGMIX8o();
		p[0] = p[0] + h[1];	p[1] = p[1] + h[2];
		p[2] = p[2] + h[3];	p[3] = p[3] + h[4];
		p[4] = p[4] + h[5];	p[5] = p[5] + h[6] + t1;
		p[6] = p[6] + h[7] + t2;p[7] = p[7] + h[8] + 10U;
		TFBIGMIX8e();
		p[0] = p[0] + h[2];	p[1] = p[1] + h[3];
		p[2] = p[2] + h[4];	p[3] = p[3] + h[5];
		p[4] = p[4] + h[6];	p[5] = p[5] + h[7] + t2;
		p[6] = p[6] + h[8] + t0;p[7] = p[7] + h[0] + 11U;
		TFBIGMIX8o();
		p[0] = p[0] + h[3];	p[1] = p[1] + h[4];
		p[2] = p[2] + h[5];	p[3] = p[3] + h[6];
		p[4] = p[4] + h[7];	p[5] = p[5] + h[8] + t0;
		p[6] = p[6] + h[0] + t1;p[7] = p[7] + h[1] + 12U;
		TFBIGMIX8e();
		p[0] = p[0] + h[4];	p[1] = p[1] + h[5];
		p[2] = p[2] + h[6];	p[3] = p[3] + h[7];
		p[4] = p[4] + h[8];	p[5] = p[5] + h[0] + t1;
		p[6] = p[6] + h[1] + t2;p[7] = p[7] + h[2] + 13U;
		TFBIGMIX8o();
		p[0] = p[0] + h[5];	p[1] = p[1] + h[6];
		p[2] = p[2] + h[7];	p[3] = p[3] + h[8];
		p[4] = p[4] + h[0];	p[5] = p[5] + h[1] + t2;
		p[6] = p[6] + h[2] + t0;p[7] = p[7] + h[3] + 14U;
		TFBIGMIX8e();
		p[0] = p[0] + h[6];	p[1] = p[1] + h[7];
		p[2] = p[2] + h[8];	p[3] = p[3] + h[0];
		p[4] = p[4] + h[1];	p[5] = p[5] + h[2] + t0;
		p[6] = p[6] + h[3] + t1;p[7] = p[7] + h[4] + 15U;
		TFBIGMIX8o();
		p[0] = p[0] + h[7];	p[1] = p[1] + h[8];
		p[2] = p[2] + h[0];	p[3] = p[3] + h[1];
		p[4] = p[4] + h[2];	p[5] = p[5] + h[3] + t1;
		p[6] = p[6] + h[4] + t2;p[7] = p[7] + h[5] + 16U;
		TFBIGMIX8e();
		p[0] = p[0] + h[8];	p[1] = p[1] + h[0];
		p[2] = p[2] + h[1];	p[3] = p[3] + h[2];
		p[4] = p[4] + h[3];	p[5] = p[5] + h[4] + t2;
		p[6] = p[6] + h[5] + t0;p[7] = p[7] + h[6] + 17U;
		TFBIGMIX8o();
		p[0] = p[0] + h[0];	p[1] = p[1] + h[1];
		p[2] = p[2] + h[2];	p[3] = p[3] + h[3];
		p[4] = p[4] + h[4];	p[5] = p[5] + h[5] + t0;
		p[6] = p[6] + h[6] + t1;p[7] = p[7] + h[7] + 18U;
		
		phash = (uint28*)p;
		uint28 *outpt = (uint28*)Hash;
		outpt[0] = phash[0];
		outpt[1] = phash[1];

	}
}


__host__ void quark_skein512_cpu_init(int thr_id)
{
	cudaMalloc(&d_nonce[thr_id], 2*sizeof(uint32_t));
}

__host__ void quark_skein512_setTarget(const void *ptarget)
{
}
__host__ void quark_skein512_cpu_free(int32_t thr_id)
{
	cudaFreeHost(&d_nonce[thr_id]);
}


static __device__ __constant__ uint32_t sha256_hashTable[] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};


/* Elementary functions used by SHA256 */
#define SWAB32(x)     cuda_swab32(x)
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define R(x, n)       ((x) >> (n))
#define Ch(x, y, z)   ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)  ((x & (y | z)) | (y & z))
#define S0(x)         (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)         (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)         (ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)         (ROTR32(x, 17) ^ ROTR32(x, 19) ^ R(x, 10))


__constant__ uint32_t sha256_constantTable[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__global__ __launch_bounds__(1024)
void skein512_gpu_hash_80_52(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ d_found, uint64_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;
		uint2 p[8];

		uint32_t nonce = (startNounce + thread);


		h0 = precalcvalues[0];
		h1 = precalcvalues[1];
		h2 = precalcvalues[2];
		h3 = precalcvalues[3];
		h4 = precalcvalues[4];
		h5 = precalcvalues[5];
		h6 = precalcvalues[6];
		h7 = precalcvalues[7];
		t2 = precalcvalues[8];

		const uint2 nounce2 = make_uint2(_LOWORD(c_PaddedMessage80[1]), cuda_swab32(startNounce + thread));

		// skein_big_close -> etype = 0x160, ptr = 16, bcount = 1, extra = 16
		p[0] = vectorize(c_PaddedMessage80[0]);
		p[1] = nounce2;

		#pragma unroll
		for (int i = 2; i < 8; i++)
			p[i] = make_uint2(0,0);

		t0 = vectorizelow(0x50ull); // SPH_T64(bcount << 6) + (sph_u64)(extra);
		t1 = vectorizehigh(0xB0000000ul); // (bcount >> 58) + ((sph_u64)(etype) << 55);
		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);


		t0 = vectorizelow(8); // extra
		t1 = vectorizehigh(0xFF000000ul); // etype

		h0 = vectorize(c_PaddedMessage80[0]) ^ p[0];
		h1 = nounce2 ^ p[1];
		h2 = p[2];
		h3 = p[3];
		h4 = p[4];
		h5 = p[5];
		h6 = p[6];
		h7 = p[7];

		h8 = h0 ^ h1 ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7] ^ vectorize(0x1BD11BDAA9FC1A22);
		t2 = vectorize(0xFF00000000000008ull);

		// p[8] = { 0 };
		#pragma unroll 8
		for (int i = 0; i<8; i++)
			p[i] = make_uint2(0, 0);

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint32_t *message = (uint32_t *)p;	

		uint32_t W1[16];
		uint32_t W2[16];

		uint32_t regs[8];
		uint32_t hash[8];

		// Init with Hash-Table
#pragma unroll 8
		for (int k = 0; k < 8; k++)
		{
			hash[k] = regs[k] = sha256_hashTable[k];
		}

#pragma unroll 16
		for (int k = 0; k<16; k++)
			W1[k] = SWAB32(message[k]);

		// Progress W1
#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j] + W1[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int k = 6; k >= 0; k--) regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		// Progress W2...W3

		////// PART 1
#pragma unroll 2
		for (int j = 0; j<2; j++)
			W2[j] = s1(W1[14 + j]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];
#pragma unroll 5
		for (int j = 2; j<7; j++)
			W2[j] = s1(W2[j - 2]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

#pragma unroll 8
		for (int j = 7; j<15; j++)
			W2[j] = s1(W2[j - 2]) + W2[j - 7] + s0(W1[1 + j]) + W1[j];

		W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

		// Round function
#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 16] + W2[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		////// PART 2
#pragma unroll 2
		for (int j = 0; j<2; j++)
			W1[j] = s1(W2[14 + j]) + W2[9 + j] + s0(W2[1 + j]) + W2[j];

#pragma unroll 5
		for (int j = 2; j<7; j++)
			W1[j] = s1(W1[j - 2]) + W2[9 + j] + s0(W2[1 + j]) + W2[j];

#pragma unroll 8
		for (int j = 7; j<15; j++)
			W1[j] = s1(W1[j - 2]) + W1[j - 7] + s0(W2[1 + j]) + W2[j];

		W1[15] = s1(W1[13]) + W1[8] + s0(W1[0]) + W2[15];

		// Round function
#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 32] + W1[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		////// PART 3
#pragma unroll 2
		for (int j = 0; j<2; j++)
			W2[j] = s1(W1[14 + j]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

#pragma unroll 5
		for (int j = 2; j<7; j++)
			W2[j] = s1(W2[j - 2]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

#pragma unroll 8
		for (int j = 7; j<15; j++)
			W2[j] = s1(W2[j - 2]) + W2[j - 7] + s0(W1[1 + j]) + W1[j];

		W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

		// Round function
#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 48] + W2[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

#pragma unroll 8
		for (int k = 0; k<8; k++)
			hash[k] += regs[k];

		/////
		///// Second Pass (ending)
		/////
#pragma unroll 8
		for (int k = 0; k<8; k++)
			regs[k] = hash[k];

		// Progress W1
		uint32_t T1, T2;
#pragma unroll 
		for (int j = 0; j<56; j++)
		{
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_endingTable[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int k = 6; k >= 0; k--)
				regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_endingTable[56];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		regs[7] = T1 + T2;
		regs[3] += T1;

		T1 = regs[6] + S1(regs[3]) + Ch(regs[3], regs[4], regs[5]) + sha256_endingTable[57];
		T2 = S0(regs[7]) + Maj(regs[7], regs[0], regs[1]);
		regs[6] = T1 + T2;
		regs[2] += T1;
		//************
		regs[1] += regs[5] + S1(regs[2]) + Ch(regs[2], regs[3], regs[4]) + sha256_endingTable[58];
		regs[0] += regs[4] + S1(regs[1]) + Ch(regs[1], regs[2], regs[3]) + sha256_endingTable[59];
		regs[7] += regs[3] + S1(regs[0]) + Ch(regs[0], regs[1], regs[2]) + sha256_endingTable[60];
		regs[6] += regs[2] + S1(regs[7]) + Ch(regs[7], regs[0], regs[1]) + sha256_endingTable[61];

		uint64_t test = SWAB32(hash[7] + regs[7]);
		test <<= 32;
		test |= SWAB32(hash[6] + regs[6]);
		if (test <= target)
		{
			uint32_t tmp = atomicExch(&(d_found[0]), nonce);
			if (tmp != 0xffffffff)
				d_found[1] = nonce;
		}
	}
}
__global__
void skein512_gpu_hash_80_50(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ d_found, uint64_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;
		uint2 p[8];

		uint32_t nounce = (startNounce + thread);

		h0 = precalcvalues[0];
		h1 = precalcvalues[1];
		h2 = precalcvalues[2];
		h3 = precalcvalues[3];
		h4 = precalcvalues[4];
		h5 = precalcvalues[5];
		h6 = precalcvalues[6];
		h7 = precalcvalues[7];
		t2 = precalcvalues[8];

		const uint2 nounce2 = make_uint2(_LOWORD(c_PaddedMessage80[1]), cuda_swab32(startNounce + thread));

		// skein_big_close -> etype = 0x160, ptr = 16, bcount = 1, extra = 16
		p[0] = vectorize(c_PaddedMessage80[0]);
		p[1] = nounce2;

#pragma unroll
		for (int i = 2; i < 8; i++)
			p[i] = make_uint2(0, 0);

		t0 = vectorizelow(0x50ull); // SPH_T64(bcount << 6) + (sph_u64)(extra);
		t1 = vectorizehigh(0xB0000000ul); // (bcount >> 58) + ((sph_u64)(etype) << 55);
		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);


		t0 = vectorizelow(8); // extra
		t1 = vectorizehigh(0xFF000000ul); // etype

		h0 = vectorize(c_PaddedMessage80[0]) ^ p[0];
		h1 = nounce2 ^ p[1];
		h2 = p[2];
		h3 = p[3];
		h4 = p[4];
		h5 = p[5];
		h6 = p[6];
		h7 = p[7];

		h8 = h0 ^ h1 ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7] ^ vectorize(0x1BD11BDAA9FC1A22);
		t2 = vectorize(0xFF00000000000008ull);

		// p[8] = { 0 };
#pragma unroll 8
		for (int i = 0; i<8; i++)
			p[i] = make_uint2(0, 0);

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint32_t *message = (uint32_t *)p;

		uint32_t W1[16];
		uint32_t W2[16];

		uint32_t regs[8];
		uint32_t hash[8];

		// Init with Hash-Table
#pragma unroll 8
		for (int k = 0; k < 8; k++)
		{
			hash[k] = regs[k] = sha256_hashTable[k];
		}

#pragma unroll 16
		for (int k = 0; k<16; k++)
			W1[k] = SWAB32(message[k]);

		// Progress W1
#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j] + W1[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int k = 6; k >= 0; k--) regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		// Progress W2...W3

		////// PART 1
#pragma unroll 2
		for (int j = 0; j<2; j++)
			W2[j] = s1(W1[14 + j]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];
#pragma unroll 5
		for (int j = 2; j<7; j++)
			W2[j] = s1(W2[j - 2]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

#pragma unroll 8
		for (int j = 7; j<15; j++)
			W2[j] = s1(W2[j - 2]) + W2[j - 7] + s0(W1[1 + j]) + W1[j];

		W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

		// Round function
#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 16] + W2[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		////// PART 2
#pragma unroll 2
		for (int j = 0; j<2; j++)
			W1[j] = s1(W2[14 + j]) + W2[9 + j] + s0(W2[1 + j]) + W2[j];

#pragma unroll 5
		for (int j = 2; j<7; j++)
			W1[j] = s1(W1[j - 2]) + W2[9 + j] + s0(W2[1 + j]) + W2[j];

#pragma unroll 8
		for (int j = 7; j<15; j++)
			W1[j] = s1(W1[j - 2]) + W1[j - 7] + s0(W2[1 + j]) + W2[j];

		W1[15] = s1(W1[13]) + W1[8] + s0(W1[0]) + W2[15];

		// Round function
#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 32] + W1[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		////// PART 3
#pragma unroll 2
		for (int j = 0; j<2; j++)
			W2[j] = s1(W1[14 + j]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

#pragma unroll 5
		for (int j = 2; j<7; j++)
			W2[j] = s1(W2[j - 2]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

#pragma unroll 8
		for (int j = 7; j<15; j++)
			W2[j] = s1(W2[j - 2]) + W2[j - 7] + s0(W1[1 + j]) + W1[j];

		W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

		// Round function
#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 48] + W2[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

#pragma unroll 8
		for (int k = 0; k<8; k++)
			hash[k] += regs[k];

		/////
		///// Second Pass (ending)
		/////
#pragma unroll 8
		for (int k = 0; k<8; k++)
			regs[k] = hash[k];

		// Progress W1
		uint32_t T1, T2;
#pragma unroll 1
		for (int j = 0; j<56; j++)//62
		{
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_endingTable[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int k = 6; k >= 0; k--)
				regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6])+sha256_endingTable[56];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		regs[7] = T1 + T2;
		regs[3] += T1;

		T1 = regs[6] + S1(regs[3]) + Ch(regs[3], regs[4], regs[5]) + sha256_endingTable[57];
		T2 = S0(regs[7]) + Maj(regs[7], regs[0], regs[1]);
		regs[6] = T1 + T2;
		regs[2] += T1;
		//************
		regs[1] += regs[5] + S1(regs[2]) + Ch(regs[2], regs[3], regs[4]) + sha256_endingTable[58];
		regs[0] += regs[4] + S1(regs[1]) + Ch(regs[1], regs[2], regs[3]) + sha256_endingTable[59];
		regs[7] += regs[3] + S1(regs[0]) + Ch(regs[0], regs[1], regs[2]) + sha256_endingTable[60];
		regs[6] += regs[2] + S1(regs[7]) + Ch(regs[7], regs[0], regs[1]) + sha256_endingTable[61];

		uint64_t test = SWAB32(hash[7] + regs[7]);
		test <<= 32;
		test|= SWAB32(hash[6] + regs[6]);
		if (test <= target)
		{
			uint32_t tmp = atomicCAS(d_found, 0xffffffff, nounce);
			if (tmp != 0xffffffff)
				d_found[1] = nounce;
		}
	}
}

__host__
void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	uint32_t tpb = TPB52;
	int dev_id = device_map[thr_id];
	
	if (device_sm[dev_id] <= 500) tpb = TPB50;
	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);
	
	quark_skein512_gpu_hash_64 << <grid, block >> >(threads, (uint64_t*)d_hash);

}

static uint64_t PaddedMessage[16];

static void precalc()
{
	uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
	uint64_t t0, t1, t2;

	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	uint64_t p[8];
	for (int i = 0; i<8; i++)
		p[i] = PaddedMessage[i];

	TFBIG_4e_PRE(0);
	TFBIG_4o_PRE(1);
	TFBIG_4e_PRE(2);
	TFBIG_4o_PRE(3);
	TFBIG_4e_PRE(4);
	TFBIG_4o_PRE(5);
	TFBIG_4e_PRE(6);
	TFBIG_4o_PRE(7);
	TFBIG_4e_PRE(8);
	TFBIG_4o_PRE(9);
	TFBIG_4e_PRE(10);
	TFBIG_4o_PRE(11);
	TFBIG_4e_PRE(12);
	TFBIG_4o_PRE(13);
	TFBIG_4e_PRE(14);
	TFBIG_4o_PRE(15);
	TFBIG_4e_PRE(16);
	TFBIG_4o_PRE(17);
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	uint64_t buffer[9];

	buffer[0] = PaddedMessage[0] ^ p[0];
	buffer[1] = PaddedMessage[1] ^ p[1];
	buffer[2] = PaddedMessage[2] ^ p[2];
	buffer[3] = PaddedMessage[3] ^ p[3];
	buffer[4] = PaddedMessage[4] ^ p[4];
	buffer[5] = PaddedMessage[5] ^ p[5];
	buffer[6] = PaddedMessage[6] ^ p[6];
	buffer[7] = PaddedMessage[7] ^ p[7];
	buffer[8] = t2;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(precalcvalues, buffer, sizeof(buffer), 0, cudaMemcpyHostToDevice));

	int endingTable[] = {
		0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
		0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
		0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549,
		0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
		0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7,
		0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
		0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484
	};

	int constantTable[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};
	for (int i = 0; i < 64; i++)
	{
		endingTable[i] = constantTable[i] + endingTable[i];
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(sha256_endingTable, endingTable, sizeof(uint32_t) * 64, 0, cudaMemcpyHostToDevice));

}



__host__
void skein512_cpu_setBlock_80(uint32_t thr_id, void *pdata)
{
	memcpy(&PaddedMessage[0], pdata, 80);

	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(c_PaddedMessage80, &PaddedMessage[8], 8*2, 0, cudaMemcpyHostToDevice)
	);
	CUDA_SAFE_CALL(cudaMalloc(&(d_found[thr_id]), 3 * sizeof(uint32_t)));

	precalc();
}

__host__
void skein512_cpu_hash_80_52(int thr_id, uint32_t threads, uint32_t startNounce, int swapu,uint64_t target, uint32_t *h_found)
{
	dim3 grid((threads + 1024 - 1) / 1024);
	dim3 block(1024);
	cudaMemset(d_found[thr_id], 0xffffffff, 2 * sizeof(uint32_t));
	skein512_gpu_hash_80_52 << < grid, block >> > (threads, startNounce, d_found[thr_id], target);
	cudaMemcpy(h_found, d_found[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
__host__
void skein512_cpu_hash_80_50(int thr_id, uint32_t threads, uint32_t startNounce, int swapu, uint64_t target, uint32_t *h_found)
{
	dim3 grid((threads + 256 - 1) / 256);
	dim3 block(256);
	cudaMemset(d_found[thr_id], 0xffffffff, 2 * sizeof(uint32_t));
	skein512_gpu_hash_80_50 << < grid, block >> > (threads, startNounce, d_found[thr_id], target);
	cudaMemcpy(h_found, d_found[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
