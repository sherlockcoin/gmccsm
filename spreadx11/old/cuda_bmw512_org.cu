#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"


// die Message it Padding zur Berechnung auf der GPU
__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

//#define SHL(x, n)            ((x) << (n))
//#define SHR(x, n)            ((x) >> (n))
#define SHR(x, n) SHR2(x, n) 
#define SHL(x, n) SHL2(x, n) 

#undef ROTL64
#define ROTL64 ROL2


#define CONST_EXP2(i)    q[i+0] + ROTL64(q[i+1], 5)  + q[i+2] + ROTL64(q[i+3], 11) + \
                    q[i+4] + ROTL64(q[i+5], 27) + q[i+6] + SWAPDWORDS2(q[i+7]) + \
                    q[i+8] + ROTL64(q[i+9], 37) + q[i+10] + ROTL64(q[i+11], 43) + \
                    q[i+12] + ROTL64(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])



__device__ __forceinline__ void Compression512(const uint2 *msg, uint2 *hash)
{

	const uint2 precalc[16] =
	{
		{ 0x55555550, 0x55555555 },
		{ 0xAAAAAAA5, 0x5AAAAAAA },
		{ 0xFFFFFFFA, 0x5FFFFFFF },
		{ 0x5555554F, 0x65555555 },
		{ 0xAAAAAAA4, 0x6AAAAAAA },
		{ 0xFFFFFFF9, 0x6FFFFFFF },
		{ 0x5555554E, 0x75555555 },
		{ 0xAAAAAAA3, 0x7AAAAAAA },
		{ 0xFFFFFFF8, 0x7FFFFFFF },
		{ 0x5555554D, 0x85555555 },
		{ 0xAAAAAAA2, 0x8AAAAAAA },
		{ 0xFFFFFFF7, 0x8FFFFFFF },
		{ 0x5555554C, 0x95555555 },
		{ 0xAAAAAAA1, 0x9AAAAAAA },
		{ 0xFFFFFFF6, 0x9FFFFFFF },
		{ 0x5555554B, 0xA5555555 },
	};


	// Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

    tmp = (msg[ 5] ^ hash[ 5]) - (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
    q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[1];
    tmp = (msg[ 6] ^ hash[ 6]) - (msg[ 8] ^ hash[ 8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
    q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[2];
    tmp = (msg[ 0] ^ hash[ 0]) + (msg[ 7] ^ hash[ 7]) + (msg[ 9] ^ hash[ 9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
    q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[3];
    tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 1] ^ hash[ 1]) + (msg[ 8] ^ hash[ 8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
    q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[4];
    tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 2] ^ hash[ 2]) + (msg[ 9] ^ hash[ 9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
    q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
    tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 2] ^ hash[ 2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
    q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[6];
    tmp = (msg[ 4] ^ hash[ 4]) - (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
    q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[7];
    tmp = (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 5] ^ hash[ 5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
    q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[8];
    tmp = (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) - (msg[ 6] ^ hash[ 6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
    q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[9];
    tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) + (msg[ 6] ^ hash[ 6]) - (msg[ 7] ^ hash[ 7]) + (msg[14] ^ hash[14]);
    q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
    tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 7] ^ hash[ 7]) + (msg[15] ^ hash[15]);
    q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[11];
    tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 0] ^ hash[ 0]) - (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) + (msg[ 9] ^ hash[ 9]);
    q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[12];
    tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 3] ^ hash[ 3]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[10] ^ hash[10]);
    q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[13];
    tmp = (msg[ 2] ^ hash[ 2]) + (msg[ 4] ^ hash[ 4]) + (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
    q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[14];
    tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 5] ^ hash[ 5]) + (msg[ 8] ^ hash[ 8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
    q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
    tmp = (msg[12] ^ hash[12]) - (msg[ 4] ^ hash[ 4]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[13] ^ hash[13]);
    q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[0];

        q[0+16] =
        (SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROTL64(q[0], 13) ^ ROTL64(q[0], 43)) +
        (SHR(q[0+1], 2) ^ SHL(q[0+1], 1) ^ ROTL64(q[0+1], 19) ^ ROTL64(q[0+1], 53)) +
        (SHR(q[0+2], 2) ^ SHL(q[0+2], 2) ^ ROTL64(q[0+2], 28) ^ ROTL64(q[0+2], 59)) +
        (SHR(q[0+3], 1) ^ SHL(q[0+3], 3) ^ ROTL64(q[0+3],  4) ^ ROTL64(q[0+3], 37)) +
        (SHR(q[0+4], 1) ^ SHL(q[0+4], 2) ^ ROTL64(q[0+4], 13) ^ ROTL64(q[0+4], 43)) +
        (SHR(q[0+5], 2) ^ SHL(q[0+5], 1) ^ ROTL64(q[0+5], 19) ^ ROTL64(q[0+5], 53)) +
        (SHR(q[0+6], 2) ^ SHL(q[0+6], 2) ^ ROTL64(q[0+6], 28) ^ ROTL64(q[0+6], 59)) +
        (SHR(q[0+7], 1) ^ SHL(q[0+7], 3) ^ ROTL64(q[0+7],  4) ^ ROTL64(q[0+7], 37)) +
        (SHR(q[0+8], 1) ^ SHL(q[0+8], 2) ^ ROTL64(q[0+8], 13) ^ ROTL64(q[0+8], 43)) +
        (SHR(q[0+9], 2) ^ SHL(q[0+9], 1) ^ ROTL64(q[0+9], 19) ^ ROTL64(q[0+9], 53)) +
        (SHR(q[0+10], 2) ^ SHL(q[0+10], 2) ^ ROTL64(q[0+10], 28) ^ ROTL64(q[0+10], 59)) +
        (SHR(q[0+11], 1) ^ SHL(q[0+11], 3) ^ ROTL64(q[0+11],  4) ^ ROTL64(q[0+11], 37)) +
        (SHR(q[0+12], 1) ^ SHL(q[0+12], 2) ^ ROTL64(q[0+12], 13) ^ ROTL64(q[0+12], 43)) +
        (SHR(q[0+13], 2) ^ SHL(q[0+13], 1) ^ ROTL64(q[0+13], 19) ^ ROTL64(q[0+13], 53)) +
        (SHR(q[0+14], 2) ^ SHL(q[0+14], 2) ^ ROTL64(q[0+14], 28) ^ ROTL64(q[0+14], 59)) +
        (SHR(q[0+15], 1) ^ SHL(q[0+15], 3) ^ ROTL64(q[0+15],  4) ^ ROTL64(q[0+15], 37)) +
		((precalc[0] + ROTL64(msg[0], 0 + 1) +
            ROTL64(msg[0+3], 0+4) - ROTL64(msg[0+10], 0+11) ) ^ hash[0+7]);
		q[1 + 16] =
			(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROTL64(q[1], 13) ^ ROTL64(q[1], 43)) +
			(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROTL64(q[1 + 1], 19) ^ ROTL64(q[1 + 1], 53)) +
			(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROTL64(q[1 + 2], 28) ^ ROTL64(q[1 + 2], 59)) +
			(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROTL64(q[1 + 3], 4) ^ ROTL64(q[1 + 3], 37)) +
			(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROTL64(q[1 + 4], 13) ^ ROTL64(q[1 + 4], 43)) +
			(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROTL64(q[1 + 5], 19) ^ ROTL64(q[1 + 5], 53)) +
			(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROTL64(q[1 + 6], 28) ^ ROTL64(q[1 + 6], 59)) +
			(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROTL64(q[1 + 7], 4) ^ ROTL64(q[1 + 7], 37)) +
			(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROTL64(q[1 + 8], 13) ^ ROTL64(q[1 + 8], 43)) +
			(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROTL64(q[1 + 9], 19) ^ ROTL64(q[1 + 9], 53)) +
			(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROTL64(q[1 + 10], 28) ^ ROTL64(q[1 + 10], 59)) +
			(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROTL64(q[1 + 11], 4) ^ ROTL64(q[1 + 11], 37)) +
			(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROTL64(q[1 + 12], 13) ^ ROTL64(q[1 + 12], 43)) +
			(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROTL64(q[1 + 13], 19) ^ ROTL64(q[1 + 13], 53)) +
			(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROTL64(q[1 + 14], 28) ^ ROTL64(q[1 + 14], 59)) +
			(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROTL64(q[1 + 15], 4) ^ ROTL64(q[1 + 15], 37)) +
			((precalc[1] + ROTL64(msg[1], 1 + 1) +
			ROTL64(msg[1 + 3], 1 + 4) - ROTL64(msg[1 + 10], 1 + 11)) ^ hash[1 + 7]);

		q[2 + 16] = CONST_EXP2(2) +
			((precalc[2] + ROTL64(msg[2], 2 + 1) +
            ROTL64(msg[2+3], 2+4) - ROTL64(msg[2+10], 2+11) ) ^ hash[2+7]);
		q[3 + 16] = CONST_EXP2(3) +
			((precalc[3] + ROTL64(msg[3], 3 + 1) +
			ROTL64(msg[3 + 3], 3 + 4) - ROTL64(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
		q[4 + 16] = CONST_EXP2(4) +
			((precalc[4] + ROTL64(msg[4], 4 + 1) +
			ROL8(msg[4 + 3]) - ROTL64(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
		q[5 + 16] = CONST_EXP2(5) +
			((precalc[5] + ROTL64(msg[5], 5 + 1) +
			ROTL64(msg[5 + 3], 5 + 4) - ROL16(msg[5 + 10])) ^ hash[5 + 7]);


		q[6 + 16] = CONST_EXP2(6) +
			((precalc[6]+ ROTL64(msg[6], 6 + 1) +
			ROTL64(msg[6 + 3], 6 + 4) - ROTL64(msg[6 - 6], (6 - 6) + 1)) ^ hash[6 + 7]);
		q[7 + 16] = CONST_EXP2(7) +
			((precalc[7] + ROL8(msg[7]) +
			ROTL64(msg[7 + 3], 7 + 4) - ROTL64(msg[7 - 6], (7 - 6) + 1)) ^ hash[7 + 7]);
		q[8 + 16] = CONST_EXP2(8) +
			((precalc[8] + ROTL64(msg[8], 8 + 1) +
			ROTL64(msg[8 + 3], 8 + 4) - ROTL64(msg[8 - 6], (8 - 6) + 1)) ^ hash[8 + 7]);

	q[9 + 16] = CONST_EXP2(9) +
	((precalc[9] + ROTL64(msg[9], 9 + 1) +
		ROTL64(msg[9 + 3], 9 + 4) - ROTL64(msg[9 - 6], (9 - 6) + 1)) ^ hash[9 - 9]);
	q[10 + 16] = CONST_EXP2(10) +
		((precalc[10] + ROTL64(msg[10], 10 + 1) +
		ROTL64(msg[10 + 3], 10 + 4) - ROTL64(msg[10 - 6], (10 - 6) + 1)) ^ hash[10 - 9]);
	q[11 + 16] = CONST_EXP2(11) +
		((precalc[11] + ROTL64(msg[11], 11 + 1) +
		ROTL64(msg[11 + 3], 11 + 4) - ROTL64(msg[11 - 6], (11 - 6) + 1)) ^ hash[11 - 9]);
	q[12 + 16] = CONST_EXP2(12) +
		((precalc[12] + ROTL64(msg[12], 12 + 1) +
		ROL16(msg[12 + 3]) - ROTL64(msg[12 - 6], (12 - 6) + 1)) ^ hash[12 - 9]);

	

		q[13 + 16] = CONST_EXP2(13) +
			((precalc[13] + ROTL64(msg[13], 13 + 1) +
			ROTL64(msg[13 - 13], (13 - 13) + 1) - ROL8(msg[13 - 6])) ^ hash[13 - 9]);
		q[14 + 16] = CONST_EXP2(14) +
			((precalc[14] + ROTL64(msg[14], 14 + 1) +
			ROTL64(msg[14 - 13], (14 - 13) + 1) - ROTL64(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
		q[15 + 16] = CONST_EXP2(15) +
			((precalc[15] + ROL16(msg[15]) +
			ROTL64(msg[15 - 13], (15 - 13) + 1) - ROTL64(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);

    uint2 XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

    hash[0] =                       (SHL(XH64, 5) ^ SHR(q[16],5) ^ msg[ 0]) + (    XL64    ^ q[24] ^ q[ 0]);
    hash[1] =                       (SHR(XH64, 7) ^ SHL(q[17],8) ^ msg[ 1]) + (    XL64    ^ q[25] ^ q[ 1]);
    hash[2] =                       (SHR(XH64, 5) ^ SHL(q[18],5) ^ msg[ 2]) + (    XL64    ^ q[26] ^ q[ 2]);
    hash[3] =                       (SHR(XH64, 1) ^ SHL(q[19],5) ^ msg[ 3]) + (    XL64    ^ q[27] ^ q[ 3]);
    hash[4] =                       (SHR(XH64, 3) ^     q[20]    ^ msg[ 4]) + (    XL64    ^ q[28] ^ q[ 4]);
    hash[5] =                       (SHL(XH64, 6) ^ SHR(q[21],6) ^ msg[ 5]) + (    XL64    ^ q[29] ^ q[ 5]);
    hash[6] =                       (SHR(XH64, 4) ^ SHL(q[22],6) ^ msg[ 6]) + (    XL64    ^ q[30] ^ q[ 6]);
    hash[7] =                       (SHR(XH64,11) ^ SHL(q[23],2) ^ msg[ 7]) + (    XL64    ^ q[31] ^ q[ 7]);

    hash[ 8] = ROTL64(hash[4], 9) + (    XH64     ^     q[24]    ^ msg[ 8]) + (SHL(XL64,8) ^ q[23] ^ q[ 8]);
    hash[ 9] = ROTL64(hash[5],10) + (    XH64     ^     q[25]    ^ msg[ 9]) + (SHR(XL64,6) ^ q[16] ^ q[ 9]);
    hash[10] = ROTL64(hash[6],11) + (    XH64     ^     q[26]    ^ msg[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
    hash[11] = ROTL64(hash[7],12) + (    XH64     ^     q[27]    ^ msg[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
    hash[12] = ROTL64(hash[0],13) + (    XH64     ^     q[28]    ^ msg[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
    hash[13] = ROTL64(hash[1],14) + (    XH64     ^     q[29]    ^ msg[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
    hash[14] = ROTL64(hash[2],15) + (    XH64     ^     q[30]    ^ msg[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
	hash[15] = ROL16(hash[3]) + (XH64     ^     q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}
__global__ __launch_bounds__(32, 16)
void quark_bmw512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector)
{
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
        const uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        const int hashPosition = nounce - startNounce;
        uint64_t *const inpHash = &g_hash[8 * hashPosition];

		const uint2 hash[16] = 
		{
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};

		const uint64_t hash2[16] =
		{
			 0x8081828384858687 ,
			 0x88898A8B8C8D8E8F ,
			 0x9091929394959697 ,
			 0x98999A9B9C9D9E9F ,
			 0xA0A1A2A3A4A5A6A7 ,
			 0xA8A9AAABACADAEAF ,
			 0xB0B1B2B3B4B5B6B7 ,
			 0xB8B9BABBBCBDBEBF ,
			 0xC0C1C2C3C4C5C6C7 ,
			 0xC8C9CACBCCCDCECF ,
			 0xD0D1D2D3D4D5D6D7 ,
			 0xD8D9DADBDCDDDEDF ,
			 0xE0E1E2E3E4E5E6E7 ,
			 0xE8E9EAEBECEDEEEF ,
			 0xF0F1F2F3F4F5F6F7 ,
			 0xF8F9FAFBFCFDFEFF
		};

		uint2 msg[16];
		uint2 mxh[8];
		uint2 h[16];
		msg[0] = vectorize(inpHash[0]);
		msg[1] = vectorize(inpHash[1]);
		msg[2] = vectorize(inpHash[2]);
		msg[3] = vectorize(inpHash[3]);
		msg[4] = vectorize(inpHash[4]);
		msg[5] = vectorize(inpHash[5]);
		msg[6] = vectorize(inpHash[6]);
		msg[7] = vectorize(inpHash[7]);
		msg[8] = vectorizelow(0x80);
		msg[15] = vectorizelow(512);
		mxh[0] = msg[0] ^ hash[0];
		mxh[1] = msg[1] ^ hash[1];
		mxh[2] = msg[2] ^ hash[2];
		mxh[3] = msg[3] ^ hash[3];
		mxh[4] = msg[4] ^ hash[4];
		mxh[5] = msg[5] ^ hash[5];
		mxh[6] = msg[6] ^ hash[6];
		mxh[7] = msg[7] ^ hash[7];

		const uint2 precalcf[9] =
		{
			{ 0x55555550ul, 0x55555555 },
			{ 0xAAAAAAA5, 0x5AAAAAAA },
			{ 0xFFFFFFFA, 0x5FFFFFFF },
			{ 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA },
			{ 0xFFFFFFF9, 0x6FFFFFFF },
			{ 0xAAAAAAA1, 0x9AAAAAAA },
			{ 0xFFFFFFF6, 0x9FFFFFFF },
			{ 0x5555554B, 0xA5555555 },
		};

		uint2 q[32];

		uint2 tmp;
		tmp = (mxh[5]) - (mxh[7]) + vectorize(hash2[10] + hash2[13] + hash2[14]);
		q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[1];
		tmp = (mxh[6]) + vectorize(hash2[11] + hash2[14] - (512 ^ hash2[15]) - (0x80^hash2[8]));
		q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[2];
		tmp = (mxh[0] + mxh[7]) + vectorize(hash2[9] - hash2[12] + (512 ^ hash2[15]));
		q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[3];
		tmp = (mxh[0] - mxh[1]) + vectorize((0x80 ^ hash2[8])- hash2[10] + hash2[13]);
		q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[4];
		tmp = (mxh[1] + mxh[2]) + vectorize(hash2[9] - hash2[11] - hash2[14]);
		q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
		tmp = (mxh[3] - mxh[2]) + vectorize(hash2[10] - hash2[12] + (512 ^ hash2[15]));
		q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[6];
		tmp = (mxh[4]) - (mxh[0]) - (mxh[3]) + vectorize(hash2[13] - hash2[11] );
		q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[7];
		tmp = (mxh[1]) - (mxh[4]) - (mxh[5]) + vectorize(-hash2[12] - hash2[14]);
		q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[8];
		tmp = (mxh[2]) - (mxh[5]) - (mxh[6]) + vectorize(hash2[13] - (512 ^ hash2[15]));
		q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[9];
		tmp = (mxh[0]) - (mxh[3]) + (mxh[6]) - (mxh[7]) + (hash[14]);
		q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
		tmp = vectorize((512 ^ hash2[15]) + (0x80 ^ hash2[8])) - (mxh[1]) - (mxh[4]) - (mxh[7]);
		q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[11];
		tmp = vectorize(hash2[9] + (0x80 ^ hash2[8])) - (mxh[0]) - (mxh[2]) - (mxh[5]);
		q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[12];
		tmp = (mxh[1]) + (mxh[3]) - (mxh[6]) + vectorize(hash2[10] - hash2[9]) ;
		q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[13];
		tmp = (mxh[2]) + (mxh[4]) + (mxh[7]) + vectorize(hash2[10] + hash2[11]);
		q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[14];
		tmp = (mxh[3]) - (mxh[5]) + vectorize((0x80 ^ hash2[8]) - hash2[11] - hash2[12]);
		q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
		tmp = vectorize(hash2[12] - hash2[9] + hash2[13]) - (mxh[4]) - (mxh[6]);
		q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[0];

		q[0 + 16] =
			(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROTL64(q[0], 13) ^ ROTL64(q[0], 43)) +
			(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROTL64(q[0 + 1], 19) ^ ROTL64(q[0 + 1], 53)) +
			(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROTL64(q[0 + 2], 28) ^ ROTL64(q[0 + 2], 59)) +
			(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROTL64(q[0 + 3], 4) ^ ROTL64(q[0 + 3], 37)) +
			(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROTL64(q[0 + 4], 13) ^ ROTL64(q[0 + 4], 43)) +
			(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROTL64(q[0 + 5], 19) ^ ROTL64(q[0 + 5], 53)) +
			(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROTL64(q[0 + 6], 28) ^ ROTL64(q[0 + 6], 59)) +
			(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROTL64(q[0 + 7], 4) ^ ROTL64(q[0 + 7], 37)) +
			(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROTL64(q[0 + 8], 13) ^ ROTL64(q[0 + 8], 43)) +
			(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROTL64(q[0 + 9], 19) ^ ROTL64(q[0 + 9], 53)) +
			(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROTL64(q[0 + 10], 28) ^ ROTL64(q[0 + 10], 59)) +
			(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROTL64(q[0 + 11], 4) ^ ROTL64(q[0 + 11], 37)) +
			(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROTL64(q[0 + 12], 13) ^ ROTL64(q[0 + 12], 43)) +
			(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROTL64(q[0 + 13], 19) ^ ROTL64(q[0 + 13], 53)) +
			(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROTL64(q[0 + 14], 28) ^ ROTL64(q[0 + 14], 59)) +
			(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROTL64(q[0 + 15], 4) ^ ROTL64(q[0 + 15], 37)) +
			((precalcf[0] + ROTL64(msg[0], 0 + 1) +
			ROTL64(msg[0 + 3], 0 + 4)) ^ hash[0 + 7]);
		q[1 + 16] =
			(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROTL64(q[1], 13) ^ ROTL64(q[1], 43)) +
			(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROTL64(q[1 + 1], 19) ^ ROTL64(q[1 + 1], 53)) +
			(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROTL64(q[1 + 2], 28) ^ ROTL64(q[1 + 2], 59)) +
			(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROTL64(q[1 + 3], 4) ^ ROTL64(q[1 + 3], 37)) +
			(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROTL64(q[1 + 4], 13) ^ ROTL64(q[1 + 4], 43)) +
			(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROTL64(q[1 + 5], 19) ^ ROTL64(q[1 + 5], 53)) +
			(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROTL64(q[1 + 6], 28) ^ ROTL64(q[1 + 6], 59)) +
			(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROTL64(q[1 + 7], 4) ^ ROTL64(q[1 + 7], 37)) +
			(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROTL64(q[1 + 8], 13) ^ ROTL64(q[1 + 8], 43)) +
			(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROTL64(q[1 + 9], 19) ^ ROTL64(q[1 + 9], 53)) +
			(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROTL64(q[1 + 10], 28) ^ ROTL64(q[1 + 10], 59)) +
			(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROTL64(q[1 + 11], 4) ^ ROTL64(q[1 + 11], 37)) +
			(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROTL64(q[1 + 12], 13) ^ ROTL64(q[1 + 12], 43)) +
			(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROTL64(q[1 + 13], 19) ^ ROTL64(q[1 + 13], 53)) +
			(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROTL64(q[1 + 14], 28) ^ ROTL64(q[1 + 14], 59)) +
			(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROTL64(q[1 + 15], 4) ^ ROTL64(q[1 + 15], 37)) +
			((precalcf[1] + ROTL64(msg[1], 1 + 1) +
			ROTL64(msg[1 + 3], 1 + 4)) ^ hash[1 + 7]);

		q[2 + 16] = CONST_EXP2(2) +
			((precalcf[2] + ROTL64(msg[2], 2 + 1) +
			ROTL64(msg[2 + 3], 2 + 4)) ^ hash[2 + 7]);
		q[3 + 16] = CONST_EXP2(3) +
			((precalcf[3] + ROTL64(msg[3], 3 + 1) +
			ROTL64(msg[3 + 3], 3 + 4)) ^ hash[3 + 7]);
		q[4 + 16] = CONST_EXP2(4) +
			((precalcf[4] + ROTL64(msg[4], 4 + 1) +
			ROL8(msg[4 + 3])) ^ hash[4 + 7]);
		q[5 + 16] = CONST_EXP2(5) +
			((precalcf[5] + ROTL64(msg[5], 5 + 1) +
			ROTL64(msg[5 + 3], 5 + 4) - ROL16(msg[5 + 10])) ^ hash[5 + 7]);


		//#pragma unroll 3
		for (int i = 6; i < 9; i++) {
			q[i + 16] = CONST_EXP2(i) +
				((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) -
				ROTL64(msg[i - 6], (i - 6) + 1)) ^ hash[i + 7]);
		}

		q[25] = CONST_EXP2(9) +
			((vectorize((25)*(0x0555555555555555ull)) - ROTL64(msg[3], 4)) ^ hash[0]);
		q[26] = CONST_EXP2(10) +
			((vectorize((26)*(0x0555555555555555ull)) - ROTL64(msg[4], 5)) ^ hash[1]);
		q[27] = CONST_EXP2(11) +
			((vectorize((27)*(0x0555555555555555ull)) - ROTL64(msg[5], 6)) ^ hash[2]);
		q[28] = CONST_EXP2(12) +
			((vectorize((28)*(0x0555555555555555ull)) +	ROL16(msg[15]) - ROTL64(msg[6], 7)) ^ hash[3]);

		q[13 + 16] = CONST_EXP2(13) +
			((precalcf[6] + 
			ROTL64(msg[13 - 13], (13 - 13) + 1) - ROL8(msg[13 - 6])) ^ hash[13 - 9]);
		q[14 + 16] = CONST_EXP2(14) +
			((precalcf[7] + 
			ROTL64(msg[14 - 13], (14 - 13) + 1) - ROTL64(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
		q[15 + 16] = CONST_EXP2(15) +
			((precalcf[8] + ROL16(msg[15]) +
			ROTL64(msg[15 - 13], (15 - 13) + 1)) ^ hash[15 - 9]);

		uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
		uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

		h[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ msg[0]) + (XL64    ^ q[24] ^ q[0]);
		h[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ msg[1]) + (XL64    ^ q[25] ^ q[1]);
		h[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ msg[2]) + (XL64    ^ q[26] ^ q[2]);
		h[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ msg[3]) + (XL64    ^ q[27] ^ q[3]);
		h[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + (XL64    ^ q[28] ^ q[4]);
		h[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ msg[5]) + (XL64    ^ q[29] ^ q[5]);
		h[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ msg[6]) + (XL64    ^ q[30] ^ q[6]);
		h[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ msg[7]) + (XL64    ^ q[31] ^ q[7]);

		h[8] = ROTL64(h[4], 9) + (XH64     ^     q[24] ^ msg[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);
		h[9] = ROTL64(h[5], 10) + (XH64     ^     q[25]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
		h[10] = ROTL64(h[6], 11) + (XH64     ^     q[26]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
		h[11] = ROTL64(h[7], 12) + (XH64     ^     q[27]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
		h[12] = ROTL64(h[0], 13) + (XH64     ^     q[28]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
		h[13] = ROTL64(h[1], 14) + (XH64     ^     q[29]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
		h[14] = ROTL64(h[2], 15) + (XH64     ^     q[30]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
		h[15] = ROL16(h[3]) + (XH64     ^     q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);

		const uint2 cmsg[16] =
		{
			0xaaaaaaa0, 0xaaaaaaaa,
			0xaaaaaaa1, 0xaaaaaaaa,
			0xaaaaaaa2, 0xaaaaaaaa,
			0xaaaaaaa3, 0xaaaaaaaa,
			0xaaaaaaa4, 0xaaaaaaaa,
			0xaaaaaaa5, 0xaaaaaaaa,
			0xaaaaaaa6, 0xaaaaaaaa,
			0xaaaaaaa7, 0xaaaaaaaa,
			0xaaaaaaa8, 0xaaaaaaaa,
			0xaaaaaaa9, 0xaaaaaaaa,
			0xaaaaaaaa, 0xaaaaaaaa,
			0xaaaaaaab, 0xaaaaaaaa,
			0xaaaaaaac, 0xaaaaaaaa,
			0xaaaaaaad, 0xaaaaaaaa,
			0xaaaaaaae, 0xaaaaaaaa,
			0xaaaaaaaf, 0xaaaaaaaa
		};

#pragma unroll 16
		for (int i = 0; i < 16; i++)
		{
			msg[i] = cmsg[i] ^ h[i];
		}


		const uint2 precalc[16] =
		{
			{ 0x55555550, 0x55555555 },
			{ 0xAAAAAAA5, 0x5AAAAAAA },
			{ 0xFFFFFFFA, 0x5FFFFFFF },
			{ 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA },
			{ 0xFFFFFFF9, 0x6FFFFFFF },
			{ 0x5555554E, 0x75555555 },
			{ 0xAAAAAAA3, 0x7AAAAAAA },
			{ 0xFFFFFFF8, 0x7FFFFFFF },
			{ 0x5555554D, 0x85555555 },
			{ 0xAAAAAAA2, 0x8AAAAAAA },
			{ 0xFFFFFFF7, 0x8FFFFFFF },
			{ 0x5555554C, 0x95555555 },
			{ 0xAAAAAAA1, 0x9AAAAAAA },
			{ 0xFFFFFFF6, 0x9FFFFFFF },
			{ 0x5555554B, 0xA5555555 },
		};

		tmp = (msg[5]) - (msg[7]) + (msg[10]) + (msg[13]) + (msg[14]);
		q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + cmsg[1];
		tmp = (msg[6]) - (msg[8]) + (msg[11]) + (msg[14]) - (msg[15]);
		q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + cmsg[2];
		tmp = (msg[0]) + (msg[7]) + (msg[9]) - (msg[12]) + (msg[15]);
		q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + cmsg[3];
		tmp = (msg[0]) - (msg[1]) + (msg[8]) - (msg[10]) + (msg[13]);
		q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + cmsg[4];
		tmp = (msg[1]) + (msg[2]) + (msg[9]) - (msg[11]) - (msg[14]);
		q[4] = (SHR(tmp, 1) ^ tmp) + cmsg[5];
		tmp = (msg[3]) - (msg[2]) + (msg[10]) - (msg[12]) + (msg[15]);
		q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + cmsg[6];
		tmp = (msg[4]) - (msg[0]) - (msg[3]) - (msg[11]) + (msg[13]);
		q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + cmsg[7];
		tmp = (msg[1]) - (msg[4]) - (msg[5]) - (msg[12]) - (msg[14]);
		q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + cmsg[8];
		tmp = (msg[2]) - (msg[5]) - (msg[6]) + (msg[13]) - (msg[15]);
		q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + cmsg[9];
		tmp = (msg[0]) - (msg[3]) + (msg[6]) - (msg[7]) + (msg[14]);
		q[9] = (SHR(tmp, 1) ^ tmp) + cmsg[10];
		tmp = (msg[8]) - (msg[1]) - (msg[4]) - (msg[7]) + (msg[15]);
		q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + cmsg[11];
		tmp = (msg[8]) - (msg[0]) - (msg[2]) - (msg[5]) + (msg[9]);
		q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + cmsg[12];
		tmp = (msg[1]) + (msg[3]) - (msg[6]) - (msg[9]) + (msg[10]);
		q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + cmsg[13];
		tmp = (msg[2]) + (msg[4]) + (msg[7]) + (msg[10]) + (msg[11]);
		q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + cmsg[14];
		tmp = (msg[3]) - (msg[5]) + (msg[8]) - (msg[11]) - (msg[12]);
		q[14] = (SHR(tmp, 1) ^ tmp) + cmsg[15];
		tmp = (msg[12]) - (msg[4]) - (msg[6]) - (msg[9]) + (msg[13]);
		q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + cmsg[0];

		q[0 + 16] =
			(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROTL64(q[0], 13) ^ ROTL64(q[0], 43)) +
			(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROTL64(q[0 + 1], 19) ^ ROTL64(q[0 + 1], 53)) +
			(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROTL64(q[0 + 2], 28) ^ ROTL64(q[0 + 2], 59)) +
			(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROTL64(q[0 + 3], 4) ^ ROTL64(q[0 + 3], 37)) +
			(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROTL64(q[0 + 4], 13) ^ ROTL64(q[0 + 4], 43)) +
			(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROTL64(q[0 + 5], 19) ^ ROTL64(q[0 + 5], 53)) +
			(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROTL64(q[0 + 6], 28) ^ ROTL64(q[0 + 6], 59)) +
			(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROTL64(q[0 + 7], 4) ^ ROTL64(q[0 + 7], 37)) +
			(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROTL64(q[0 + 8], 13) ^ ROTL64(q[0 + 8], 43)) +
			(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROTL64(q[0 + 9], 19) ^ ROTL64(q[0 + 9], 53)) +
			(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROTL64(q[0 + 10], 28) ^ ROTL64(q[0 + 10], 59)) +
			(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROTL64(q[0 + 11], 4) ^ ROTL64(q[0 + 11], 37)) +
			(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROTL64(q[0 + 12], 13) ^ ROTL64(q[0 + 12], 43)) +
			(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROTL64(q[0 + 13], 19) ^ ROTL64(q[0 + 13], 53)) +
			(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROTL64(q[0 + 14], 28) ^ ROTL64(q[0 + 14], 59)) +
			(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROTL64(q[0 + 15], 4) ^ ROTL64(q[0 + 15], 37)) +
			((precalc[0] + ROTL64(h[0], 0 + 1) +
			ROTL64(h[0 + 3], 0 + 4) - ROTL64(h[0 + 10], 0 + 11)) ^ cmsg[0 + 7]);
		q[1 + 16] =
			(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROTL64(q[1], 13) ^ ROTL64(q[1], 43)) +
			(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROTL64(q[1 + 1], 19) ^ ROTL64(q[1 + 1], 53)) +
			(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROTL64(q[1 + 2], 28) ^ ROTL64(q[1 + 2], 59)) +
			(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROTL64(q[1 + 3], 4) ^ ROTL64(q[1 + 3], 37)) +
			(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROTL64(q[1 + 4], 13) ^ ROTL64(q[1 + 4], 43)) +
			(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROTL64(q[1 + 5], 19) ^ ROTL64(q[1 + 5], 53)) +
			(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROTL64(q[1 + 6], 28) ^ ROTL64(q[1 + 6], 59)) +
			(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROTL64(q[1 + 7], 4) ^ ROTL64(q[1 + 7], 37)) +
			(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROTL64(q[1 + 8], 13) ^ ROTL64(q[1 + 8], 43)) +
			(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROTL64(q[1 + 9], 19) ^ ROTL64(q[1 + 9], 53)) +
			(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROTL64(q[1 + 10], 28) ^ ROTL64(q[1 + 10], 59)) +
			(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROTL64(q[1 + 11], 4) ^ ROTL64(q[1 + 11], 37)) +
			(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROTL64(q[1 + 12], 13) ^ ROTL64(q[1 + 12], 43)) +
			(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROTL64(q[1 + 13], 19) ^ ROTL64(q[1 + 13], 53)) +
			(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROTL64(q[1 + 14], 28) ^ ROTL64(q[1 + 14], 59)) +
			(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROTL64(q[1 + 15], 4) ^ ROTL64(q[1 + 15], 37)) +
			((precalc[1] + ROTL64(h[1], 1 + 1) +
			ROTL64(h[1 + 3], 1 + 4) - ROTL64(h[1 + 10], 1 + 11)) ^ cmsg[1 + 7]);

		q[2 + 16] = CONST_EXP2(2) +
			((precalc[2] + ROTL64(h[2], 2 + 1) +
			ROTL64(h[2 + 3], 2 + 4) - ROTL64(h[2 + 10], 2 + 11)) ^ cmsg[2 + 7]);
		q[3 + 16] = CONST_EXP2(3) +
			((precalc[3] + ROTL64(h[3], 3 + 1) +
			ROTL64(h[3 + 3], 3 + 4) - ROTL64(h[3 + 10], 3 + 11)) ^ cmsg[3 + 7]);
		q[4 + 16] = CONST_EXP2(4) +
			((precalc[4] + ROTL64(h[4], 4 + 1) +
			ROL8(h[4 + 3]) - ROTL64(h[4 + 10], 4 + 11)) ^ cmsg[4 + 7]);
		q[5 + 16] = CONST_EXP2(5) +
			((precalc[5] + ROTL64(h[5], 5 + 1) +
			ROTL64(h[5 + 3], 5 + 4) - ROL16(h[5 + 10])) ^ cmsg[5 + 7]);


		q[6 + 16] = CONST_EXP2(6) +
			((precalc[6] + ROTL64(h[6], 6 + 1) +
			ROTL64(h[6 + 3], 6 + 4) - ROTL64(h[6 - 6], (6 - 6) + 1)) ^ cmsg[6 + 7]);
		q[7 + 16] = CONST_EXP2(7) +
			((precalc[7] + ROL8(h[7]) +
			ROTL64(h[7 + 3], 7 + 4) - ROTL64(h[7 - 6], (7 - 6) + 1)) ^ cmsg[7 + 7]);
		q[8 + 16] = CONST_EXP2(8) +
			((precalc[8] + ROTL64(h[8], 8 + 1) +
			ROTL64(h[8 + 3], 8 + 4) - ROTL64(h[8 - 6], (8 - 6) + 1)) ^ cmsg[8 + 7]);

		q[9 + 16] = CONST_EXP2(9) +
			((precalc[9] + ROTL64(h[9], 9 + 1) +
			ROTL64(h[9 + 3], 9 + 4) - ROTL64(h[9 - 6], (9 - 6) + 1)) ^ cmsg[9 - 9]);
		q[10 + 16] = CONST_EXP2(10) +
			((precalc[10] + ROTL64(h[10], 10 + 1) +
			ROTL64(h[10 + 3], 10 + 4) - ROTL64(h[10 - 6], (10 - 6) + 1)) ^ cmsg[10 - 9]);
		q[11 + 16] = CONST_EXP2(11) +
			((precalc[11] + ROTL64(h[11], 11 + 1) +
			ROTL64(h[11 + 3], 11 + 4) - ROTL64(h[11 - 6], (11 - 6) + 1)) ^ cmsg[11 - 9]);
		q[12 + 16] = CONST_EXP2(12) +
			((precalc[12] + ROTL64(h[12], 12 + 1) +
			ROL16(h[12 + 3]) - ROTL64(h[12 - 6], (12 - 6) + 1)) ^ cmsg[12 - 9]);



		q[13 + 16] = CONST_EXP2(13) +
			((precalc[13] + ROTL64(h[13], 13 + 1) +
			ROTL64(h[13 - 13], (13 - 13) + 1) - ROL8(h[13 - 6])) ^ cmsg[13 - 9]);
		q[14 + 16] = CONST_EXP2(14) +
			((precalc[14] + ROTL64(h[14], 14 + 1) +
			ROTL64(h[14 - 13], (14 - 13) + 1) - ROTL64(h[14 - 6], (14 - 6) + 1)) ^ cmsg[14 - 9]);
		q[15 + 16] = CONST_EXP2(15) +
			((precalc[15] + ROL16(h[15]) +
			ROTL64(h[15 - 13], (15 - 13) + 1) - ROTL64(h[15 - 6], (15 - 6) + 1)) ^ cmsg[15 - 9]);

		XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
		XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

		msg[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ h[0]) + (XL64    ^ q[24] ^ q[0]);
		msg[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ h[1]) + (XL64    ^ q[25] ^ q[1]);
		msg[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ h[2]) + (XL64    ^ q[26] ^ q[2]);
		msg[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ h[3]) + (XL64    ^ q[27] ^ q[3]);
		msg[4] = (SHR(XH64, 3) ^ q[20] ^ h[4]) + (XL64    ^ q[28] ^ q[4]);
		msg[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ h[5]) + (XL64    ^ q[29] ^ q[5]);
		msg[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ h[6]) + (XL64    ^ q[30] ^ q[6]);
		msg[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ h[7]) + (XL64    ^ q[31] ^ q[7]);
		msg[8] = ROTL64(msg[4], 9) + (XH64     ^     q[24] ^ h[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);

		msg[9] = ROTL64(msg[5], 10) + (XH64     ^     q[25] ^ h[9]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
		msg[10] = ROTL64(msg[6], 11) + (XH64     ^     q[26] ^ h[10]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
		msg[11] = ROTL64(msg[7], 12) + (XH64     ^     q[27] ^ h[11]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
		msg[12] = ROTL64(msg[0], 13) + (XH64     ^     q[28] ^ h[12]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
		msg[13] = ROTL64(msg[1], 14) + (XH64     ^     q[29] ^ h[13]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
		msg[14] = ROTL64(msg[2], 15) + (XH64     ^     q[30] ^ h[14]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
		msg[15] = ROL16(msg[3]) + (XH64     ^     q[31] ^ h[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);

		inpHash[0] = devectorize(msg[0 + 8]);
		inpHash[1] = devectorize(msg[1 + 8]);
		inpHash[2] = devectorize(msg[2 + 8]);
		inpHash[3] = devectorize(msg[3 + 8]);
		inpHash[4] = devectorize(msg[4 + 8]);
		inpHash[5] = devectorize(msg[5 + 8]);
		inpHash[6] = devectorize(msg[6 + 8]);
		inpHash[7] = devectorize(msg[7 + 8]);
	}
}

__global__ __launch_bounds__(32, 16)
void quark_bmw512_gpu_hash_64_quark(uint32_t threads, uint32_t startNounce, uint64_t *const __restrict__ g_hash, uint32_t *g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (startNounce + thread);

		const int hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[8 * hashPosition];

		const uint2 hash[16] =
		{
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};

		const uint64_t hash2[16] =
		{
			0x8081828384858687,
			0x88898A8B8C8D8E8F,
			0x9091929394959697,
			0x98999A9B9C9D9E9F,
			0xA0A1A2A3A4A5A6A7,
			0xA8A9AAABACADAEAF,
			0xB0B1B2B3B4B5B6B7,
			0xB8B9BABBBCBDBEBF,
			0xC0C1C2C3C4C5C6C7,
			0xC8C9CACBCCCDCECF,
			0xD0D1D2D3D4D5D6D7,
			0xD8D9DADBDCDDDEDF,
			0xE0E1E2E3E4E5E6E7,
			0xE8E9EAEBECEDEEEF,
			0xF0F1F2F3F4F5F6F7,
			0xF8F9FAFBFCFDFEFF
		};

		uint2 msg[16];
		uint2 mxh[8];
		uint2 h[16];
		msg[0] = vectorize(inpHash[0]);
		msg[1] = vectorize(inpHash[1]);
		msg[2] = vectorize(inpHash[2]);
		msg[3] = vectorize(inpHash[3]);
		msg[4] = vectorize(inpHash[4]);
		msg[5] = vectorize(inpHash[5]);
		msg[6] = vectorize(inpHash[6]);
		msg[7] = vectorize(inpHash[7]);
		msg[8] = vectorizelow(0x80);
		msg[15] = vectorizelow(512);
		mxh[0] = msg[0] ^ hash[0];
		mxh[1] = msg[1] ^ hash[1];
		mxh[2] = msg[2] ^ hash[2];
		mxh[3] = msg[3] ^ hash[3];
		mxh[4] = msg[4] ^ hash[4];
		mxh[5] = msg[5] ^ hash[5];
		mxh[6] = msg[6] ^ hash[6];
		mxh[7] = msg[7] ^ hash[7];

		const uint2 precalcf[9] =
		{
			{ 0x55555550ul, 0x55555555 },
			{ 0xAAAAAAA5, 0x5AAAAAAA },
			{ 0xFFFFFFFA, 0x5FFFFFFF },
			{ 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA },
			{ 0xFFFFFFF9, 0x6FFFFFFF },
			{ 0xAAAAAAA1, 0x9AAAAAAA },
			{ 0xFFFFFFF6, 0x9FFFFFFF },
			{ 0x5555554B, 0xA5555555 },
		};

		uint2 q[32];

		uint2 tmp;
		tmp = (mxh[5]) - (mxh[7]) + vectorize(hash2[10] + hash2[13] + hash2[14]);
		q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[1];
		tmp = (mxh[6]) + vectorize(hash2[11] + hash2[14] - (512 ^ hash2[15]) - (0x80 ^ hash2[8]));
		q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[2];
		tmp = (mxh[0] + mxh[7]) + vectorize(hash2[9] - hash2[12] + (512 ^ hash2[15]));
		q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[3];
		tmp = (mxh[0] - mxh[1]) + vectorize((0x80 ^ hash2[8]) - hash2[10] + hash2[13]);
		q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[4];
		tmp = (mxh[1] + mxh[2]) + vectorize(hash2[9] - hash2[11] - hash2[14]);
		q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
		tmp = (mxh[3] - mxh[2]) + vectorize(hash2[10] - hash2[12] + (512 ^ hash2[15]));
		q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[6];
		tmp = (mxh[4]) - (mxh[0]) - (mxh[3]) + vectorize(hash2[13] - hash2[11]);
		q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[7];
		tmp = (mxh[1]) - (mxh[4]) - (mxh[5]) + vectorize(-hash2[12] - hash2[14]);
		q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[8];
		tmp = (mxh[2]) - (mxh[5]) - (mxh[6]) + vectorize(hash2[13] - (512 ^ hash2[15]));
		q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[9];
		tmp = (mxh[0]) - (mxh[3]) + (mxh[6]) - (mxh[7]) + (hash[14]);
		q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
		tmp = vectorize((512 ^ hash2[15]) + (0x80 ^ hash2[8])) - (mxh[1]) - (mxh[4]) - (mxh[7]);
		q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[11];
		tmp = vectorize(hash2[9] + (0x80 ^ hash2[8])) - (mxh[0]) - (mxh[2]) - (mxh[5]);
		q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[12];
		tmp = (mxh[1]) + (mxh[3]) - (mxh[6]) + vectorize(hash2[10] - hash2[9]);
		q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[13];
		tmp = (mxh[2]) + (mxh[4]) + (mxh[7]) + vectorize(hash2[10] + hash2[11]);
		q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[14];
		tmp = (mxh[3]) - (mxh[5]) + vectorize((0x80 ^ hash2[8]) - hash2[11] - hash2[12]);
		q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
		tmp = vectorize(hash2[12] - hash2[9] + hash2[13]) - (mxh[4]) - (mxh[6]);
		q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[0];

		q[0 + 16] =
			(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROTL64(q[0], 13) ^ ROTL64(q[0], 43)) +
			(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROTL64(q[0 + 1], 19) ^ ROTL64(q[0 + 1], 53)) +
			(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROTL64(q[0 + 2], 28) ^ ROTL64(q[0 + 2], 59)) +
			(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROTL64(q[0 + 3], 4) ^ ROTL64(q[0 + 3], 37)) +
			(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROTL64(q[0 + 4], 13) ^ ROTL64(q[0 + 4], 43)) +
			(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROTL64(q[0 + 5], 19) ^ ROTL64(q[0 + 5], 53)) +
			(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROTL64(q[0 + 6], 28) ^ ROTL64(q[0 + 6], 59)) +
			(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROTL64(q[0 + 7], 4) ^ ROTL64(q[0 + 7], 37)) +
			(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROTL64(q[0 + 8], 13) ^ ROTL64(q[0 + 8], 43)) +
			(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROTL64(q[0 + 9], 19) ^ ROTL64(q[0 + 9], 53)) +
			(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROTL64(q[0 + 10], 28) ^ ROTL64(q[0 + 10], 59)) +
			(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROTL64(q[0 + 11], 4) ^ ROTL64(q[0 + 11], 37)) +
			(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROTL64(q[0 + 12], 13) ^ ROTL64(q[0 + 12], 43)) +
			(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROTL64(q[0 + 13], 19) ^ ROTL64(q[0 + 13], 53)) +
			(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROTL64(q[0 + 14], 28) ^ ROTL64(q[0 + 14], 59)) +
			(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROTL64(q[0 + 15], 4) ^ ROTL64(q[0 + 15], 37)) +
			((precalcf[0] + ROTL64(msg[0], 0 + 1) +
			ROTL64(msg[0 + 3], 0 + 4)) ^ hash[0 + 7]);
		q[1 + 16] =
			(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROTL64(q[1], 13) ^ ROTL64(q[1], 43)) +
			(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROTL64(q[1 + 1], 19) ^ ROTL64(q[1 + 1], 53)) +
			(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROTL64(q[1 + 2], 28) ^ ROTL64(q[1 + 2], 59)) +
			(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROTL64(q[1 + 3], 4) ^ ROTL64(q[1 + 3], 37)) +
			(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROTL64(q[1 + 4], 13) ^ ROTL64(q[1 + 4], 43)) +
			(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROTL64(q[1 + 5], 19) ^ ROTL64(q[1 + 5], 53)) +
			(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROTL64(q[1 + 6], 28) ^ ROTL64(q[1 + 6], 59)) +
			(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROTL64(q[1 + 7], 4) ^ ROTL64(q[1 + 7], 37)) +
			(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROTL64(q[1 + 8], 13) ^ ROTL64(q[1 + 8], 43)) +
			(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROTL64(q[1 + 9], 19) ^ ROTL64(q[1 + 9], 53)) +
			(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROTL64(q[1 + 10], 28) ^ ROTL64(q[1 + 10], 59)) +
			(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROTL64(q[1 + 11], 4) ^ ROTL64(q[1 + 11], 37)) +
			(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROTL64(q[1 + 12], 13) ^ ROTL64(q[1 + 12], 43)) +
			(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROTL64(q[1 + 13], 19) ^ ROTL64(q[1 + 13], 53)) +
			(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROTL64(q[1 + 14], 28) ^ ROTL64(q[1 + 14], 59)) +
			(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROTL64(q[1 + 15], 4) ^ ROTL64(q[1 + 15], 37)) +
			((precalcf[1] + ROTL64(msg[1], 1 + 1) +
			ROTL64(msg[1 + 3], 1 + 4)) ^ hash[1 + 7]);

		q[2 + 16] = CONST_EXP2(2) +
			((precalcf[2] + ROTL64(msg[2], 2 + 1) +
			ROTL64(msg[2 + 3], 2 + 4)) ^ hash[2 + 7]);
		q[3 + 16] = CONST_EXP2(3) +
			((precalcf[3] + ROTL64(msg[3], 3 + 1) +
			ROTL64(msg[3 + 3], 3 + 4)) ^ hash[3 + 7]);
		q[4 + 16] = CONST_EXP2(4) +
			((precalcf[4] + ROTL64(msg[4], 4 + 1) +
			ROL8(msg[4 + 3])) ^ hash[4 + 7]);
		q[5 + 16] = CONST_EXP2(5) +
			((precalcf[5] + ROTL64(msg[5], 5 + 1) +
			ROTL64(msg[5 + 3], 5 + 4) - ROL16(msg[5 + 10])) ^ hash[5 + 7]);


		//#pragma unroll 3
		for (int i = 6; i < 9; i++) {
			q[i + 16] = CONST_EXP2(i) +
				((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) -
				ROTL64(msg[i - 6], (i - 6) + 1)) ^ hash[i + 7]);
		}

		q[25] = CONST_EXP2(9) +
			((vectorize((25)*(0x0555555555555555ull)) - ROTL64(msg[3], 4)) ^ hash[0]);
		q[26] = CONST_EXP2(10) +
			((vectorize((26)*(0x0555555555555555ull)) - ROTL64(msg[4], 5)) ^ hash[1]);
		q[27] = CONST_EXP2(11) +
			((vectorize((27)*(0x0555555555555555ull)) - ROTL64(msg[5], 6)) ^ hash[2]);
		q[28] = CONST_EXP2(12) +
			((vectorize((28)*(0x0555555555555555ull)) + ROL16(msg[15]) - ROTL64(msg[6], 7)) ^ hash[3]);

		q[13 + 16] = CONST_EXP2(13) +
			((precalcf[6] +
			ROTL64(msg[13 - 13], (13 - 13) + 1) - ROL8(msg[13 - 6])) ^ hash[13 - 9]);
		q[14 + 16] = CONST_EXP2(14) +
			((precalcf[7] +
			ROTL64(msg[14 - 13], (14 - 13) + 1) - ROTL64(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
		q[15 + 16] = CONST_EXP2(15) +
			((precalcf[8] + ROL16(msg[15]) +
			ROTL64(msg[15 - 13], (15 - 13) + 1)) ^ hash[15 - 9]);

		uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
		uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

		uint2 test = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ msg[0]) + (XL64    ^ q[24] ^ q[0]);

		h[0] = test;
		h[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ msg[1]) + (XL64    ^ q[25] ^ q[1]);
		h[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ msg[2]) + (XL64    ^ q[26] ^ q[2]);
		h[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ msg[3]) + (XL64    ^ q[27] ^ q[3]);
		h[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + (XL64    ^ q[28] ^ q[4]);
		h[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ msg[5]) + (XL64    ^ q[29] ^ q[5]);
		h[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ msg[6]) + (XL64    ^ q[30] ^ q[6]);
		h[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ msg[7]) + (XL64    ^ q[31] ^ q[7]);

		h[8] = ROTL64(h[4], 9) + (XH64     ^     q[24] ^ msg[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);
		h[9] = ROTL64(h[5], 10) + (XH64     ^     q[25]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
		h[10] = ROTL64(h[6], 11) + (XH64     ^     q[26]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
		h[11] = ROTL64(h[7], 12) + (XH64     ^     q[27]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
		h[12] = ROTL64(h[0], 13) + (XH64     ^     q[28]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
		h[13] = ROTL64(h[1], 14) + (XH64     ^     q[29]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
		h[14] = ROTL64(h[2], 15) + (XH64     ^     q[30]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
		h[15] = ROL16(h[3]) + (XH64     ^     q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);

		const uint2 cmsg[16] =
		{
			0xaaaaaaa0, 0xaaaaaaaa,
			0xaaaaaaa1, 0xaaaaaaaa,
			0xaaaaaaa2, 0xaaaaaaaa,
			0xaaaaaaa3, 0xaaaaaaaa,
			0xaaaaaaa4, 0xaaaaaaaa,
			0xaaaaaaa5, 0xaaaaaaaa,
			0xaaaaaaa6, 0xaaaaaaaa,
			0xaaaaaaa7, 0xaaaaaaaa,
			0xaaaaaaa8, 0xaaaaaaaa,
			0xaaaaaaa9, 0xaaaaaaaa,
			0xaaaaaaaa, 0xaaaaaaaa,
			0xaaaaaaab, 0xaaaaaaaa,
			0xaaaaaaac, 0xaaaaaaaa,
			0xaaaaaaad, 0xaaaaaaaa,
			0xaaaaaaae, 0xaaaaaaaa,
			0xaaaaaaaf, 0xaaaaaaaa
		};

		// Final
#pragma unroll 16
		for (int i = 0; i < 16; i++)
		{
			msg[i] = cmsg[i]^h[i];
		}



		const uint2 precalc[16] =
		{
			{ 0x55555550, 0x55555555 },
			{ 0xAAAAAAA5, 0x5AAAAAAA },
			{ 0xFFFFFFFA, 0x5FFFFFFF },
			{ 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA },
			{ 0xFFFFFFF9, 0x6FFFFFFF },
			{ 0x5555554E, 0x75555555 },
			{ 0xAAAAAAA3, 0x7AAAAAAA },
			{ 0xFFFFFFF8, 0x7FFFFFFF },
			{ 0x5555554D, 0x85555555 },
			{ 0xAAAAAAA2, 0x8AAAAAAA },
			{ 0xFFFFFFF7, 0x8FFFFFFF },
			{ 0x5555554C, 0x95555555 },
			{ 0xAAAAAAA1, 0x9AAAAAAA },
			{ 0xFFFFFFF6, 0x9FFFFFFF },
			{ 0x5555554B, 0xA5555555 },
		};

		tmp = (msg[5]) - (msg[7]) + (msg[10]) + (msg[13]) + (msg[14]);
		q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + cmsg[1];
		tmp = (msg[6]) - (msg[8]) + (msg[11]) + (msg[14]) - (msg[15]);
		q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + cmsg[2];
		tmp = (msg[0]) + (msg[7]) + (msg[9]) - (msg[12]) + (msg[15]);
		q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + cmsg[3];
		tmp = (msg[0]) - (msg[1]) + (msg[8]) - (msg[10]) + (msg[13]);
		q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + cmsg[4];
		tmp = (msg[1]) + (msg[2]) + (msg[9]) - (msg[11]) - (msg[14]);
		q[4] = (SHR(tmp, 1) ^ tmp) + cmsg[5];
		tmp = (msg[3]) - (msg[2]) + (msg[10]) - (msg[12]) + (msg[15]);
		q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + cmsg[6];
		tmp = (msg[4]) - (msg[0]) - (msg[3]) - (msg[11]) + (msg[13]);
		q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + cmsg[7];
		tmp = (msg[1]) - (msg[4]) - (msg[5]) - (msg[12]) - (msg[14]);
		q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + cmsg[8];
		tmp = (msg[2]) - (msg[5]) - (msg[6]) + (msg[13]) - (msg[15]);
		q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + cmsg[9];
		tmp = (msg[0]) - (msg[3]) + (msg[6]) - (msg[7]) + (msg[14]);
		q[9] = (SHR(tmp, 1) ^ tmp) + cmsg[10];
		tmp = (msg[8]) - (msg[1]) - (msg[4]) - (msg[7]) + (msg[15]);
		q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + cmsg[11];
		tmp = (msg[8]) - (msg[0]) - (msg[2]) - (msg[5]) + (msg[9]);
		q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + cmsg[12];
		tmp = (msg[1]) + (msg[3]) - (msg[6]) - (msg[9]) + (msg[10]);
		q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + cmsg[13];
		tmp = (msg[2]) + (msg[4]) + (msg[7]) + (msg[10]) + (msg[11]);
		q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + cmsg[14];
		tmp = (msg[3]) - (msg[5]) + (msg[8]) - (msg[11]) - (msg[12]);
		q[14] = (SHR(tmp, 1) ^ tmp) + cmsg[15];
		tmp = (msg[12]) - (msg[4]) - (msg[6]) - (msg[9]) + (msg[13]);
		q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + cmsg[0];

		q[0 + 16] =
			(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROTL64(q[0], 13) ^ ROTL64(q[0], 43)) +
			(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROTL64(q[0 + 1], 19) ^ ROTL64(q[0 + 1], 53)) +
			(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROTL64(q[0 + 2], 28) ^ ROTL64(q[0 + 2], 59)) +
			(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROTL64(q[0 + 3], 4) ^ ROTL64(q[0 + 3], 37)) +
			(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROTL64(q[0 + 4], 13) ^ ROTL64(q[0 + 4], 43)) +
			(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROTL64(q[0 + 5], 19) ^ ROTL64(q[0 + 5], 53)) +
			(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROTL64(q[0 + 6], 28) ^ ROTL64(q[0 + 6], 59)) +
			(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROTL64(q[0 + 7], 4) ^ ROTL64(q[0 + 7], 37)) +
			(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROTL64(q[0 + 8], 13) ^ ROTL64(q[0 + 8], 43)) +
			(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROTL64(q[0 + 9], 19) ^ ROTL64(q[0 + 9], 53)) +
			(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROTL64(q[0 + 10], 28) ^ ROTL64(q[0 + 10], 59)) +
			(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROTL64(q[0 + 11], 4) ^ ROTL64(q[0 + 11], 37)) +
			(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROTL64(q[0 + 12], 13) ^ ROTL64(q[0 + 12], 43)) +
			(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROTL64(q[0 + 13], 19) ^ ROTL64(q[0 + 13], 53)) +
			(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROTL64(q[0 + 14], 28) ^ ROTL64(q[0 + 14], 59)) +
			(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROTL64(q[0 + 15], 4) ^ ROTL64(q[0 + 15], 37)) +
			((precalc[0] + ROTL64(h[0], 0 + 1) +
			ROTL64(h[0 + 3], 0 + 4) - ROTL64(h[0 + 10], 0 + 11)) ^ cmsg[0 + 7]);
		q[1 + 16] =
			(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROTL64(q[1], 13) ^ ROTL64(q[1], 43)) +
			(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROTL64(q[1 + 1], 19) ^ ROTL64(q[1 + 1], 53)) +
			(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROTL64(q[1 + 2], 28) ^ ROTL64(q[1 + 2], 59)) +
			(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROTL64(q[1 + 3], 4) ^ ROTL64(q[1 + 3], 37)) +
			(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROTL64(q[1 + 4], 13) ^ ROTL64(q[1 + 4], 43)) +
			(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROTL64(q[1 + 5], 19) ^ ROTL64(q[1 + 5], 53)) +
			(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROTL64(q[1 + 6], 28) ^ ROTL64(q[1 + 6], 59)) +
			(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROTL64(q[1 + 7], 4) ^ ROTL64(q[1 + 7], 37)) +
			(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROTL64(q[1 + 8], 13) ^ ROTL64(q[1 + 8], 43)) +
			(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROTL64(q[1 + 9], 19) ^ ROTL64(q[1 + 9], 53)) +
			(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROTL64(q[1 + 10], 28) ^ ROTL64(q[1 + 10], 59)) +
			(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROTL64(q[1 + 11], 4) ^ ROTL64(q[1 + 11], 37)) +
			(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROTL64(q[1 + 12], 13) ^ ROTL64(q[1 + 12], 43)) +
			(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROTL64(q[1 + 13], 19) ^ ROTL64(q[1 + 13], 53)) +
			(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROTL64(q[1 + 14], 28) ^ ROTL64(q[1 + 14], 59)) +
			(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROTL64(q[1 + 15], 4) ^ ROTL64(q[1 + 15], 37)) +
			((precalc[1] + ROTL64(h[1], 1 + 1) +
			ROTL64(h[1 + 3], 1 + 4) - ROTL64(h[1 + 10], 1 + 11)) ^ cmsg[1 + 7]);

		q[2 + 16] = CONST_EXP2(2) +
			((precalc[2] + ROTL64(h[2], 2 + 1) +
			ROTL64(h[2 + 3], 2 + 4) - ROTL64(h[2 + 10], 2 + 11)) ^ cmsg[2 + 7]);
		q[3 + 16] = CONST_EXP2(3) +
			((precalc[3] + ROTL64(h[3], 3 + 1) +
			ROTL64(h[3 + 3], 3 + 4) - ROTL64(h[3 + 10], 3 + 11)) ^ cmsg[3 + 7]);
		q[4 + 16] = CONST_EXP2(4) +
			((precalc[4] + ROTL64(h[4], 4 + 1) +
			ROL8(h[4 + 3]) - ROTL64(h[4 + 10], 4 + 11)) ^ cmsg[4 + 7]);
		q[5 + 16] = CONST_EXP2(5) +
			((precalc[5] + ROTL64(h[5], 5 + 1) +
			ROTL64(h[5 + 3], 5 + 4) - ROL16(h[5 + 10])) ^ cmsg[5 + 7]);


		q[6 + 16] = CONST_EXP2(6) +
			((precalc[6] + ROTL64(h[6], 6 + 1) +
			ROTL64(h[6 + 3], 6 + 4) - ROTL64(h[6 - 6], (6 - 6) + 1)) ^ cmsg[6 + 7]);
		q[7 + 16] = CONST_EXP2(7) +
			((precalc[7] + ROL8(h[7]) +
			ROTL64(h[7 + 3], 7 + 4) - ROTL64(h[7 - 6], (7 - 6) + 1)) ^ cmsg[7 + 7]);
		q[8 + 16] = CONST_EXP2(8) +
			((precalc[8] + ROTL64(h[8], 8 + 1) +
			ROTL64(h[8 + 3], 8 + 4) - ROTL64(h[8 - 6], (8 - 6) + 1)) ^ cmsg[8 + 7]);

		q[9 + 16] = CONST_EXP2(9) +
			((precalc[9] + ROTL64(h[9], 9 + 1) +
			ROTL64(h[9 + 3], 9 + 4) - ROTL64(h[9 - 6], (9 - 6) + 1)) ^ cmsg[9 - 9]);
		q[10 + 16] = CONST_EXP2(10) +
			((precalc[10] + ROTL64(h[10], 10 + 1) +
			ROTL64(h[10 + 3], 10 + 4) - ROTL64(h[10 - 6], (10 - 6) + 1)) ^ cmsg[10 - 9]);
		q[11 + 16] = CONST_EXP2(11) +
			((precalc[11] + ROTL64(h[11], 11 + 1) +
			ROTL64(h[11 + 3], 11 + 4) - ROTL64(h[11 - 6], (11 - 6) + 1)) ^ cmsg[11 - 9]);
		q[12 + 16] = CONST_EXP2(12) +
			((precalc[12] + ROTL64(h[12], 12 + 1) +
			ROL16(h[12 + 3]) - ROTL64(h[12 - 6], (12 - 6) + 1)) ^ cmsg[12 - 9]);



		q[13 + 16] = CONST_EXP2(13) +
			((precalc[13] + ROTL64(h[13], 13 + 1) +
			ROTL64(h[13 - 13], (13 - 13) + 1) - ROL8(h[13 - 6])) ^ cmsg[13 - 9]);
		q[14 + 16] = CONST_EXP2(14) +
			((precalc[14] + ROTL64(h[14], 14 + 1) +
			ROTL64(h[14 - 13], (14 - 13) + 1) - ROTL64(h[14 - 6], (14 - 6) + 1)) ^ cmsg[14 - 9]);
		q[15 + 16] = CONST_EXP2(15) +
			((precalc[15] + ROL16(h[15]) +
			ROTL64(h[15 - 13], (15 - 13) + 1) - ROTL64(h[15 - 6], (15 - 6) + 1)) ^ cmsg[15 - 9]);

		XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
		XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

		msg[4] = (SHR(XH64, 3) ^ q[20] ^ h[4]) + (XL64    ^ q[28] ^ q[4]);
		msg[8] = ROTL64(msg[4], 9) + (XH64     ^     q[24] ^ h[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);

		inpHash[0] = devectorize(msg[0 + 8]);

		if (((msg[8].x) & 0x8)) return;

		msg[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ h[0]) + (XL64    ^ q[24] ^ q[0]);
		msg[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ h[1]) + (XL64    ^ q[25] ^ q[1]);
		msg[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ h[2]) + (XL64    ^ q[26] ^ q[2]);
		msg[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ h[3]) + (XL64    ^ q[27] ^ q[3]);
		msg[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ h[5]) + (XL64    ^ q[29] ^ q[5]);
		msg[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ h[6]) + (XL64    ^ q[30] ^ q[6]);
		msg[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ h[7]) + (XL64    ^ q[31] ^ q[7]);

		msg[9] = ROTL64(msg[5], 10) + (XH64     ^     q[25] ^ h[9]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
		msg[10] = ROTL64(msg[6], 11) + (XH64     ^     q[26] ^ h[10]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
		msg[11] = ROTL64(msg[7], 12) + (XH64     ^     q[27] ^ h[11]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
		msg[12] = ROTL64(msg[0], 13) + (XH64     ^     q[28] ^ h[12]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
		msg[13] = ROTL64(msg[1], 14) + (XH64     ^     q[29] ^ h[13]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
		msg[14] = ROTL64(msg[2], 15) + (XH64     ^     q[30] ^ h[14]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
		msg[15] = ROL16(msg[3]) + (XH64     ^     q[31] ^ h[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);

		inpHash[1] = devectorize(msg[1 + 8]);
		inpHash[2] = devectorize(msg[2 + 8]);
		inpHash[3] = devectorize(msg[3 + 8]);
		inpHash[4] = devectorize(msg[4 + 8]);
		inpHash[5] = devectorize(msg[5 + 8]);
		inpHash[6] = devectorize(msg[6 + 8]);
		inpHash[7] = devectorize(msg[7 + 8]);
	}
}


__global__ __launch_bounds__(256, 2)
void quark_bmw512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        const uint32_t nounce = startNounce + thread;

        // Init
		uint2 __align__(64) h[16] = {
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};

		uint2 message[16];
#pragma unroll 16
        for(int i=0;i<16;i++)
			message[i] = vectorize(c_PaddedMessage80[i]);

		message[9].y = cuda_swab32(nounce);	//REPLACE_HIWORD(message[9], cuda_swab32(nounce));
        Compression512(message, h);

#pragma unroll 16
        for(int i=0;i<16;i++)
			message[i] = make_uint2(0xaaaaaaa0+i,0xaaaaaaaa);


		Compression512(h, message);

        // fertig
        uint64_t *outpHash = &g_hash[8 * thread];

#pragma unroll 8
        for(int i=0;i<8;i++)
            outpHash[i] = devectorize(message[i+8]);
    }
}

// Bmw512 f�r 80 Byte grosse Eingangsdaten
__host__ void quark_bmw512_cpu_setBlock_80(void *pdata)
{
	// Message mit Padding bereitstellen
	// lediglich die korrekte Nonce ist noch ab Byte 76 einzusetzen.
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	uint64_t *message = (uint64_t*)PaddedMessage;
	// Padding einf�gen (Byteorder?!?)
	message[10] = SPH_C64(0x80);
	// L�nge (in Bits, d.h. 80 Byte * 8 = 640 Bits
	message[15] = SPH_C64(640);

	// die Message zur Berechnung auf der GPU
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__ void quark_bmw512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 32;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    quark_bmw512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
}
__host__ void quark_bmw512_cpu_hash_64_quark(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 32;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	quark_bmw512_gpu_hash_64_quark << <grid, block >> >(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
}



__host__ void quark_bmw512_cpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
    const uint32_t threadsperblock = 128;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    quark_bmw512_gpu_hash_80<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash);
}

