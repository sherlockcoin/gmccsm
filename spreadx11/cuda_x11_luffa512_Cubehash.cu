#include "cuda_helper.h"
#include "cuda_vector.h"

//---LUFFA------------------------------

#define MULT0(a) {\
	tmp = a[7]; \
	a[7] = a[6]; \
	a[6] = a[5]; \
	a[5] = a[4]; \
	a[4] = a[3] ^ tmp; \
	a[3] = a[2] ^ tmp; \
	a[2] = a[1]; \
	a[1] = a[0] ^ tmp; \
	a[0] = tmp; \
}

#define MULT2(a,j) {\
	tmp = a[7+(8*j)];\
	a[7+(8*j)] = a[6+(8*j)];\
	a[6+(8*j)] = a[5+(8*j)];\
	a[5+(8*j)] = a[4+(8*j)];\
	a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
	a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
	a[2+(8*j)] = a[1+(8*j)];\
	a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
	a[0+(8*j)] = tmp;\
}

#define TWEAK(a0,a1,a2,a3,j)\
	a0 = ROTL32(a0,j);\
	a1 = ROTL32(a1,j);\
	a2 = ROTL32(a2,j);\
	a3 = ROTL32(a3,j);

#define STEP(c0,c1) {\
\
	uint32_t temp[ 2];\
	temp[ 0]  = chainv[0];\
	temp[ 1]  = chainv[ 5];\
	chainv[ 2] ^= chainv[ 3];\
	chainv[ 7] ^= chainv[ 4];\
	chainv[ 0] |= chainv[ 1];\
	chainv[ 5] |= chainv[ 6];\
	chainv[ 1]  = ~chainv[ 1];\
	chainv[ 6]  = ~chainv[ 6];\
	chainv[ 0] ^= chainv[ 3];\
	chainv[ 5] ^= chainv[ 4];\
	chainv[ 3] &= temp[ 0];\
	chainv[ 4] &= temp[ 1];\
	chainv[ 1] ^= chainv[ 3];\
	chainv[ 6] ^= chainv[ 4];\
	chainv[ 3] ^= chainv[ 2];\
	chainv[ 4] ^= chainv[ 7];\
	chainv[ 2] &= chainv[ 0];\
	chainv[ 7] &= chainv[ 5];\
	chainv[ 0]  = ~chainv[ 0];\
	chainv[ 5]  = ~chainv[ 5];\
	chainv[ 2] ^= chainv[ 1];\
	chainv[ 7] ^= chainv[ 6];\
	chainv[ 1] |= chainv[ 3];\
	chainv[ 6] |= chainv[ 4];\
	temp[ 0] ^= chainv[ 1];\
	temp[ 1] ^= chainv[ 6];\
	chainv[ 3] ^= chainv[ 2];\
	chainv[ 4] ^= chainv[ 7] ^ temp[ 0];\
	chainv[ 2] &= chainv[ 1];\
	chainv[ 7]  = (chainv[ 7] & chainv[ 6]) ^ chainv[ 3];\
	chainv[ 1] ^= chainv[ 0];\
	chainv[ 6] ^= chainv[ 5] ^ chainv[ 2];\
	chainv[ 5]  = chainv[ 1] ^ temp[ 1];\
	chainv[ 0]  = chainv[ 4] ^ ROTL32(temp[ 0],2); \
	chainv[ 1]  = chainv[ 5] ^ ROTL32(chainv[ 1],2); \
	chainv[ 2]  = chainv[ 6] ^ ROTL32(chainv[ 2],2); \
	chainv[ 3]  = chainv[ 7] ^ ROTL32(chainv[ 3],2); \
	chainv[ 4]  = chainv[ 0] ^ ROTL32(chainv[ 4],14); \
	chainv[ 5]  = chainv[ 1] ^ ROTL32(chainv[ 5],14); \
	chainv[ 6]  = chainv[ 2] ^ ROTL32(chainv[ 6],14); \
	chainv[ 7]  = chainv[ 3] ^ ROTL32(chainv[ 7],14); \
	chainv[ 0]  = chainv[ 4] ^ ROTL32(chainv[ 0],10) ^ c0; \
	chainv[ 1]  = chainv[ 5] ^ ROTL32(chainv[ 1],10); \
	chainv[ 2]  = chainv[ 6] ^ ROTL32(chainv[ 2],10); \
	chainv[ 3]  = chainv[ 7] ^ ROTL32(chainv[ 3],10); \
	chainv[ 4]  = ROTL32(chainv[ 4],1) ^ c1; \
	chainv[ 5]  = ROTL32(chainv[ 5],1); \
	chainv[ 6]  = ROTL32(chainv[ 6],1); \
	chainv[ 7]  = ROTL32(chainv[ 7],1); \
}

__device__ __forceinline__
void STEP2(uint32_t *t, const uint2 c0, const uint2 c1){
	uint32_t temp[ 4];
	temp[ 0] = t[ 0];
	temp[ 1] = t[ 5];	
	temp[ 2] = t[0+8];
	temp[ 3] = t[8+5];
	t[ 2] ^= t[ 3];
	t[ 7] ^= t[ 4];		
	t[8+2] ^= t[8+3];
	t[8+7] ^= t[8+4];
	t[ 0] |= t[ 1];	
	t[ 5] |= t[ 6];
	t[8+0]|= t[8+1];
	t[8+5]|= t[8+6];
	t[ 1]  = ~t[ 1];
	t[ 6]  = ~t[ 6];
	t[8+1] = ~t[8+1];
	t[8+6] = ~t[8+6];
	t[ 0] ^= t[ 3];
	t[ 5] ^= t[ 4];
	t[8+0]^= t[8+3];	
	t[8+5]^= t[8+4];
	t[ 3] &= temp[ 0];
	t[ 4] &= temp[ 1];
	t[8+3]&= temp[ 2];
	t[8+4]&= temp[ 3];
	t[ 1] ^= t[ 3];	
	t[ 6] ^= t[ 4];
	t[8+1]^= t[8+3];
	t[8+6]^= t[8+4];
	t[ 3] ^= t[ 2];	
	t[ 4] ^= t[ 7];	
	t[8+3]^= t[8+2];
	t[8+4]^= t[8+7];
	t[ 2] &= t[ 0];	
	t[ 7] &= t[ 5];	
	t[8+2]&= t[8+0];
	t[8+7]&= t[8+5];
	t[ 0]  = ~t[ 0];
	t[ 5]  = ~t[ 5];
	t[8+0] = ~t[8+0];
	t[8+5] = ~t[8+5];
	t[ 2] ^= t[ 1];
	t[ 7] ^= t[ 6];
	t[8+2]^= t[8+1];
	t[8+7]^= t[8+6];
	t[ 1] |= t[ 3];	
	t[ 6] |= t[ 4];	
	t[8+1]|= t[8+3];
	t[8+6]|= t[8+4];
	
	temp[ 0] ^= t[ 1];
	temp[ 1] ^= t[ 6];
	temp[ 2] ^= t[8+1];
	temp[ 3] ^= t[8+6];
	
	t[ 3] ^= t[ 2];
	t[ 4] ^= t[ 7] ^ temp[ 0];
	t[8+3]^= t[8+2];	
	t[8+4]^= t[8+7] ^ temp[ 2];
	t[ 2] &= t[ 1];
	t[ 7]  = (t[ 7] & t[ 6]) ^ t[ 3];
	t[8+2]&= t[8+1];
	t[ 1] ^= t[ 0];
	t[8+7] = (t[8+6] & t[8+7]) ^ t[8+3];
	t[ 6] ^= t[ 5] ^ t[ 2];		
	t[8+1]^= t[8+0];	
	t[8+6]^= t[8+2]^ t[8+5];
	t[ 5]  = t[ 1] ^ temp[ 1];
	t[ 0]  = t[ 4] ^ ROTL32(temp[ 0],2);
	t[8+5] = t[8+1]^ temp[ 3];	
	t[8+0] = t[8+4]^ ROTL32(temp[ 2],2);
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],2);
	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],2);
	t[8+1] = t[8+5]^ ROTL32(t[8+1],2);
	t[8+2] = t[8+6]^ ROTL32(t[8+2],2);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],2);
	t[ 4]  = t[ 0] ^ ROTL32(t[ 4],14);
	t[8+3] = t[8+7] ^ ROTL32(t[8+3],2);
	t[8+4] = t[8+0] ^ ROTL32(t[8+4],14);
	t[ 5]  = t[ 1] ^ ROTL32(t[ 5],14);
	t[ 6]  = t[ 2] ^ ROTL32(t[ 6],14);
	t[8+5] = t[8+1] ^ ROTL32(t[8+5],14);
	t[8+6] = t[8+2] ^ ROTL32(t[8+6],14);
	t[ 7]  = t[ 3] ^ ROTL32(t[ 7],14);
	t[ 0]  = t[ 4] ^ ROTL32(t[ 0],10) ^ c0.x;
	t[8+7] = t[8+3]^ ROTL32(t[8+7],14);
	t[8+0] = t[8+4]^ ROTL32(t[8+0],10) ^ c1.x;
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],10);
	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],10);
	t[8+1] = t[8+5]^ ROTL32(t[8+1],10);
	t[8+2] = t[8+6]^ ROTL32(t[8+2],10);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],10);
	t[ 4]  = ROTL32(t[ 4],1) ^ c0.y;
	t[8+3] = t[8+7] ^ ROTL32(t[8+3],10);
	t[8+4] = ROTL32(t[8+4],1) ^ c1.y;
	t[ 5]  = ROTL32(t[ 5],1);
	t[ 6]  = ROTL32(t[ 6],1);
	t[8+5] = ROTL32(t[8+5],1);
	t[8+6] = ROTL32(t[8+6],1);
	t[ 7]  = ROTL32(t[ 7],1);
	t[8+7] = ROTL32(t[8+7],1);
}

__device__ __forceinline__
void STEP1(uint32_t *t, const uint2 c){
	uint32_t temp[ 2];
	temp[ 0] = t[ 0];			temp[ 1] = t[ 5];
	t[ 2] ^= t[ 3];				t[ 7] ^= t[ 4];
	t[ 0] |= t[ 1];				t[ 5] |= t[ 6];
	t[ 1]  = ~t[ 1];			t[ 6]  = ~t[ 6];
	t[ 0] ^= t[ 3];				t[ 5] ^= t[ 4];
	t[ 3] &= temp[ 0];			t[ 4] &= temp[ 1];
	t[ 1] ^= t[ 3];				t[ 6] ^= t[ 4];
	t[ 3] ^= t[ 2];				t[ 4] ^= t[ 7];
	t[ 2] &= t[ 0];				t[ 7] &= t[ 5];
	t[ 0]  = ~t[ 0];			t[ 5]  = ~t[ 5];
	t[ 2] ^= t[ 1];				t[ 7] ^= t[ 6];
	t[ 1] |= t[ 3];				t[ 6] |= t[ 4];
	temp[ 0] ^= t[ 1];			temp[ 1] ^= t[ 6];
	t[ 3] ^= t[ 2];				t[ 4] ^= t[ 7] ^ temp[ 0];
	t[ 2] &= t[ 1];				t[ 7]  = (t[ 7] & t[ 6]) ^ t[ 3];
	t[ 1] ^= t[ 0];				t[ 6] ^= t[ 5] ^ t[ 2];
	t[ 5]  = t[ 1] ^ temp[ 1];		t[ 0]  = t[ 4] ^ ROTL32(temp[ 0],2);
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],2);	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],2);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],2);	t[ 4]  = t[ 0] ^ ROTL32(t[ 4],14);
	t[ 5]  = t[ 1] ^ ROTL32(t[ 5],14);	t[ 6]  = t[ 2] ^ ROTL32(t[ 6],14);
	t[ 7]  = t[ 3] ^ ROTL32(t[ 7],14);	t[ 0]  = t[ 4] ^ ROTL32(t[ 0],10) ^ c.x;
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],10);	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],10);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],10);	t[ 4]  = ROTL32(t[ 4],1) ^ c.y;
	t[ 5]  = ROTL32(t[ 5],1);		t[ 6]  = ROTL32(t[ 6],1);
						t[ 7]  = ROTL32(t[ 7],1);
}

/* initial values of chaining variables */
__constant__ const uint32_t c_CNS[80] = {
		0x303994a6,0xe0337818,0xc0e65299,0x441ba90d, 0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f, 0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4, 0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
		0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4, 0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28, 0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b, 0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
		0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72, 0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7, 0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719, 0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
		0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91, 0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be, 0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5, 0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
		0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab, 0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0, 0x78602649,0x29131ab6,0x8edae952,0x0fc053c3, 0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31
	};

__device__
static void rnd512_first(uint32_t *const __restrict__ state, uint32_t *const __restrict__ buffer)
{
	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		uint32_t tmp;
		#pragma unroll 8
		for(int i=0;i<8;i++)
			state[i+(j<<3)] ^= buffer[i];
		MULT0(buffer);
	}
	TWEAK(state[12], state[13], state[14], state[15], 1);
	TWEAK(state[20], state[21], state[22], state[23], 2);
	TWEAK(state[28], state[29], state[30], state[31], 3);
	TWEAK(state[36], state[37], state[38], state[39], 4);
	
	for (int i = 0; i<8; i++) {
		STEP2(&state[ 0],*(uint2*)&c_CNS[(2 * i) +  0],*(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&state[16],*(uint2*)&c_CNS[(2 * i) + 32],*(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&state[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}

__device__
static void rnd512(uint32_t *const __restrict__ statebuffer, uint32_t *const __restrict__ statechainv){
	uint32_t t[40];
	uint32_t tmp;

	tmp = statechainv[ 7] ^ statechainv[7 + 8] ^ statechainv[7 +16] ^ statechainv[7 +24] ^ statechainv[7 +32];
	t[7] = statechainv[ 6] ^ statechainv[6 + 8] ^ statechainv[6 +16] ^ statechainv[6 +24] ^ statechainv[6 +32];
	t[6] = statechainv[ 5] ^ statechainv[5 + 8] ^ statechainv[5 +16] ^ statechainv[5 +24] ^ statechainv[5 +32];
	t[5] = statechainv[ 4] ^ statechainv[4 + 8] ^ statechainv[4 +16] ^ statechainv[4 +24] ^ statechainv[4 +32];
	t[4] = statechainv[ 3] ^ statechainv[3 + 8] ^ statechainv[3 +16] ^ statechainv[3 +24] ^ statechainv[3 +32] ^ tmp;
	t[3] = statechainv[ 2] ^ statechainv[2 + 8] ^ statechainv[2 +16] ^ statechainv[2 +24] ^ statechainv[2 +32] ^ tmp;
	t[2] = statechainv[ 1] ^ statechainv[1 + 8] ^ statechainv[1 +16] ^ statechainv[1 +24] ^ statechainv[1 +32];
	t[1] = statechainv[ 0] ^ statechainv[0 + 8] ^ statechainv[0 +16] ^ statechainv[0 +24] ^ statechainv[0 +32] ^ tmp;
	t[0] = tmp;

//	*(uint2x4*)statechainv ^= *(uint2x4*)t;
	#pragma unroll 8
	for(int i=0;i<8;i++)
		statechainv[i] ^= t[i];

	#pragma unroll 4
	for (int j=1;j<5;j++) {
		#pragma unroll 8
		for(int i=0;i<8;i++)
			statechainv[i+(j<<3)] ^= t[i];
//		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)t;
		#pragma unroll 8
		for(int i=0;i<8;i++)
			t[i+(j<<3)] = statechainv[i+(j<<3)];
//		*(uint2x4*)&t[8*j] = *(uint2x4*)&statechainv[8*j];
	}

//	*(uint2x4*)t = *(uint2x4*)statechainv;
	#pragma unroll 8
	for(int i=0;i<8;i++)
		t[i] = statechainv[i];

	MULT0(statechainv);

	#pragma unroll 4
	for (int j=1;j<5;j++)
		MULT2(statechainv, j);

	#pragma unroll 5
	for (int j=0;j<5;j++)
		#pragma unroll 8
		for(int i=0;i<8;i++)
			statechainv[i+8*j] ^= t[i+(8*((j+1)%5))];
//		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)&t[8*((j+1)%5)];

	#pragma unroll 5
	for (int j=0;j<5;j++)
		*(uint2x4*)&t[8*j] = *(uint2x4*)&statechainv[8*j];

	MULT0(statechainv);
	#pragma unroll 4
	for (int j=1;j<5;j++)
		MULT2(statechainv, j);

	#pragma unroll 5
	for (int j=0;j<5;j++)
		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)&t[8*((j+4)%5)];

	#pragma unroll 5
	for (int j=0;j<5;j++) {
		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)statebuffer;
		MULT0(statebuffer);
	}

	TWEAK(statechainv[12], statechainv[13], statechainv[14], statechainv[15], 1);
	TWEAK(statechainv[20], statechainv[21], statechainv[22], statechainv[23], 2);
	TWEAK(statechainv[28], statechainv[29], statechainv[30], statechainv[31], 3);
	TWEAK(statechainv[36], statechainv[37], statechainv[38], statechainv[39], 4);

	for (int i = 0; i<8; i++){
		STEP2( statechainv    ,*(uint2*)&c_CNS[(2 * i) +  0], *(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&statechainv[16],*(uint2*)&c_CNS[(2 * i) + 32], *(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&statechainv[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}

__device__ __forceinline__
static void rnd512_nullhash(uint32_t *const __restrict__ state){

	uint32_t t[40];
	uint32_t tmp;

	tmp = state[ 7] ^ state[7 + 8] ^ state[7 +16] ^ state[7 +24] ^ state[7 +32];
	t[7] = state[ 6] ^ state[6 + 8] ^ state[6 +16] ^ state[6 +24] ^ state[6 +32];
	t[6] = state[ 5] ^ state[5 + 8] ^ state[5 +16] ^ state[5 +24] ^ state[5 +32];
	t[5] = state[ 4] ^ state[4 + 8] ^ state[4 +16] ^ state[4 +24] ^ state[4 +32];
	t[4] = state[ 3] ^ state[3 + 8] ^ state[3 +16] ^ state[3 +24] ^ state[3 +32] ^ tmp;
	t[3] = state[ 2] ^ state[2 + 8] ^ state[2 +16] ^ state[2 +24] ^ state[2 +32] ^ tmp;
	t[2] = state[ 1] ^ state[1 + 8] ^ state[1 +16] ^ state[1 +24] ^ state[1 +32];
	t[1] = state[ 0] ^ state[0 + 8] ^ state[0 +16] ^ state[0 +24] ^ state[0 +32] ^ tmp;
	t[0] = tmp;
	
	#pragma unroll 5
	for (int j = 0; j<5; j++){
//		#pragma unroll 8
//		for(int i=0;i<8;i++)
//			state[i+(j<<3)] ^= t[i];
		*(uint2x4*)&state[8*j] ^= *(uint2x4*)t;
	}
	
	#pragma unroll 5
	for (int j = 0; j<5; j++){
///		#pragma unroll 8
///		for(int i=0;i<8;i++)
//			t[i+(j<<3)] = state[i+(j<<3)];
		*(uint2x4*)&t[8*j] = *(uint2x4*)&state[8*j];
	}
	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		MULT2(state, j);
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
//		#pragma unroll 8
//		for(int i=0;i<8;i++)
//			state[i+(j<<3)] ^= t[i + (((j + 1) % 5)<<3)];
		*(uint2x4*)&state[8*j] ^= *(uint2x4*)&t[8 * ((j + 1) % 5)];
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
//		#pragma unroll 8
//		for(int i=0;i<8;i++)
//			t[i+8*j] = state[i+8*j];
		*(uint2x4*)&t[8*j] = *(uint2x4*)&state[8*j];
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		MULT2(state, j);
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		#pragma unroll 8
		for(int i=0;i<8;i++)
			state[i+8*j] ^= t[i+(8 * ((j + 4) % 5))];
//		*(uint2x4*)&state[8*j] ^= *(uint2x4*)&t[8 * ((j + 4) % 5)];
	}

	TWEAK(state[12], state[13], state[14], state[15], 1);
	TWEAK(state[20], state[21], state[22], state[23], 2);
	TWEAK(state[28], state[29], state[30], state[31], 3);
	TWEAK(state[36], state[37], state[38], state[39], 4);
	
//	#pragma unroll 8
	for (int i = 0; i<8; i++) {
		STEP2(&state[ 0],*(uint2*)&c_CNS[(2 * i) +  0],*(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&state[16],*(uint2*)&c_CNS[(2 * i) + 32],*(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&state[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}

__global__ __launch_bounds__(384,2)
void x11_luffa512_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t statebuffer[8];
	
	if (thread < threads)
	{
		uint32_t statechainv[40] = {
			0x8bb0a761, 0xc2e4aa8b, 0x2d539bc9, 0x381408f8,	0x478f6633, 0x255a46ff, 0x581c37f7, 0x601c2e8e,
			0x266c5f9d, 0xc34715d8, 0x8900670e, 0x51a540be,	0xe4ce69fb, 0x5089f4d4, 0x3cc0a506, 0x609bcb02,
			0xa4e3cd82, 0xd24fd6ca, 0xc0f196dc, 0xcf41eafe,	0x0ff2e673, 0x303804f2, 0xa7b3cd48, 0x677addd4,
			0x66e66a8a, 0x2303208f, 0x486dafb4, 0xc0d37dc6,	0x634d15af, 0xe5af6747, 0x10af7e38, 0xee7e6428,
			0x01262e5d, 0xc92c2e64, 0x82fee966, 0xcea738d3,	0x867de2b0, 0xe0714818, 0xda6e831f, 0xa7062529
		};
		uint2x4* Hash = (uint2x4*)&g_hash[thread<<4];

		uint32_t hash[16];

		*(uint2x4*)&hash[0] = __ldg4(&Hash[0]);
		*(uint2x4*)&hash[8] = __ldg4(&Hash[1]);
		
		#pragma unroll 8
		for(int i=0;i<8;i++){
			statebuffer[i] = cuda_swab32(hash[i]);
		}
		
		rnd512_first(statechainv, statebuffer);

		#pragma unroll 8
		for(int i=0;i<8;i++){
			statebuffer[i] = cuda_swab32(hash[8+i]);
		}

		rnd512(statebuffer, statechainv);
		
		statebuffer[0] = 0x80000000;
		#pragma unroll 7
		for(uint32_t i=1;i<8;i++)
			statebuffer[i] = 0;

		rnd512(statebuffer, statechainv);

		/*---- blank round with m=0 ----*/
		rnd512_nullhash(statechainv);

		#pragma unroll 8
		for(int i=0;i<8;i++)
			hash[i] = cuda_swab32(statechainv[i] ^ statechainv[i+8] ^ statechainv[i+16] ^ statechainv[i+24] ^ statechainv[i+32]);

		rnd512_nullhash(statechainv);

		#pragma unroll 8
		for(int i=0;i<8;i++)
			hash[8+i] = cuda_swab32(statechainv[i] ^ statechainv[i+8] ^ statechainv[i+16] ^ statechainv[i+24] ^ statechainv[i+32]);

		Hash[ 0] = *(uint2x4*)&hash[ 0];
		Hash[ 1] = *(uint2x4*)&hash[ 8];
	}
}
//---END OF LUFFA-----------------------

//---CUBEHASH---------------------------
#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }

#define TPB 768

__device__ __forceinline__
static void rrounds(uint32_t *x){
	#pragma unroll 2
	for (int r = 0; r < 16; r++) {
		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0], 7);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1], 7);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2], 7);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3], 7);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4], 7);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5], 7);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6], 7);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7], 7);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8], 7);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9], 7);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 7);x[27] = x[27] + x[11];x[11] = ROTL32(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 7);x[29] = x[29] + x[13];x[13] = ROTL32(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 7);x[31] = x[31] + x[15];x[15] = ROTL32(x[15], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[ 0], x[ 8]);x[ 0] ^= x[16];x[ 8] ^= x[24];SWAP(x[ 1], x[ 9]);x[ 1] ^= x[17];x[ 9] ^= x[25];
		SWAP(x[ 2], x[10]);x[ 2] ^= x[18];x[10] ^= x[26];SWAP(x[ 3], x[11]);x[ 3] ^= x[19];x[11] ^= x[27];
		SWAP(x[ 4], x[12]);x[ 4] ^= x[20];x[12] ^= x[28];SWAP(x[ 5], x[13]);x[ 5] ^= x[21];x[13] ^= x[29];
		SWAP(x[ 6], x[14]);x[ 6] ^= x[22];x[14] ^= x[30];SWAP(x[ 7], x[15]);x[ 7] ^= x[23];x[15] ^= x[31];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[16], x[18]); SWAP(x[17], x[19]); SWAP(x[20], x[22]); SWAP(x[21], x[23]);SWAP(x[24], x[26]); SWAP(x[25], x[27]); SWAP(x[28], x[30]); SWAP(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0],11);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1],11);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2],11);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3],11);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4],11);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5],11);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6],11);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7],11);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8],11);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9],11);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10],11);x[27] = x[27] + x[11];x[11] = ROTL32(x[11],11);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12],11);x[29] = x[29] + x[13];x[13] = ROTL32(x[13],11);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14],11);x[31] = x[31] + x[15];x[15] = ROTL32(x[15],11);
		/* "swap x_0j0lm with x_0j1lm" */
		SWAP(x[ 0], x[ 4]); x[ 0] ^= x[16]; x[ 4] ^= x[20]; SWAP(x[ 1], x[ 5]); x[ 1] ^= x[17]; x[ 5] ^= x[21];
		SWAP(x[ 2], x[ 6]); x[ 2] ^= x[18]; x[ 6] ^= x[22]; SWAP(x[ 3], x[ 7]); x[ 3] ^= x[19]; x[ 7] ^= x[23];
		SWAP(x[ 8], x[12]); x[ 8] ^= x[24]; x[12] ^= x[28]; SWAP(x[ 9], x[13]); x[ 9] ^= x[25]; x[13] ^= x[29];
		SWAP(x[10], x[14]); x[10] ^= x[26]; x[14] ^= x[30]; SWAP(x[11], x[15]); x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[16], x[17]); SWAP(x[18], x[19]); SWAP(x[20], x[21]); SWAP(x[22], x[23]);SWAP(x[24], x[25]); SWAP(x[26], x[27]); SWAP(x[28], x[29]); SWAP(x[30], x[31]);
	}
}

/***************************************************/
// GPU Hash Function
__global__ __launch_bounds__(TPB)
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint64_t *g_hash){

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		uint32_t *Hash = (uint32_t*)&g_hash[8 * thread];

		uint32_t x[32] = {
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
			0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
			0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
			0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
			0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
			0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
			0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
		};
	
		// erste Hälfte des Hashes (32 bytes)
		//Update32(x, (const BitSequence*)Hash);
		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[0]);

		rrounds(x);

		// zweite Hälfte des Hashes (32 bytes)
	//        Update32(x, (const BitSequence*)(Hash+8));
		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[8]);
		
		rrounds(x);

		// Padding Block
		x[ 0] ^= 0x80;
		rrounds(x);
	
	//	Final(x, (BitSequence*)Hash);
		x[31] ^= 1;

		/* "the state is then transformed invertibly through 10r identical rounds" */
		#pragma unroll 10
		for (int i = 0;i < 10;++i)
			rrounds(x);

		/* "output the first h/8 bytes of the state" */
		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&x[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&x[ 8];
	}
}

//---END OF CUBEHASH--------------------
__host__ void x11_luffa512_cpu_hash_64(uint32_t threads, uint32_t *d_hash){

	const uint32_t tpb = 384;
	dim3 gridL((threads + tpb-1)/tpb);
	dim3 blockL(tpb);

	x11_luffa512_gpu_hash_64<<<gridL, blockL>>>(threads,d_hash);
}

