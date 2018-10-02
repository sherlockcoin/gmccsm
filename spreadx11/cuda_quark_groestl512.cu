// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_vector.h"

#define TPB 512
#define THF 4

// 64 Register Variante für Compute 3.0
#include "groestl_functions_quad.cu"
#include "bitslice_transformations_quad.cu"

__global__ __launch_bounds__(TPB, 2)
void quark_groestl512_gpu_hash_64_quad(uint32_t threads, uint32_t *const __restrict__ g_hash)
{
	uint32_t msgBitsliced[8];
	uint32_t state[8];
	uint32_t output[16];
	// durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
    if (thread < threads)
    {
        // GROESTL
        uint32_t *inpHash = &g_hash[thread * 16];

        const uint32_t thr = threadIdx.x & (THF-1);

		uint32_t message[8] =
		{
			inpHash[thr], inpHash[(THF)+thr], inpHash[(2 * THF) + thr], inpHash[(3 * THF) + thr],0, 0, 0, 
		};
		if (thr == 0) message[4] = 0x80UL;
		if (thr == 3) message[7] = 0x01000000UL;

		to_bitslice_quad(message, msgBitsliced);

        groestl512_progressMessage_quad(state, msgBitsliced,thr);

		from_bitslice_quad(state, output);

#if __CUDA_ARCH__ <= 500
		output[0] = __byte_perm(output[0], __shfl(output[0], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[2] = __byte_perm(output[2], __shfl(output[2], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[4] = __byte_perm(output[4], __shfl(output[4], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[6] = __byte_perm(output[6], __shfl(output[6], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[8] = __byte_perm(output[8], __shfl(output[8], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[10] = __byte_perm(output[10], __shfl(output[10], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[12] = __byte_perm(output[12], __shfl(output[12], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[14] = __byte_perm(output[14], __shfl(output[14], (threadIdx.x + 1) & 3, 4), 0x7632);
	
		if (thr == 0 || thr == 2){
			output[0 + 1] = __shfl(output[0], (threadIdx.x + 2) & 3, 4);
			output[2 + 1] = __shfl(output[2], (threadIdx.x + 2) & 3, 4);
			output[4 + 1] = __shfl(output[4], (threadIdx.x + 2) & 3, 4);
			output[6 + 1] = __shfl(output[6], (threadIdx.x + 2) & 3, 4);
			output[8 + 1] = __shfl(output[8], (threadIdx.x + 2) & 3, 4);
			output[10 + 1] = __shfl(output[10], (threadIdx.x + 2) & 3, 4);
			output[12 + 1] = __shfl(output[12], (threadIdx.x + 2) & 3, 4);
			output[14 + 1] = __shfl(output[14], (threadIdx.x + 2) & 3, 4);		
			if(thr==0){
				*(uint28*)&inpHash[0] = *(uint28*)&output[0];
				*(uint28*)&inpHash[8] = *(uint28*)&output[8];
			}
		}
#else
		output[0] = __byte_perm(output[0], __shfl(output[0], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[0 + 1] = __shfl(output[0], (threadIdx.x + 2) & 3, 4);

		output[2] = __byte_perm(output[2], __shfl(output[2], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[2 + 1] = __shfl(output[2], (threadIdx.x + 2) & 3, 4);
		
		output[4] = __byte_perm(output[4], __shfl(output[4], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[4 + 1] = __shfl(output[4], (threadIdx.x + 2) & 3, 4);
		
		output[6] = __byte_perm(output[6], __shfl(output[6], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[6 + 1] = __shfl(output[6], (threadIdx.x + 2) & 3, 4);
		
		output[8] = __byte_perm(output[8], __shfl(output[8], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[8 + 1] = __shfl(output[8], (threadIdx.x + 2) & 3, 4);

		output[10] = __byte_perm(output[10], __shfl(output[10], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[10 + 1] = __shfl(output[10], (threadIdx.x + 2) & 3, 4);
		
		output[12] = __byte_perm(output[12], __shfl(output[12], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[12 + 1] = __shfl(output[12], (threadIdx.x + 2) & 3, 4);
		
		output[14] = __byte_perm(output[14], __shfl(output[14], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[14 + 1] = __shfl(output[14], (threadIdx.x + 2) & 3, 4);

		if(thr==0){
			*(uint28*)&inpHash[0] = *(uint28*)&output[0];
			*(uint28*)&inpHash[8] = *(uint28*)&output[8];
		}
#endif
	}
}

__host__ void quark_groestl512_cpu_hash_64(uint32_t threads, uint32_t *d_hash)
{
    // Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
    // mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
    const int factor = THF;

    // berechne wie viele Thread Blocks wir brauchen
	dim3 grid(factor*((threads + TPB - 1) / TPB));
	dim3 block(TPB);

    quark_groestl512_gpu_hash_64_quad<<<grid, block>>>(threads, d_hash);
}

