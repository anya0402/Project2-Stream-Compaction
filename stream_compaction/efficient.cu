#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* idata, int d) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
			int thread_stride = 1 << (d + 1);
            int new_k = k * thread_stride;
            if (new_k >= n) {
                return;
			}

            int index_right = new_k + thread_stride - 1;
            int index_left = new_k + (1 << d) - 1;

			idata[index_right] += idata[index_left];
        }

        __global__ void kernDownSweep(int n, int* idata, int d) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            int thread_stride = 1 << (d + 1);
            int new_k = k * thread_stride;
            if (new_k >= n) {
                return;
            }

            int index_right = new_k + thread_stride - 1;
            int index_left = new_k + (1 << d) - 1;

			int temp = idata[index_left];
            idata[index_left] = idata[index_right];
            idata[index_right] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int new_n = 1 << ilog2ceil(n);
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, new_n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(dev_idata + n, 0, (new_n - n) * sizeof(int));

            timer().startGpuTimer();

            const int blockSize = 128;
			
			printf("starting upsweep\n");

            for (int d = 0; d <= ilog2ceil(new_n)-1; ++d) {
				int num_threads = new_n / (1 << (d + 1));
                dim3 fullBlocksPerGrid((num_threads + blockSize - 1) / blockSize);
				kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(new_n, dev_idata, d);
				cudaDeviceSynchronize();
            }
			printf("upsweep done\n");

			cudaMemset(dev_idata + new_n - 1, 0, sizeof(int));

            for (int d = ilog2ceil(new_n) - 1; d >= 0; --d) {
                int num_threads = new_n / (1 << (d + 1));
                dim3 fullBlocksPerGrid((num_threads + blockSize - 1) / blockSize);
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(new_n, dev_idata, d);
                cudaDeviceSynchronize();
            }
			printf("downsweep done\n");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO
            int* dev_idata;
			int* dev_odata;
            int* dev_bools;
			int* dev_indices;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            const int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
			cudaDeviceSynchronize();
			scan(n, dev_indices, dev_bools);
			StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
			cudaDeviceSynchronize();

			int bool_val;
			int index_val;

            timer().endGpuTimer();

			cudaMemcpy(&bool_val, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&index_val, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);

            return bool_val + index_val;
        }
    }
}
