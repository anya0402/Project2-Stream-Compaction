#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int* odata, int* idata, int d) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
			int logval = 1 << (d - 1);
            if (k >= logval) {
                odata[k] = idata[k - logval] + idata[k];
            }
            else {
				odata[k] = idata[k];
            }
        }

        __global__ void kernInclusiveToExclusive(int n, int* odata, int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            if (k == 0) {
                odata[0] = 0;
            }
            else {
                odata[k] = idata[k - 1];
            }
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            //checkCUDAErrorWithLine("cudaMemcpy to device failed!");

            timer().startGpuTimer();

            const int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            for (int d = 1; d <= ilog2ceil(n); ++d) {
				kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, d);
                cudaDeviceSynchronize();

                int* temp = dev_idata;
				dev_idata = dev_odata;
				dev_odata = temp;
            }
			kernInclusiveToExclusive<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            //checkCUDAErrorWithLine("cudaMemcpy to host failed!");
            cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
