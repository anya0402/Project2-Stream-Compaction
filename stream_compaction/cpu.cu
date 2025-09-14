#include <cstdio>
#include <iostream>

#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i-1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int next = 0;
            for (int i = 0; i < n; ++i) {
                int curr_val = idata[i];
                if (curr_val != 0) {
                    odata[next] = curr_val;
                    next += 1;
                }
            }
            timer().endCpuTimer();
            return next;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* bool_array = new int[n];
            int* scan_result =  new int[n];
            for (int i = 0; i < n; ++i) {
                int curr_val = idata[i];
                if (curr_val) {
                    bool_array[i] = 1;
                }
                else {
                    bool_array[i] = 0;
                }
            }
            scan_result[0] = 0;
            for (int i = 1; i < n; ++i) {
                scan_result[i] = scan_result[i - 1] + bool_array[i - 1];
            }
            for (int i = 0; i < n; ++i) {
                if (bool_array[i]) {
                    odata[scan_result[i]] = idata[i];
                }
            }
            int final_num = scan_result[n - 1] + bool_array[n - 1];
            delete[] bool_array;
            delete[] scan_result;
            timer().endCpuTimer();
            return final_num;
        }
    }
}
