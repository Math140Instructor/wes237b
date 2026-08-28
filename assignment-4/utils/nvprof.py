from abc import ABC
from pathlib import Path
from subprocess import Popen, PIPE
from typing import List
from prof_base import ProfBase

class NVProfBase(ProfBase, ABC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self) -> float:
        lines = self.profile()

        entered_context = False
        time = 0.0
        for line in lines:
            line = line.strip()
            if line.startswith("GPU activities:"):
                entered_context = True
                line = line.replace("GPU activities:", "").strip()

            if not entered_context:
                continue

            if line.startswith("API calls:"):
                break

            if "[CUDA " in line:
                continue
            
            line = line.replace(", ", ",")
            time_str = [l.strip() for l in line.split(" ") if l.strip()][1]
            
            curr_time = float(time_str)
            time += curr_time

        return time
    
class NVProfExecutable(NVProfBase):
    def __init__(self, args: List[str]) -> None:
        super().__init__()

        self._args = args

    def profile(self) -> List[str]:
        process = Popen(["/usr/local/cuda/bin/nvprof", "-u", "ms"] + self._args, stderr=PIPE, text=True)
        _, err = process.communicate()

        return err.splitlines()
    
class NVProfFile(NVProfBase):
    def __init__(self, path: Path) -> None:
        super().__init__()

        self._path = path

    def profile(self) -> List[str]:
        with self._path.open("rb") as f:
            return f.read().decode(errors='replace').splitlines() 

class NVProfTest(NVProfBase):
    def __init__(self) -> None:
        super().__init__()

    def profile(self) -> List[str]:
        return """
==245== Profiling application: ./m2 1000
==245== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
                        %        ms                  ms        ms        ms
 GPU activities:    91.30  1.83e+03         2  916.6507  665.2411  1.17e+03  conv_forward_kernel(float*, float const *, float const *, int, int, int, int, int, int)
                     6.87  138.0496         2  69.02481  60.35802  77.69160  [CUDA memcpy DtoH]
                     1.83  36.69073         8  4.586340  1.06e-03  16.30650  [CUDA memcpy HtoD]
                     0.00  4.22e-03         2  2.11e-03  1.86e-03  2.37e-03  do_not_remove_this_kernel(void)
                     0.00  3.90e-03         2  1.95e-03  1.92e-03  1.98e-03  prefn_marker_kernel(void)
      API calls:    82.11  1.83e+03         8  229.1704  3.55e-03  1.17e+03  cudaDeviceSynchronize
                     9.52  212.4718         8  26.55897  0.199866  210.3436  cudaMalloc
                     7.90  176.4918        10  17.64918  0.041317  77.91227  cudaMemcpy
                     0.41  9.059401         8  1.132425  0.249813  3.140479  cudaFree
                     0.04  1.003387       101  9.93e-03  2.29e-04  0.462165  cuDeviceGetAttribute
                     0.01  0.259815         6  0.043302  0.030765  0.064280  cudaLaunchKernel
                     0.01  0.141061         1  0.141061  0.141061  0.141061  cuDeviceGetName
                     0.00  0.015576         1  0.015576  0.015576  0.015576  cuDeviceGetPCIBusId
                     0.00  2.53e-03         3  8.44e-04  3.47e-04  1.70e-03  cuDeviceGetCount
                     0.00  1.96e-03         1  1.96e-03  1.96e-03  1.96e-03  cuDeviceTotalMem
                     0.00  1.24e-03         2  6.20e-04  2.40e-04  1.00e-03  cuDeviceGet
                     0.00  5.13e-04         1  5.13e-04  5.13e-04  5.13e-04  cuModuleGetLoadingMode
                     0.00  4.78e-04         1  4.78e-04  4.78e-04  4.78e-04  cuDeviceGetUuid
""".splitlines()
        

if __name__ == "__main__":
    test = NVProfTest()
    print(test())
