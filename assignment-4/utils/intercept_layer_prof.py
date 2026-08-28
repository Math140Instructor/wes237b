from abc import ABC
from pathlib import Path
from subprocess import Popen, PIPE
from typing import List
from prof_base import ProfBase

class InterceptLayerProfBase(ProfBase, ABC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self) -> float:
        lines = self.profile()

        entered_context = False
        time = 0.0
        for line in lines:
            line = line.strip()
            if line.startswith("Function Name"):
                entered_context = True
                continue

            if not entered_context:
                continue

            if line.startswith("cl"):
                continue

            if line == "CLIntercept is shutting down...":
                break
            
            line = line.replace(", ", " ")
            time_str = [l.strip() for l in line.split(" ") if l.strip()][2]
            
            curr_time = float(time_str) * 1.0E-6
            time += curr_time

        return time
    
class InterceptLayerProfExecutable(InterceptLayerProfBase):
    def __init__(self, args: List[str]) -> None:
        super().__init__()

        self._args = args

    def profile(self) -> List[str]:
        process = Popen(["/bin/cliloader", "-d"] + self._args, stderr=PIPE, text=True)
        _, err = process.communicate()

        return err.splitlines()
    
class InterceptLayerProfFile(InterceptLayerProfBase):
    def __init__(self, path: Path) -> None:
        super().__init__()

        self._path = path

    def profile(self) -> List[str]:
        with self._path.open("rb") as f:
            return f.read().decode(errors='replace').splitlines() 

class InterceptLayerProfTest(InterceptLayerProfBase):
    def __init__(self) -> None:
        super().__init__()

    def profile(self) -> List[str]:
        return """
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
CLIntercept (64-bit) is loading...
CLIntercept file location: /ocl-intercept/bin/../lib/libOpenCL.so
CLIntercept URL: https://github.com/intel/opencl-intercept-layer
CLIntercept git description: v3.0.5
CLIntercept git refspec: 
CLIntercept git hash: 0a3583e528e2d595a5937f53d40915b52e102cd2
CLIntercept optional features:
    cliloader(supported)
    cliprof(supported)
    kernel overrides(supported)
    ITT tracing(NOT supported)
    MDAPI(supported)
    Demangling(supported)
    clock(steady_clock)
CLIntercept environment variable prefix: CLI_
CLIntercept config file: clintercept.conf
Trying to load dispatch from: ./real_libOpenCL.so
Couldn't load library: ./real_libOpenCL.so
Trying to load dispatch from: /usr/lib/x86_64-linux-gnu/libOpenCL.so.1
... success!
Control ReportToStderr is set to non-default value: true
Control DevicePerformanceTiming is set to non-default value: true
Timer Started!
... loading complete.
input_file_a: Dataset/without_strides/15/input0.raw
input_file_b: Dataset/without_strides/15/kernel0.raw
!!SOLUTION IS CORRECT!!
Total Enqueues: 4


Device Performance Timing Results for cpu-znver3-AMD EPYC 7763 64-Core Processor (64CUs, 3529MHz):

Total Time (ns): 1709917993

                   Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
             clEnqueueReadBuffer,      1,      84847074,    4.96%,      84847074,      84847074,      84847074
            clEnqueueWriteBuffer,      2,      24322290,    1.42%,      12161145,          1970,      24320320
                   convolution2D,      1,    1600748629,   93.62%,    1600748629,    1600748629,    1600748629
CLIntercept is shutting down...
... shutdown complete.
""".splitlines()
        

if __name__ == "__main__":
    test = InterceptLayerProfTest()
    print(test())
