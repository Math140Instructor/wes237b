from subprocess import Popen, PIPE
import os

class GPU:
    @property
    def model(self) -> str:
        if os.path.exists("/usr/bin/nvidia-smi"):
            process = Popen(["/usr/bin/nvidia-smi", "--query"], stdout=PIPE, text=True)
            out, _ = process.communicate()

            for line in out.splitlines():
                line = line.strip()
                
                if not line.startswith("Product Name"):
                    continue

                return " ".join([l for l in line.split(" ") if l.strip()][3:])
        elif os.path.exists("/proc/driver/nvidia/gpus/"):
            with open(f"{os.listdir('/proc/driver/nvidia/gpus/')[0]}/information", "r") as f:
                for line in f.readlines():
                    line = line.strip()

                    if not line.startswith("Model"):
                        continue

                    return " ".join([l for l in line.split(" ") if l.strip()][1:])
        
        return None


if __name__ == "__main__":
    gpu = GPU()
    print(gpu.model)
