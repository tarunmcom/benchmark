import torch

def bytes_to_gb(bytes_val):
    return bytes_val / (1024 ** 3)

def print_gpu_info():
    if not torch.cuda.is_available():
        print("No GPU available (CUDA/ROCm not detected).")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs detected: {num_gpus}\n")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)

        name = props.name
        total_mem = props.total_memory  # in bytes

        print(f"GPU {i}:")
        print(f"  Name        : {name}")
        print(f"  Total VRAM  : {bytes_to_gb(total_mem):.2f} GB")

        # Optional: current memory usage (requires active context)
        try:
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)

            print(f"  Allocated   : {bytes_to_gb(allocated):.2f} GB")
            print(f"  Reserved    : {bytes_to_gb(reserved):.2f} GB")
        except Exception:
            pass

        print("-" * 40)

if __name__ == "__main__":
    print_gpu_info()
