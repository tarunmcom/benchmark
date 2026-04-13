## ⚙️ Setup Instructions

### 1. Login to Control Node
```bash
ssh <your-control-node>
```

### 2. Clone the Repository
```bash
git clone https://github.com/tarunmcom/benchmark.git
```

### 3. Navigate to Directory & Set Permissions
```bash
cd benchmark
chmod +x benchmark.sh
```

### 4. Create a tmux Session
```bash
tmux new -s my_session
```

### 5. Allocate GPUs
```bash
salloc --reservation=mi355-gpu-24_reservation --exclusive --account=zq3 --mem=0
```

---

## 🐳 Container Setup

### 6. Configure Container Environment
```bash
export SINGULARITY_CACHEDIR=/var/tmp/$USER/singularity-cache
export SINGULARITY_TMPDIR=/var/tmp/$USER/singularity-tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
```

### 7. Start Container Instance
```bash
singularity instance start docker://rocm/vllm-dev:nightly_main_20260318 my_instance
```

### 8. Attach to the Instance
```bash
singularity shell instance://my_instance
```

### 9. Set Hugging Face Token
```bash
export HF_TOKEN='<YOUR TOKEN>'
```

---

## 🛠️ Run Benchmark

### 10. Configure Benchmark
Edit the configuration file:
```bash
nano benchmark.conf
```

### 11. Start Benchmark
```bash
./benchmark.sh -c benchmark.conf
```

---
