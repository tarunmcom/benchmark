
⚙️ Setup Instructions
1. Login to Control Node
```bash
ssh <your-control-node>
```
2. Clone the Repository
```bash
git clone https://github.com/tarunmcom/benchmark.git
```
3. Navigate to Directory & Set Permissions
```bash
cd benchmark
chmod +x benchmark.sh
```
4. Create a tmux Session
```bash
tmux new -s my_session
```
5. Allocate GPUs
```bash
salloc --reservation=mi355-gpu-24_reservation --exclusive --account=zq3 --mem=0
```
---
🛠️ Run Benchmark
6. Configure Benchmark
Edit the configuration file:
```bash
nano benchmark.conf
```
7. Start Benchmark
```bash
./benchmark.sh -c benchmark.conf
```
---
