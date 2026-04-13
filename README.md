1. Login to control node
2. Clone this repository
	git clone https://github.com/tarunmcom/benchmark.git
3. Give execution permission to the script
	cd benchmark
	chmod +x benchmark.sh
4. Create a tmux session
	tmux new -s my_session
5. Allocate GPUs
	salloc --reservation=mi355-gpu-24_reservation --exclusive --account=zq3 --mem=0
6.Make changes to the benchmark.conf and start the benchmarks
  ./benchmark.sh -c benchmark.conf
