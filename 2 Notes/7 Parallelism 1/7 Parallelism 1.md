Modern LLM models get too large to fit on 1 GPU, and we have to take advantage of distributed training of them on parallel setups.

![[Screenshot 2026-04-01 at 9.03.49 PM.png]]

But, just as a single machine has a memory hierarchy (registers → L1/L2/L3 cache → DRAM → SSD → network) where speed decreases and capacity increases as you move outward, a distributed parallel system has a communication hierarchy:

Let's look at this diagram of parallelism.

![[Screenshot 2026-04-01 at 9.24.05 PM.png|800]]

| Interconnect       | Typical Total Bandwidth (per direction) |
| ------------------ | --------------------------------------- |
| **NVLink 3.0**     | 600–900 GB/s (per GPU, aggregated)      |
| **xGMI‑2**         | Up to ~184 GB/s per link (depending)    |
| **HDR InfiniBand** | 200 Gb/s (25 GB/s) per port             |
| **PCIe 4.0**       | 32 GB/s (x16)                           |
|                    |                                         |

- NVLink – GPU ↔ GPU, intra‑node, ultra‑fast (up to 400 GT/s per lane). Bypasses CPU/PCIe.
- xGMI – CPU ↔ CPU, AMD’s high‑speed coherent interconnect.
- HDR InfiniBand – Node ↔ node, inter‑node, uses RDMA for direct GPU‑to‑GPU communication across machines.
- PCIe – CPU ↔ everything else (GPUs, SSDs, HCAs); the standard intra‑node I/O backbone.

So communications between GPUs are the quickest, GPUs and CPUs are slower, nodes and nodes (a node is considered that entire setup there) are more slower, and so on.

---

## Quick Recall of Parallelism Operations

For the communications between parallel systems, these are some of the common collective communication operations:

![[Screenshot 2026-04-01 at 9.29.02 PM.png|800]]

![[3.5 The Stupid Names In Parallelism| Parallelism In A Nutshll]]


An important note is that an all reduce could be implemented as two steps, a reduce scatter and a all gather.

![[Screenshot 2026-04-01 at 9.46.54 PM.png|500]]



As we scale these parallel systems, we wish that they scale linearly with the number of GPUs we add:

- Memory — total available memory should increase proportionally with the number of GPUs (e.g., 8 GPUs × 80 GB = 640 GB usable).
- Compute — total throughput should also increase linearly (e.g., 8 GPUs = 8× the FLOPs of one GPU).

The TLDR is, the less overhead per GPU, the more throughput per GPU, the happier we are for parallelism, but in reality... it isn't linear. The whole point of the interconnect hierarchy (NVLink, PCIe, InfiniBand) is to minimize that overhead so that scaling remains as close to linear as possible.

---
## The Modern LLM Parallelization Primitives

There are 3 main ideas in parallelizing LLM operations:

- Data parallelism: split the training data across GPUs
- Model parallelism:  split the model across GPUs
- Activation parallelism: split the activations across GPUs


The naive starting point in data parallelism is simply split the elements of B sized batch across M machines.

![[Screenshot 2026-04-01 at 10.08.50 PM.png|400]]

- Each machine will calculate forward/back propagation independently and get gradients independently. 
- Each machine will perform an all reduce operation and take the average of the gradients across all M machines.
- Each machine updates the model's parameters using the same average gradient.

Now, how does this scale per 1 GPU?

- compute scaling: (assuming B is not changed) ach GPU processes B/M, so per-GPU compute decreases with 1 more GPU, but total work is unchanged.
- memory scaling: no scaling, as each GPU stores the entire model, per‑GPU memory usage does not change with 1 more GPU

The all-reduce operation has a communication cost of approximately 2 * model_size per GPU per iteration.


### Why This Sucks

We are being very wasteful on the memory side of things, for 1 model, we need 5 copies of weights, totaling 16 bytes per parameter, already assuming we are using half precisions.

- 2 bytes for FP/BF 16 model parameters  
- 2 bytes for FP/BF 16 gradients  
- 4 bytes for FP32 master weights (the thing you accumulate into in SGD)  
- 4 (or 2) bytes for FP32/BF16 Adam first moment estimates  
- 4 (or 2) bytes for FP32/BF16 Adam second moment estimates  

A big part of our model isn't even the problem of loading the model, it's just replicating the same optimizer states across GPUs.


## ZeRO - Solving Data Parallel Overhead

We don't need to copy everything to every machine, and we could shard the optimizer states, gradients, and even the parameters, each step reducing down the memory needed down.

![[Screenshot 2026-04-01 at 10.30.14 PM.png|500]]

- $K$: the memory multiplier for the optimizer states per parameter. 
	- For Adam, K=12 bytes per parameter in mixed-precision training, because you typically keep:
	    - fp16 parameter: 2 bytes
	    - fp16 gradient: 2 bytes
	    - fp32 master weights + Adam moments m, v: 4+4+4=12 bytes

- $\Psi$ : the total number of model parameters.
- $N_d$​: the number of data-parallel devices,  the number of GPUs. 





#### ZeRO 1

Starting from ZeRO 1, the idea is that each GPU holds the model + gradient, and each GPU is responsible for updating only a subset (a shard) of parameters with the optimizer.

Then, as each shard of parameter is updated on different GPUs, we need to get all those shards together and give the fully updated model to every GPU.

![[Screenshot 2026-04-01 at 10.37.29 PM.png]]

In principle, the idea is:

- every GPU performs forward + backward pass to get gradient
- reduce scatter gradients
- every GPU updates a shard of the total params using shard of gradient + shard of optimizer state
- all gather parameters

This 

