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
## Memory of a LLM training

As LLMs train, the memory on GPUs change over time, starting from the first pass, we can see activation memory adding up as we perform the forward pass, then they slowly get released in the backward pass.

Gradient values start to accumulate in the backward pass, and slowly get released as the optimizer states are being applied layer by layer

![[Screenshot 2026-04-08 at 11.53.36 AM.png]]

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

---
#### ZeRO 1

Starting from ZeRO 1, the idea is that each GPU holds the model + gradient, and each GPU is responsible for updating only a subset (a shard) of parameters with the optimizer.

Then, as each shard of parameter is updated on different GPUs, we need to get all those shards together and give the fully updated model to every GPU.

![[Screenshot 2026-04-01 at 10.37.29 PM.png]]

In principle, the idea is:

- every GPU performs forward + backward pass to get gradient
- reduce scatter gradients
- every GPU updates a shard of the total params using shard of gradient + shard of optimizer state
- all gather parameters

Compared to normal naive DDP, ZeRO 1 adds just 1 more all gather, but reduces the memory by `Optimizer States / Number of GPU` parameters.

![[Screenshot 2026-04-07 at 10.21.18 PM.png|500]]


## Quick Run of ZeRO 2, 3

Since we already discussed most of parallelism in huggingface's GPU playbook, this will just be a quick run down.

Zero 2 shards gradients as well, but since we now can't store a full gradient vector in memory of a GPU (since sharding), we have to perform a reduce per layer's back propagation to send it to he respective GPU.
- This reduce makes overhead increase compared to ZeRO 1

Then the rest is similar as Zero 1, update separately on each machine, then an all gather.

![[Screenshot 2026-04-07 at 10.24.13 PM.png|500]]

Then we can go further to shard parameters as well, but this makes communication a lot more annoying:

- load shards
- all gather layer 0
- all GPUs run forward
- free parameters
- all gather layer 1
- all GPUs run forward
- and on...

While it seems that communication has drastically increased since we are basically calling an all gather at every operation, these communications could be overlapped with computation, and therefore the overhead isn't that much.

![[Screenshot 2026-04-07 at 10.28.48 PM.png|500]]

Now using ZeRO techniques, we are able to store model sizes from 6B to 53B for a A100 cluster (8 GPUs).

![[Screenshot 2026-04-07 at 10.30.18 PM.png|500]]

But here's the problem, while it's nice to have Zero 1 & 2 reduce gradient & optimizer states memory, Zero 3 is what really matters for storing larger model sizes, but it's not a very good method due to the communication overhead (yes it's still slow), and it doesn't solve the problem of storing activations.

## Model Parallelism

To split up the model in memory, we can use pipeline parallel and tensor parallel.

The idea of pipeline parallel is simple, if a model can't fit on 1 GPU, divide it's layers onto multiple GPUs.

![[Screenshot 2026-04-07 at 10.37.14 PM.png|500]]

But this sucks! When GPU 0 computes layer 0, GPU 1, 2, 3 are all waiting around for data to come around, same thing happens during backward pass.

![[Screenshot 2026-04-07 at 10.38.09 PM.png|500]]

There are some solutions like micro-batches, which frankly looks like process parallelism in a CPU at a naive level.

![[Screenshot 2026-04-07 at 10.42.26 PM.png|500]]

There are more crazy engineering solutions to pipeline parallelism (which, frankly I didn't even look at myself), the big problem is the utilization vs. communication overhead, which requires more bandwidth.

![[Screenshot 2026-04-07 at 11.00.26 PM.png|500]]

Like what is going on here...

Deepseek created a pipeline parallel to have 'Zero bubble' pipelining, where the big idea is that certain parameters' gradients in back propagation could be computed whenever, while others must be computed in sequence.

Like take this example:

|**Task**|**Component**|**Destination**|**Priority**|
|---|---|---|---|
|**Forward (F)**|$y = xW$|Next Device|**High** (Required for next F)|
|**Backward (B)**|$\nabla_x L$|Previous Device|**High** (Required for next B)|
|**Weight (W)**|$\nabla_W L$|Local Memory|**Low** (Only needed for Optimizer)|

The parameters which can have their gradients computed whenever without harming the computation graph therefore could be slotted in whenever there's space for compute in the GPUs.

Like the backward gradient for activations x will be used for the loss gradient calculation in previous layers due to the chain rule, but the weights gradients aren't used anymore in the previous layer's loss gradient calculation.

![[Screenshot 2026-04-07 at 11.21.03 PM.png|500]]

---
## Tensor Parallel

As most operations in LLMs is just matrix multiplication, this process can be divided down into sub-matrices + concats/sums. Fundamentally, we've already discussed them in the huggingface book, again, but as a quick recap:

- Column Parallel

![[Pasted image 20260407234635.png|500]]

- Row Parallel

![[Pasted image 20260407234651.png|500]]

However, you don't want to perform tensor parallel between too many GPUs, again, we get to the problem of too much overhead in communication.

![[Screenshot 2026-04-08 at 12.08.21 AM.png|500]]

- 8 GPUs seem to be the sweet spot for tensor parallel (this is from the huggingface book!)


## Sequence Parallel

Tensor parallelism shards tensors across the **hidden dimension**, which breaks operations that require access to the _full hidden state per token_. Examples include Layer Normalization and dropout.

To address this, sequence parallelism redistributes work across the **sequence dimension** instead. Since operations like LayerNorm and dropout are **token-wise** (they operate independently on each token’s full hidden vector), we can safely shard the sequence across GPUs. Each GPU holds:
- the **full hidden dimension** (needed for correctness), and
- only a **subset of tokens** (`sequence_length / num_GPUs`)

In practice, both are combined together:
- compute-heavy ops (e.g. matmuls) benefit from tensor parallelism
- elementwise / token-wise ops (e.g. LayerNorm, dropout) avoid expensive communication

---
## Activation Memory

With all the conclusion of the above description methods of parallelism, we can see how much of an impact it makes on memory of activations.

This is the formula (roughly) for the naive, no parallelism memory per transformer layer.

![[Screenshot 2026-04-08 at 11.46.47 AM.png|500]]

And we can stack tensor + sequence + activation re-computation to get linear scaling for memory. 

![[Screenshot 2026-04-08 at 11.47.48 AM.png|500]]

---
## Putting it Together

In 2021, Nvidia researchers tried to see which combinations of parallelism is best for different model sizes, and in practice:

![[Screenshot 2026-04-08 at 12.02.56 PM.png|500]]

We have a few takeaways:
- tensor parallel caps at 8
- pipeline parallel can scale indefinitely
- data parallel dece

In real LLMs:
- Deepseek V3 used ZeRO 1 + Tensor + Sequence + Pipeline parallel.
- Llama 3 405B used Tensor + Context (doesn't matter a lot for now) + Pipeline + Data parallel