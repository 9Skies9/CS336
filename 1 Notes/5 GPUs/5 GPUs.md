Before this point, we often treat GPUs as a black box. We move tensors to the GPU and assume everything will run efficiently, but to understand optimizations like **FlashAttention**, we need a clearer picture of how GPUs actually work.

---
## More Compute

Historically, hardware improvements followed **Moore’s Law**, which observed that the number of transistors on a chip roughly doubled every couple of years. Alongside this was **Dennard scaling**, which suggested that as transistors shrink, voltage and current can be reduced proportionally, allowing clock speeds to increase without increasing power density.

However, around the late 2000s this scaling slowed. Since then, performance gains no longer come mainly from faster single processors.
  
![[Pasted image 20260312163211.png|500]]

Instead, modern compute increases through **parallelism**. Rather than making one processor dramatically faster, we run many processing units simultaneously. This allows far more data to be processed at the same time.

![[Screenshot 2026-03-12 at 4.33.38 PM.png|500]]

---
## The GPU

This poster's actually pretty nice.

![[Screenshot 2026-03-12 at 4.45.42 PM.png|600]]

As you can see, they have different design goals:
- The CPU is optimized for _serial execution_ — performs a smaller number of operations very quickly, often one after another.
- The GPU is optimized for _parallel execution_ — performs many operations simultaneously across many cores.

![[Screenshot 2026-03-12 at 4.46.14 PM.png|600]]

As we can see in the execution diagram on how they differ:
- The CPU is a _Low-latency processor_ — optimized to quickly execute diverse tasks and complex logic.
- The GPU is a _High-throughput processor_ — optimized to run the same operation on many pieces of data at once.

---
## The Anatomy of a GPU

### Chips

For a Nvidia GPU, it's composed of several Streaming Multiprocessors (SMs), and you can think an SM as somewhat analogous to a **CPU core**, except it is designed to run **thousands of lightweight threads** instead of a few heavy ones.

![[Pasted image 20260312170640.png|600]]

Inside each SM are many Streaming Processors (SPs), which you can think as an ALU-like execution unit inside a GPU.

- **SPs** mainly handle FP32 arithmetic
- **Tensor Cores** handle FP16 / BF16 / FP8 / INT8 matrix multiplications
- **INT pipelines** handle integer arithmetic

### Memory

Like CPUs, GPUs use a memory hierarchy: **closer memory is faster but smaller**.

**Registers**
- private to each thread
- fastest storage

**Shared Memory / L1 Cache**
- located inside the SM
- shared memory is programmer-controlled
- L1 cache automatically stores recently used data

**L2 Cache**
- shared by all SMs
- reduces global memory traffic
  
**Global Memory (VRAM)**
- largest memory pool
- highest latency

![[Pasted image 20260312171734.png|500]]

- This is the memory mapping of a NVIDIA Tesla V100.

### Execution

CUDA programs organize work hierarchically.

```
Kernel
 └─ Grid
     └─ Blocks
         └─ Threads
```

- **Thread** – smallest execution unit
- **Block** – group of threads scheduled on an SM
- **Grid** – collection of blocks for a kernel launch

Threads are grouped into **warps** of 32 threads. The SM executes instructions at the **warp level**, meaning all threads in a warp run the same instruction simultaneously on different data. This is the **SIMT (Single Instruction, Multiple Threads)** model.

Execution pipeline:

```
kernel launch
 → blocks assigned to SMs
 → threads grouped into warps
 → warps executed by SM

```

![[Screenshot 2026-03-12 at 6.03.23 PM.png|600]]

Some additional little notes:

1. Matrix multiplication is one of the **fastest operations on modern GPUs**, especially since the introduction of **Tensor Cores**, which are specialized units designed to accelerate matrix multiply operations.
	- So, whatever machine learning stuff you are doing, you should try for most of it to be matmul!

![[Screenshot 2026-03-12 at 6.18.39 PM.png|400]]
  
2. Compute performance has increased much faster than memory bandwidth. GPUs can perform far more floating-point operations per second because more parallel compute units can be added, but memory speeds have improved much more slowly. As a result, many workloads are **memory-bound**, where compute units spend time waiting for data to arrive from memory rather than performing computations.

![[Screenshot 2026-03-12 at 6.19.41 PM.png|500]]

---
## Making ML Faster on a GPU

As we saw, memory bound sucks, we want the GPU to be on 100% throughput working all of it's cores, and not be having to be waiting for memory to be moved around.

In practice, that means trying to move less data, reuse data as much as possible once it is on-chip, and structure computation so arithmetic work dominates memory access, and we have a few tricks to do that.

## 1 Don't put conditionals in the GPU computations

As we know, all threads in a warp should execute the same instruction simultaneously on different data (SIMT execution). 

When a conditional branch occurs at the thread level, the GPU will executes one branch while masking off the other threads that should not participate, essentially setting those threads to idle.

![[Screenshot 2026-03-13 at 11.42.31 AM.png|500]]

## Lower precision in calculations!

We already remember seeing it in [[2 Pytorch, Resource Accounting]], if we had values in float 32, it would be 8 bytes, if it's float 16, it would be 4 bytes. 

If we had all our numbers in float 16, we could literally double the number of simultaneous calculations compared to when in float 32!

![[Screenshot 2026-03-13 at 11.54.18 AM.png|400]]

However, using low precision everywhere can cause numerical instability, and that's why we used mixed precision, on the left is an image of an example operation where mixed precision is used, while on the right is a list of operations that are happy/unhappy with float 16.

![[Screenshot 2026-03-13 at 12.01.25 PM.png|600]]


## Operator Fusion

Normally, ML frameworks execute operations one at a time. For example:

```
y = relu(x + b)
```

Naively this becomes:
1. compute t = x + b
2. write t to global GPU memory
3. read t back
4. compute y = relu(t)
5. write y to memory

This back and fourth between memory and computation wastes a lot of time, what if we just don't read/write those intermediate values and just return the result?

With operator fusion, the GPU does:
1. Compute t = x + b
2. Compute y = relu(t)
3. Write **y** to memory

That's basically Operator Fusion, just return the result to memory and keep all intermediate values to the GPU:

![[Screenshot 2026-03-13 at 12.15.55 PM.png|400]]


## 3. Re-computation ^1e02f6

Originally we back propagation, we compute and store values during the forward pass, then compute gradients with respect to the loss from the output back to the inputs during the backward pass.

But we can see how this sucks for our memory! We'd have to write the values of all the intermediate layers onto memory during the forward pass, then read all of values of the intermediate layers onto memory during the backward pass.

![[Screenshot 2026-03-13 at 12.30.24 PM.png|300]]

So why bother store them? What we can do instead is simply only save the output from the forward pass, then recompute all the intermediate layers in the backward pass and calculate gradients. 

![[Screenshot 2026-03-13 at 12.49.04 PM.png|300]]


Say we had a network that looks like this:

```
input → a → b → c → output
```

Forward pass, we compute:

```
x → a → b → c → y
```
  
- and just store y

Backward pass, we do the following steps:

1. compute ∂loss/∂y by reading y's value from memory.
2. compute ∂loss/∂c, but we don't know c, so we recompute:

```
x → a → b → c
```

3.  compute ∂loss/∂b, but we don't know b, so we recompute:

```
x → a → b
```
  
4. compute ∂loss/∂a, but we don't know a, so we recompute:

```
x → a
```
  
The same parts of the forward graph may be recomputed multiple times, you could say, 'wow, this is so wasteful of computation!' I can't disagree, but memory is scarce while compute is relatively more available (remember the chart above).

## Wait

2 techniques which I have 0 clue about, memory coalescing and tiling.

Bro how am I gonna understand the flash attention stuff...