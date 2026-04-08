Before getting into the class, let's ask some questions:
- How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?
- What's the largest model that can you can train on 8 H100s using AdamW (naively)?

You right now probably have little to no clue, as before our understanding of these LLMs and algorithms stopped at a theoretical level, you made a model, got some data, trained the model, used it, and viola, no questioned asked.

But when we want to scale computation up... we kinda need to know the specs and limitation of our hardware, and we'll learn about understanding these LLMs from tensors, to optimizers, to how to efficiency use computation resources.

(And you should be able to answer those questions above... not with some wave of hand, but actual calculations)

---
## Tensors

We've already seen a lot on making tensors, `zeros`, `ones`, `randn`, `tensor`... but now we want to see how much memory a tensor takes up, depending on the number of numbers stored, and the data type for numbers in the tensor.

float 32:

![[Pasted image 20260311110638.png]]

Say we had a tensor like `torch.zeros(4, 8)`, thats:
- $4 \cdot 8 = 32$ elements
- $32 \cdot 4 = 128$ bytes
	- as one float 32 is 4 bytes

Now, take that idea to something like a MLP layer of GPT-3, of size `torch.zeros(12288, 4 * 12288):
- $12288 \cdot 12288 \cdot 4 = 603,979,776$ elements
- $603,979,776 \cdot 4 = 2415919104$ bytes -> 2.4 GB

As we can see, that's a lot of memory, and for deep learning, precision actually doesn't matter that much, and so we can decrease the precision in exchange for smaller model sizes.


float 16:

![[Pasted image 20260311111511.png|500]]

Now, we went from 32 bits for a float to 16 bits for a float, and if we repeat our computation for a MLP layer of GPT-3... 2.4 GB would turn into 1.2 GB.

But, this reduces:
1. the range available for expression (as exponent is smaller)
2. the precision available for expression (as fraction is smaller)


bfloat16:

![[Pasted image 20260311111757.png|500]]

To increase the range available for expression, bfloat 16 was created for a larger range of expression.


float 8, float 4:

![[Pasted image 20260311112444.png|500]]

![[Pasted image 20260311112502.png|300]]

Well... you can even go further and further reducing the number of bits for a single number, but this is usually not advised for training:

- Training with fp8, float16 and even bfloat16 is risky, and you can get instability
- Training with float32 is memory costly, and therefore comes mixed precision training (will discuss later)

The process of turning a model's parameters from float32 to lower precision for inference (to save memory) is called quantization, smaller model you get, lower performance you get.

---
## Tensor On Memory

If we recall from C, how arrays really just store pointers to them, and how higher dimensional arrays are really just 1D arrays in memory, it's no different in PyTorch.

![[Pasted image 20260311113903.png|500]]

You just have strides that can tell the pointer where to go in memory. As such, we can create different `views` of the same tensor and not use up more memory than needed.

In PyTorch a tensor conceptually has two parts:
1. **Storage** – the actual contiguous block of memory containing the values.
2. **Tensor metadata** – shape, stride, dtype, device, etc.

For example, if we make a tensor in memory:

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])

print(x.stride())
>> (3, 1)

print(x[0])
>> tensor([1., 2., 3.])

print(x[:, 0])
>> tensor([1., 4.])
```

The stride means:
- moving along dimension 0 (rows) jumps **3 elements**
- moving along dimension 1 (columns) jumps **1 element**

And indexing is just as such: `x[i, j] = storage[i*3 + j*1]`

If we try creating a different variable by say, either changing the view or transposing x, it doesn't create a new tensor, just a new 'view'/'stride' of the existing values in memory.

```python
y = x.view(3, 2)

print(y)
>> tensor([[1., 2.],
		   [3., 4.],
	       [5., 6.]])
	       
	       
print(y.stride())
>> (2, 1)
```

This means y's indexing rule is:

```python
y[i, j] = storage[i*2 + j*1]
```

As both x and y are pointers to the same underlying storage, changing any value of the memory through x means the values y sees also changed.

There's a really good video explained about the details of how this mathematically works in memory: https://www.bilibili.com/video/BV1SB2gBFEyu

In PyTorch, the @ operator performs matrix multiplication on the last two dimensions, while all preceding dimensions are treated as batch dimensions.

```python
x = torch.ones(4, 8, 16, 32)
w = torch.ones(32, 2)
y = x @ w

print(y.shape)
>> (4, 8, 16, 2)
```

Usually we keep track of what dimensions mean what with comments, there's also libraries to help with what each dimension means if you wish.

---
## Speed of Computation

Now you know the size of computations... what's their speed? 

A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x * y), and there are 2 commonly used words here:

- FLOPs: floating-point operations (measure of computation done)
- FLOP/s: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware.

Nvidia has it all laid out on their website on how fast their hardware is, for example:
- A100 has a peak performance of 312 teraFLOP/s
- H100 has a peak performance of 990 teraFLOP/s

And now you could answer say, how many flops of operations can 8 H100s process in 2 weeks?

![[Screenshot 2026-03-11 at 12.26.34 PM.png|500]]

![[Screenshot 2026-03-11 at 12.26.49 PM.png|400]]

So, about $1.9 \cdot 10^{22}$ Flops.

We could even go deeper into say, the computation of different operations.

1. Matrix Multiplication

Starting with matrix multiplication between x of size (B, D) and w of size (D, K), the output has size (B, K). 
- You will have in total $2 * B * D * K$ operations.

2. Element wise operations

For an elementwise operation on an matrix of (m, n)the complexity is O(mn), because the matrix contains m * n elements, and the operation is applied independently to each element.

As LLMs get larger, the proportion of FLOPs spent on basic matrix multiplication becomes larger and larger.

![[Pasted image 20260311132335.png|500]]

To test how 'good' a model utilizes a GPU's theoretical flops per second computation:
 1. Researchers first compute the **total number of FLOPs required by the model** (from the architecture and tensor sizes).
 2. They then **measure how long the model actually takes to run** on the GPU. From this runtime they can compute the **achieved FLOPs per second** of the system. 
3. The ratio between the achieved throughput and the peak throughput is called **Model FLOPs Utilization (MFU)**.

Usually, MFU of >= 0.5 is quite good.

---
## Gradient Computation Costs

The above is all the 'forward pass' of the model, the crunching of numbers during inference/use of the model, and we haven't talked the additional number crunching when we have to update those numbers through gradients.

(?)


---

## Implementing A Optimizer

