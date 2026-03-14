The big idea of this lecture is a recap of the transformer architecture, and many variations to the architecture/training process that modern LLM research has attempted.

The original transformer block looks like:
- absolute positional embeddings
- multi-head attention
- add & norm
- Relu feed forward

However, the alternated transformer block which the class wants you to implement uses:
- RoPE embeddings
- RMS Norm before addition
- SwiGLU feed forward

This conclusion was reached after examining research papers from labs developing modern LLMs, 19 papers to be exact, we'll go over each of these ideas.

---
## Pre Norm

![[Screenshot 2026-03-11 at 4.57.50 PM.png|500]]

Simple idea, just have the layer norm BEFORE the addition happens, it just 'seems' to work better than post norm through experiments, being more stable.
- uh... I don't know how convincing the 'why' explanations are for this choice

---
## RMS Norm 

And beside where to place the layer norm, they've moved on to another norm called RMS Norm.

![[Screenshot 2026-03-11 at 5.10.18 PM.png|400]]

We know layer norm is alike normalizing a normal distribution, making the mean and variance of x 0 and 1 respectively.

RMS Norm just simply rescales the vector by dividing it by the root-mean-square of its elements, but does not shift it with the addition of Beta.

They perform roughly about the same, but the argument for RMS Norm is simply that:
 1. there's 1 less learnable parameter (Beta)
 2. less computation needed (no need to find the mean)

---
## No Bias

![[Screenshot 2026-03-11 at 5.34.38 PM.png|500]]

Also, the bias terms is just 'dropped' from all MLP layers in the transformers for a similar reason to switching to the RMS Norm, for less parameters and less computation.

- the max is jus ReLU
- the new little function that replaces ReLU is what we'll talk about next.

---
## Activations

There's too many of these... and the most popular one (as of the video) is SwiGLU.

To understand the SwiGLU, we start with the ReGLU, which simply is a ReLU but with another learnable matrix V that multiplies with x, then does element wise multiplication with the ReLU output.

![[Screenshot 2026-03-11 at 5.46.14 PM.png|500]]

SwiGLU is a similar idea, just a gated version of the swish activation function.

![[Screenshot 2026-03-11 at 5.54.43 PM.png|400]]

And it's experimentally proven that they work better than non-gated versions, but... you do introduce this new V matrix, so I dunno how much that takes a hit to model size.

![[Screenshot 2026-03-11 at 5.58.30 PM.png|400]]

---
## Parallel Propagation

Originally, the attention block's outputs became the input for the MLP block, this is serial.

$y = x + \mathrm{MLP}(\mathrm{LayerNorm}(x + \mathrm{Attention}(\mathrm{LayerNorm}(x))))$

A few models thought about having the output of Attention and MLP side by side instead.

$y = x + \mathrm{Attention}(\mathrm{LayerNorm}(x)) + \mathrm{MLP}(\mathrm{LayerNorm}(x))$

Apparently 15% faster! But not a lot of papers followed.

---
## Positional Embeddings

There were a LOT of ideas of how to positional embeddings:
- sin and cosine embeddings from the original paper
- a trainable position embedding to be added to the word embedding
- relative embeddings for attention
- and Rope

![[Screenshot 2026-03-11 at 9.01.01 PM.png]]

The details of Rope will be discussed in another note, but essentially the idea is ROPE embeddings allow relative embeddings through 'rotation'.

![[1_3Oi7CUWqRQDysxHtYsa-Ow.gif|500]]

(I know it's really hand wavy right now, so just go read the note)

---
## Hyper Parameters

Hyperparameters do not have a single “correct” value. 

Instead, practitioners rely on commonly used conventions—values that have worked well across many experiments and models—so certain choices appear frequently in practice even though they are not theoretically fixed.

For example, scaling the hidden layers in the transformer block by a scale of 4, or by a scale of 8/5 for GLU models.

Or, the number of heads in these transformer blocks, which usually $d_{\text{model}} = h \times d_{\text{head}}$.

In other words, the ratio of the dimension of the heads * the number of the heads usually equal the dimension of the model, again, out of convention

---
## Vocab Size Choices

With multi-lingual vocab being a thing, vocab size have gone up more, a lot more.

![[Screenshot 2026-03-11 at 9.33.15 PM.png|400]]

---
## Dropout & Weight Decay

Some argue dropout isn't really needed due to the sheer quantity of data during pre-training that makes dropout obsolete.

Weight decay doesn't actually seem to change train/test loss ratio, it just seems to make loss functions more stable.

![[Screenshot 2026-03-11 at 10.09.17 PM.png|500]]

1h 4min