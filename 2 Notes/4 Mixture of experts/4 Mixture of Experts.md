MoE (mixture of experts) is a terrible naming convention for the technique, the whole idea of it simply instead of activating a single, big MLP layer of a transformer block for forward propagation, we replace that with a 'selector' layer and many big MLP layers which we call 'experts'.

So instead of:

$x \rightarrow \text{MLP} \rightarrow y$

We have:

$x \rightarrow \text{router} \rightarrow \text{selected experts} \rightarrow y$

![[Screenshot 2026-03-12 at 10.54.00 AM.png|600]]

- The router itself is simply: a linear layer + softmax to pick the top_k experts (this number is usually 1 and no more than 2)
- Usually each expert MLP is roughly the same size as the normal transformer MLP, not smaller.

So, the idea is that overtime in training, these 'experts' are suppose to understand different domains of knowledge in language. And yeah, it makes model size explode, but compute doesn't take more time as not all experts are activated in forward propagation.

They are getting popular simply because they train faster + better loss 

![[Screenshot 2026-03-12 at 10.55.30 AM.png|500]]

And they began in the Chinese Labs! (Qwen, Deepseek) like here's a chart from Deepseek saying how MoE models outperform non-MoE ones (look to the right, the Switch, 1 MoE)

![[Screenshot 2026-03-12 at 11.07.00 AM.png|400]]

And for example, DeepSeek-V3, an MoE model outperformed a lot of the other LLMs at the time.

![[Screenshot 2026-03-12 at 11.24.38 AM.png|500]]

But MoE is also kinda hard to train, why? Like, the router may start sending most tokens to only a few experts, and then some experts overtrain and others never train.

Also, during distributed training, tokens routed to different experts often live on different GPUs. So each step requires a large **all-to-all communication**:

```
tokens → send to expert GPU → process → send back
```

This is often the hardest engineering problem in MoE.

The Top-K routing is simply selecting the top-k experts, forward propagate them, and then adding the result.
- Hashing works as well as a baseline

![[Screenshot 2026-03-12 at 11.44.48 AM.png|500]]

There's more advancements in trying deep variations of MoE, like increasing the number of experts, the number of top-k samples, having a default, shared expert... 

![[Screenshot 2026-03-12 at 11.58.58 AM.png|500]]

And DeepSeek's report generally said shared experts + more experts = better performance.

![[Screenshot 2026-03-12 at 11.59.53 AM.png|500]]

But this raises the questions on how to train MoEs.

1. The purpose of selecting the **top-k experts** is to keep the number of FLOPs the same as a standard transformer layer. 
2. However, the top-k routing decision is discrete, which is not differentiable.

So... what to do? (Idk what the class was talking about)

And also, another problem arises in the potential that during training, we might easily get stuck in a local minima where the router keeps routing the tokens to 1-2 experts while the rest are untrained.

To solve this problem, an additional loss term is introduced with the purpose of balancing the distribution of the probability of experts processing the tokens.

![[Screenshot 2026-03-12 at 1.27.11 PM.png|600]]

Let’s do a small concrete example with 3 experts and a batch of 10 tokens. For each token, the router outputs a softmax probability distribution over the 3 experts.

Suppose the 10 tokens have these router probabilities:

$\begin{aligned} x_1 &: [0.70,\ 0.20,\ 0.10] \\ x_2 &: [0.60,\ 0.30,\ 0.10] \\ x_3 &: [0.55,\ 0.25,\ 0.20] \\ x_4 &: [0.80,\ 0.10,\ 0.10] \\ x_5 &: [0.40,\ 0.35,\ 0.25] \\ x_6 &: [0.20,\ 0.70,\ 0.10] \\ x_7 &: [0.10,\ 0.80,\ 0.10] \\ x_8 &: [0.15,\ 0.75,\ 0.10] \\ x_9 &: [0.20,\ 0.30,\ 0.50] \\ x_{10} &: [0.25,\ 0.25,\ 0.50] \end{aligned}$

Now assume we are using **top-1 routing**, so each token is sent to the expert with the largest probability. So the actual token counts are:

- expert 1 gets 5 tokens
- expert 2 gets 3 tokens
- expert 3 gets 2 tokens

	$f_1 = \frac{5}{10} = 0.5,\qquad f_2 = \frac{3}{10} = 0.3,\qquad f_3 = \frac{2}{10} = 0.2$

	$f = [0.5,\ 0.3,\ 0.2]$

Now let’s compute P_i, the average router probability for each expert across the batch.

	$P_1 = \frac{1}{10}(0.70+0.60+0.55+0.80+0.40+0.20+0.10+0.15+0.20+0.25)$
	$P_2 = \frac{1}{10}(0.20+0.30+0.25+0.10+0.35+0.70+0.80+0.75+0.30+0.25)$
	$P_3 = \frac{1}{10}(0.10+0.10+0.20+0.10+0.25+0.10+0.10+0.10+0.50+0.50)$
	$P = [0.395,\ 0.400,\ 0.205]$

Now compute the balancing term:

	$\sum_{i=1}^3 f_i P_i$
	
	$=(0.5)(0.395) + (0.3)(0.400) + (0.2)(0.205)$
	$= 0.3585$


Right now in looking at the sampled counts and softmax probabilities, we can see that most tokens went to expert 1 in this batch, and theoretically the router tends to favor experts 1 and 2 much more than expert 3.

	$f = [0.5, 0.3, 0.2] ,\qquad P = [0.395, 0.400, 0.205]$

For comparison, if everything were perfectly balanced and we have an even distribution, we would have:

	$f = [1/3, 1/3, 1/3],\qquad P = [1/3, 1/3, 1/3]$

Then the loss is much lower than the above example:

	$\sum_i f_i P_i = 3 \cdot \left(\frac13 \cdot \frac13\right) = \frac13 \approx 0.3333$

So this loss function encourages an even distribution of probabilities & sampling.


## DeepSeek's Improvements

DeepSeek introduces an additional parameter b_i that adjusts the routing scores for each expert. Instead of using only the router score s_i produced by the softmax, the model adds a bias term:

$s_i' = s_i + b_i$

The purpose of this bias is to help **balance expert usage**. 

If an expert has been receiving too few tokens over recent steps, its bias b_i is increased so that its effective score becomes larger, making it more likely to be selected by the router. Conversely, if an expert is receiving too many tokens, its bias is decreased so that its score becomes smaller and the router is less likely to choose it.

NOT GOING INTO THE MATH HERE.

---
## MoE's Sad Problems


While we've already seen problems with MoE, there are more practical challenges!

Routing Imbalance: If too many tokens are routed to the same expert, especially when experts are distributed across GPUs, that GPU can become overloaded. Because each expert has a fixed processing capacity per batch, excess tokens are dropped, which wastes computation and harms training efficiency.

Training Stability: MoE models often show more unstable loss behavior during training because the routing decisions change dynamically as the router learns.

Overfitting in Fine-Tuning: As the model has a very large number of parameters across all experts, a limited domain-specific dataset may not be sufficient to properly train or adapt all of them.

## UpCycling

A way to 'attempt' to side step these problems is by training a normal LLM, but then after training, add in an additional router layer in the middle, and just copy and paste the existing trained MLP layer as different 'experts', and continue training by pretending we are doing MoE from the gourd up again.
- In this sense, we just train a small router + fine tune MLP layers

![[Screenshot 2026-03-12 at 2.06.19 PM.png|600]]