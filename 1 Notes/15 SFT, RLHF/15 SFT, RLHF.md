Finally! Post Training! Actually getting helpful chatting assistants! This usually is done with SFT (supervised fine tuning) first, and then RL (reinforcement learning).

And this class will be mostly focused around the techniques demonstrated by Instruct GPT.

![[Pasted image 20260410012943.png|600]]

# SFT

The first step is supervised fine tuning.

![[Screenshot 2026-04-10 at 1.31.28 AM.png|500]]

At this point, we are trying to get instruction response pairs for the LLMs, which by themselves is already rare enough! How do different people tackle this problem? They should 3 possible solutions:

- FLAN - use existing instruction-response pairs that were used to train language models before transformers
	- ![[Screenshot 2026-04-10 at 1.39.34 AM.png|400]]

- Open Assist - mass crowd source (getting a lot of instruction-response pairs from random people who are willing to make them) making SFT instructions from sheer force
	- ![[Screenshot 2026-04-10 at 1.40.48 AM.png|400]]

- Alpaca - can LLMs generate instruction-response pairs (synthesized data)
	- ![[Screenshot 2026-04-10 at 1.42.29 AM.png|400]]

We can see the style of these SFT datasets are very different, and these styles carry over in the post-trained LLM. Like how long the responses are, how often do they use bullet points, how much breadth/depth of knowledge is in the dataset, how safe/biased it is.... they all affect the model.

Here is an image showing a bunch of different datasets, on average how long the instructions and responses are.

![[Screenshot 2026-04-10 at 1.49.44 AM.png|500]]

There was also testing from how 'biased' humans are by themselves in these LLMs' responses. People like more lists, as well as longer outputs.

![[Screenshot 2026-04-10 at 11.08.43 AM.png|500]]

But it's weird how these post training methods don't seem to always increase a model's benchmarking scores, here's a LLaMA 13 model, and the benchmarks of the model before & after using supervised fine tuning. 

![[Screenshot 2026-04-10 at 11.11.08 AM.png|500]]

It's counter intuitive! 'Knowledge' for LLMs is not something well understood, how they have memory of common facts etc, and there's no good way to force it (for now).

---
## Safety

While not what I'm most interested in... by default, we wouldn't want LLMs to respond to instructions like 'how do I kill someone' or 'how would I build a bomb', and it should learn to reject harmful requests.

Researchers found that in SFT, if you just mix in a little bit of these safety instructions, as little as 500 samples(like saying no to questions like how do I kill someone), you can significantly improve the model's safety.

![[Screenshot 2026-04-10 at 11.21.17 AM.png|500]]

But there's also potential problems with making the LLM overly safe guarded, a good example given was "How do I kill a python process" and the LLM freaks out hearing the work Kill and rejects your request.

![[Screenshot 2026-04-10 at 11.22.41 AM.png|200]]

---

## How to SFT

The class glazed over this and just said 'do gradient descent just like in pre-training' but I'm confused about it's details. SFT and pre-training boundaries are well getting blurred, because fundamentally both are conditional probability questions just phrased differently.

**Pre-training (Unconditional Language Modeling)**

In pre-training, you give the model a sequence of tokens $X = (x_1, x_2, \dots, x_T)$. The model tries to predict every single token based on all preceding tokens. The loss function is applied to the entire sequence:

$$\mathcal{L}_{PT} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

**SFT (Conditional Sequence Generation)**

In SFT, your sequence $X$ is actually a concatenation of an instruction/prompt of length $P$ and a response of length $R$, such that $T = P + R$.

You do _not_ want the model to learn how to generate the prompt (because you, the user, will provide that). You only want it to learn how to generate the response. Therefore, we still do a forward pass over the whole sequence, but we **mask** the loss for the prompt tokens:

$$\mathcal{L}_{SFT} = -\sum_{t=P+1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

In code, this is done by setting the label for the prompt tokens to a specific `ignore_index` (usually `-100` in PyTorch). The model still reads the prompt to build its internal context ($x_{<t}$), but the gradient descent only updates the weights based on how well it predicted the response tokens.


### The Wasted Padding Problem

We know GPUs require inputs to be perfectly rectangular tensors of shape `[batch_size, sequence_length]`.

If you have three SFT pairs of different lengths, you can't just throw them into a matrix. You have to pad the shorter ones with a dummy token (like `<PAD>`) until they match the length of the longest sequence in the batch.

**A standard padded batch looks like this:**

- **Row 1:** `[Prompt A] [Response A] <EOS> <PAD> <PAD> <PAD> <PAD>`
- **Row 2:** `[Prompt B] [Response B] <EOS> <PAD> <PAD>`
- **Row 3:** `[Prompt C] [Response C] <EOS>`

The GPU still has to do the heavy matrix multiplications for all those `<PAD>` tokens, even though we tell the loss function to ignore them. In a dataset with highly variable lengths, up to 50-70% of your GPU compute might be completely wasted multiplying zeros!

To make SFT look and act exactly like pre-training computationally, engineers use **Sequence Packing** (sometimes called Data Packing).

Instead of padding short sequences, we take multiple instruction-response pairs and concatenate them together, separated by an End-Of-Sequence (`<EOS>`) token, until we reach the model's maximum context length (e.g., 4096 tokens).

A packed batch looks like this:

- Row 1: `[Prompt A] [Response A] <EOS> [Prompt B] [Response B] <EOS> [Prompt C] [Resp...`

If `Response C` gets cut off at token 4096, that's fine—the rest of it just overflows into the start of the next row in the batch.

However, "Prompt B" might accidentally pay attention to "Response A" since they are in the same sequence. So this requires Block Diagonal Attention Masks (Document Masking) to completely sever the attention connections between different documents packed into the same row.
- It's a bit more annoying!

---

## Mid-Training?

As SFT and pre-training boundaries are getting blurred, and we just shown above how SFT could essentially be put in the same shoes of pre-training, people have started to call this phase 'mid-training' instead of post training.

![[Screenshot 2026-04-10 at 12.37.33 PM.png|500]]

If you look on the right, that 'decay stage' is mid training, where they start mixing SFT data into the pre-training data.

---

## RLHF! Finally

![[Screenshot 2026-04-10 at 12.52.58 PM.png|500]]

I want to introduce RL in my own little way, and I think I might have written something similar before, so here we go:

Right now up to this point, all of our objectives, fundamentally whether SFT or Pre-training is doing conditional probability for the next token given a sequence.

- P (some text | next token)
- P (some instruction | response) -> P (some instruction | next token) on repeat


But that fundamentally doesn't answer this problem:

- P (some instruction | GOOD response) out of P (some instruction | ALL possible responses)

Take this instruction response pair for example and you'll understand the issue:

- **Instruction:** "Write a polite email declining a job offer."

- Responses:
	- **Possible Response A (The Blunt Human):** "I am not taking the job. Thanks."
	
	- **Possible Response B (The Overly Verbose Human):** "Dearest Hiring Manager, words cannot express the profound sorrow I feel in my heart as I put pen to paper to inform you that my life's journey is taking me down a different path..."
	
	- **Possible Response C (The Ideal Assistant):** "Thank you so much for the offer. While I greatly appreciate your time and the opportunity, I have decided to accept another position that better aligns with my current career goals."


To an SFT model doing next-token prediction, Response A, B, and C are all technically correct. If Response A was in the training data because a tired annotator wrote it, the model will learn to output Response A. 
- SFT has no internal concept of "quality," "helpfulness," or "politeness." It only knows "likelihood based on the dataset."


That's where RL comes in, we reframe this question to deciding which possible response is the best response through a judge, in RLHF, this original judge is a human, which ranks responses and give them scores.

These human labelled scores are then used to train a reward model, then when we show the Reward Model Responses A, B, and C, it will give them scores

- Response A: -5 points (Too rude)
- Response B: +2 points (Polite, but weird)
- Response C: +10 points (Perfectly helpful and professional)

Now, we unleash an RL algorithm (like PPO - Proximal Policy Optimization) to optimize the model's parameters to maximize the reward model's score.


So from SFT:

-  Fit $\hat{p}(y|x) \approx p^*(y|x)$

	- $x$ is the Instruction (e.g., "Write a polite email declining a job offer").

	- $y$ is the Response (the sequence of tokens the model generates).

	- $p^*(y|x)$ is the Reference Distribution.
		- This is the human dataset! It represents the actual probability of how humans respond to that instruction. It is $P(\text{ALL possible human responses} \mid \text{Instruction})$.

	- $\hat{p}(y|x)$ is our LLM.


This is a "Pure generative modeling perspective." The model is just acting like a statistical parrot. 


For RLHF:

-  Find $\hat{p}(y|x)$ such that $\max_p \mathbb{E}_p[R(y,x)]$

	- $R(y,x)$ is the Reward Function. This is the "judge" for the LLMs outputs

	- $\mathbb{E}_p$ stands for the Expected Value. It represents the average score the model gets for its responses.
	
	- $\max_p$ means **Maximize**. We want to update our model's internal weights to get the highest possible expected score from the judge.


By using gradient descent to solve that maximization equation, we finally push the LLM to consistently output the best, safest, and most helpful responses, completely breaking free from the limitations of just copying messy human data!

And this is a lot cheaper too, getting humans to consistently provide SFT data is expensive.

![[Screenshot 2026-04-10 at 1.41.55 PM.png|500]]

---
## How To RLHF?

This was already rather clear in the instruct GPT diagram:

1. model gets prompt, generates multiple answers

2. label person ranks answers in showing which one is better/worser

3. reward model learns to generate scores for each answer such that the number rankings match with the person rankings.


For some funsies, the instructor was actually able to dig up the literal annotation guidelines for the annotators, like here is Google Bard's crowd sourcing labeling requirements.

![[Screenshot 2026-04-10 at 2.21.16 PM.png|600]]

No one has to read that... basically ' is the AI being helpful and organize information so it's easy to understand'.

To a certain point, you could replace these annotators with just another LLM! And a few LLM labs tried to use AI feedback for this RLHF process.

![[Screenshot 2026-04-10 at 2.42.49 PM.png|500]]

## The Math

The math of RLHF is PPO, and more recently, DPO. Then we have RLVR (reinforcement learning with verifiable feedback)

I don't think I'll understand their explanation... `¯\_(ツ)_/¯`, RL... what a distant time that I forgot, I'll go look at some other explanation before coming back to the next class.