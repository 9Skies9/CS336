The big idea of having this class is that we want to understand LLMs from their core and build everything from scratch.

But... we can't scale up to modern size LLMs (who got that many GPUs?), there are still the takeaways from the class:
- mechanics: how LLMs work
- mindset: optimizing the hardware used
- intuitions: making good modeling decisions
	- actually no... cuz most of deep learning is hard to justify WHY it works, it just works (from trying it out)

---

## The Bitter Lesson

- ![[Stanford CS336 Language Modeling from Scratch _ Spring 2025 _ Lecture 1_ Overview and TokenizationPT9M35.113S.jpeg|Stanford CS336 Language Modeling from Scratch | Spring 2025 | Lecture 1: Overview and Tokenization - 09:35|150]] [09:33](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=575#t=09:33) 

This is from the father of reinforcement learning, and here are his thoughts on deep learning.

<iframe 
    src="http://www.incompleteideas.net/IncIdeas/BitterLesson.html"
    style="
        width: 100%;
        height: 400px;
        background-color: #f5f5f5;
        border: none;
    ">
</iframe>

The big idea I see from 'The Bitter Lesson' is that algorithms that are able to scale with computation availably will always output perform algorithms that are 'cleverly crafted' based off replicating existing human knowledge.

And the perspective the teacher has too:

```
accuracy = efficiency x resources
```

---
## The Landscape of NLP

The NLP architectures really only had it's foundations in the 2010s.

- 2014 — Seq2Seq + Attention + Adam
- 2017 — Transformers + Mixture of Experts
- 2018–2019 — Large-scale model parallelism

And towards the 2020s is when language models at large scale training saw sudden 'bursts' of intelligence.

![[Screenshot 2026-03-10 at 11.37.44 PM.png|500]]

- 2019 — GPT-2 shows large language models can generate fluent text
- 2020 — Scaling laws + GPT-3 establish the scaling paradigm
- 2022 — PaLM pushes extreme scale, Chinchilla refines optimal scaling strategy

However, those were closed source in how they were made, and open source models have also made the fair share in advancing LLMs

- 2022 — Large collaborative open models (OPT, BLOOM)
- 2023 — LLaMA catalyzes the modern open-model ecosystem
- 2024 — Rapid expansion (Qwen, DeepSeek, OLMo)

Today there's... well, this list changes time to time.


Then there's a long discussion about the course's logisitcs... and then tokenizers.

---
## Tokenizers

Oh god... Andrew's video on tokenization was... beyond confusing personally, but oh well.

Simply, tokenizers convert text into integer values (a token), and it needs to be reversible.

![[Pasted image 20260310235815.png|500]]

Usually we want a higher compression in tokenization, meaning reducing text into fewer number of tokens, while it's generally beneficial for model efficiency (as less tokens mean less computation), it's not a guaranteed for increasing performance.

- Character tokenizer: each Unicode character becomes one token. Simple and reversible, but the vocabulary is large and many characters are rare.

- Byte tokenizer: text is represented as UTF-8 bytes (0–255). Vocabulary is fixed at 256 and covers everything, but sequences become long.

- Word tokenizer: split text into words and punctuation. Shorter sequences, but vocabulary explodes and unseen words cause problems.

- BPE tokenizer: an idea that started from a compression algorithm, it keeps a manageable vocabulary while keeping sequences reasonably short.
	- See more about BPE [here](https://www.youtube.com/watch?v=HEikzVL-lZU)

## Details of BPE



However... tokenization is a sad thing, as Andrew Kaparthy pointed out:

![[Screenshot 2026-03-11 at 12.21.24 AM.png|500]]

But until the day we can get rid of it, we have to deal with tokenization.