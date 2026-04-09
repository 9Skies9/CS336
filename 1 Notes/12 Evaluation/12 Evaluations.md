We saw from [[What The Heck Are All The Datasets?]] to see what benchmarks are.

Andrew Kaparthy gave a tweet about his opinions on these benchmarks: "My reaction is that there is an evaluation crisis. I don't really know what metrics to look at right now. (...) TLDR my reaction is I don't really know how good these models are right now."

If everyone trains on the same data, make similar architectures... what's even the difference between LLM 1 and LLM 2?

The instructor of the class said evaluation doesn't have a single point, just as how we don't have a single dataset to measure everything of an LLM, just some abstract goal that needs to turn into some number to determine how an LLM is doing on something.

I don't think this class is very 'packed' with content, it's more of a breadth sweep at a bunch of LLM benchmarks.

---

## Evaluation Pipeline

Evaluation, just like making LLMs is a path full of questions:

1. What are the **inputs**?
2. How do **call** the language model?
3. How do you evaluate the **outputs**?
4. How to **interpret** the results?


This is just copied over since I don't have better questions, and I think the detailed questions themselves already provide some directions.

How do you call the language model?
1. How do you prompt the language model?
2. Does the language model use chain-of-thought, tools, RAG, etc.?

How do you evaluate the outputs?
1. What metrics do you use (e.g., pass@k)?
2. Do you factor in cost (e.g., inference + training)?
3. How do you factor in asymmetric errors (e.g., hallucinations in a medical setting)?
4. How do you handle open-ended generation (no ground truth)?

How do you interpret the metrics?
1. How do you interpret a number (e.g., 91%) - is it ready for deployment?
2. How do we assess generalization in the face of train-test overlap?
3. Are we evaluating the final model or the method?

In summary, evaluation is still a vey subjective thing in general.

---
## History of Evaluation In LLMs

Before we started making LLMs, we used 'perplexity' to evaluate a language model, without going into details, lower perplexity = less uncertainty / more confidence in the correct next token.

Researchers before assumed perplexity meant the model better captured the statistical structure of language, as back in the days (and still now), language models only cared about a probability distribution for the next possible token.

This were some of the older datasets for testing language models.
- Penn Treebank (WSJ), WikiText-103 (Wikipedia), One Billion Word Benchmark (from machine translation WMT11 - EuroParl, UN, news)

Since GPT-2 and GPT-3, language modeling papers have shifted more towards downstream task accuracy.
- A model could memorize the training data (low perplexity on seen text) but fail to reason or follow instructions.
- What users care about (helpfulness, factuality, creativity, safety) doesn't correlate perfectly with next-token surprise.

So today, we evaluate LLMs with task-based benchmarks and human preference.

---
## Evaluation Overlaps!

Sometimes... the benchmarks themselves are open source on the internet, and it's very likely these LLMs trained on the benchmarks themselves, and achieved a high score.

It's just how it is... model makers don't report this.