https://www.bilibili.com/video/BV1PAUrBVEc8

Usually when an LLM is released, they would usually show 'exciting progress', saying it's the most 'smartest model out there' or something like that.

Then they would pull up a gigantic graph filled with numbers, bars and say 'OMG highest scores ever AGI on the horizon???' Some stuff like that.

![[【闪客】大模型是如何测试的？以Gemini3为例PT14.536S.jpeg|【闪客】大模型是如何测试的？以Gemini3为例 - 00:14|500]] 

To the average joe, nobody knows what these are, but through good marketing, we highlight the bar, bold the numbers, do some fancy graphics... and everyone's happy.

But what the heck are these? Are models, from a subjective standpoint on a person's day to day use getting any smarter? Let's do a little dive into them.

## Benchmarks != Smart

Just like the real world, nerds aren't always the ones with the money and power, all of these 'benchmarks' are in one way or another deterministic, like answering yes or no questions, multiple choice questions, do the numbers add up, does the code execute, is the Capital of China Beijing...

If you take a look out there, there's no universally agreeable benchmarks on generating images or writing fanfic. The best/closest we've come to it is LLM arena, where people subjectively do majority voting for which response they like when comparing LLMs.

But... what are these LLMs being tested upon then? Let me explain.

## The Big OL List

Taking from Gemini 3's list, there's a big bunch, but there's roughly 7 categories:

- Reasoning & Academics
- Images
- Video
- Code
- Facts
- Tools
- Longgggggg Context

![[【闪客】大模型是如何测试的？以Gemini3为例PT36.133S.jpeg|【闪客】大模型是如何测试的？以Gemini3为例 - 00:36|500]] 

This isn't like a detailed guide on how to make a dataset, just a big overview as to what some of these datasets are (and what the heck your LLMs are being trained on)


## Humanity's Last Exam

As the name suggests, this is humanity's last exam (it's super hard!), the questions came from over 500 institutions from 50 countries (about 1000 experts!). 

The experts are given a huge pool of money so they'd actually be willing to make the exams, and it has to be so difficult such that LLMs are basically taking a guess.

At the end of the day... it's still providing a literal number, some piece of text, or multiple choice, take a look at a question:

![[Screenshot 2026-04-08 at 4.33.54 PM.png|500]]


## ARC AGI

This is like uh... what would I call this? Like one of those IQ tests for kids? Or like a little video game to spare your time?

![[Screenshot 2026-04-08 at 4.37.43 PM.png|200]]

Like in their demo, your job is to move the block around using up, down, left, right to the symbol, but the symbol has to correspond to the bottom left corner, and there's a limit to how many steps you can take.

Reminds me of like RL + Atari Games... or Baba is You (no fucking AI be solving that).


## Some Others

GPQA stands for 'A graduate level google proof Q & A Benchmark'

AIME is just AIME! That math competition (except almost everyone gets it correct).

MathArena is unsolved problems in mathematics.

MMLU is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, MMMLU is that + multi-lingual.


## Images

More datasets! MMMU-Pro is basically MMLU for images!

![[Screenshot 2026-04-08 at 5.03.34 PM.png]]

Screenshot Pro is just as it's name suggests, testing how well an LLM can understand a screenshot. 

And etc etc.


## Code!

This isn't hard to understand! LiveCodeBench is a pretty known one, they keep updating the benchmark to reduce the likelihood of data contamination.

![[Screenshot 2026-04-08 at 5.07.28 PM.png]]

This is what they said about themselves:
- LiveCodeBench collects problems from periodic contests on LeetCode, AtCoder, and Codeforces platforms and uses them for constructing a holistic benchmark for evaluating Code LLMs across variety of code-related scenarios continuously over time.

Terminal Bench is seeing how well LLMs can perform tasks in the CLI, and SWE bench tests about if they can solve real Github issues.

Here's their official example.

![[Screenshot 2026-04-08 at 5.09.31 PM.png]]

## Factuality

FACTS Benchmark tessts on LLM hallucinations and factuality. And this is a lot HARDER of a test! Since it's no longer a direct yes or no, and so there's people to judge the outputs of these LLMs...? No, they use other LLMs to judge the LLMs.

Bruh.

---
## But Like...

Are we AGI AGI? Nah, fancy numbers don't mean crap unless the LLM experience has actually improved for you.

