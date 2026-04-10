Usually, training is divided into pre-training and post training.

- pre-training: train on raw text, predict next token -> loss -> update parameters
- post-training: train on quality instruction data, follow instructions -> loss -> update parameters

There is some new names like 'mid training', which is still pre-training, but trying to have higher quality pre-training data.

Here's an example of datasets picked from OLMo:

![[Screenshot 2026-04-08 at 6.53.30 PM.png|500]]

![[Screenshot 2026-04-08 at 6.53.46 PM.png|500]]

And Post Training Datasets from Lambert:

![[Screenshot 2026-04-08 at 6.55.25 PM.png|500]]


What are these datasets? Why these datasets? How are they used? Again... this isn't like a crazy class, it's more breadth over depth, a walk in history.

---
## Pre-training

In the old days of BERT, it trained on a book corpus called Smashwords and Wikipedia.

Then we had GPT 2's webtext, and common crawl.

| Database Name                     | Data Amount (roughly)                                                          |
| --------------------------------- | ------------------------------------------------------------------------------ |
| Smashwords (BooksCorpus)          | 500,000 books / 150,000 authors (~800 million words)                           |
| Wikipedia (all language editions) | 62 million articles across 329 languages (English: ~6.8M articles)             |
| GPT‑2 WebText                     | ~8 million documents, 40 GB of text (pages from Reddit outlinks with ≥3 karma) |
| Common Crawl                      | over 2.7 billion web pages and 468 TiB of uncompressed webpage content         |

## Crawl

How to common crawl is another story!

![[Screenshot 2026-04-08 at 7.37.04 PM.png|300]]

It's not the 'best' crawl of the internet, as they respects robots.txt (of the websites), and don't overload the website's server, and also don't have much of a filter of what to download (yay offensive content baby).

Usually, most of these LLM companies nowadays have their own crawlers! Since common crawl isn't the smartest crawl of the internet.

---
## And More...

There's a lot of these down the road and I don't think i want to name all of them, like github's code, stack exchange's discussions... there are more!

There was 1 notable one which I liked, hugging face's FineWeb, again I'm not going to go into the details of how they've done it, here's just an image for a slap.

![[Pasted image 20260408202218.png|500]]

This filtered about 90% of the tokens of common crawl, nemotron (from nvidia) thought that's removing too much tokens and tried to increase it a bit.

---
## Post Training

Apparently not as much... it's hard to get a LOT of data for supervised fine tuning as they have to follow the LLM chatting style interface, and data on a grand scale was difficult to get.

