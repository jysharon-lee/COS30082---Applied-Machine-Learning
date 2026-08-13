# 📘 Week 10 Study Notes: Generative Deep Learning (GANs)

> **Course:** COS30082 — Applied Machine Learning
> **Topic:** Generative Deep Learning — Generative Adversarial Networks (GANs)
> **Prerequisite:** Weeks 1–9 (Neural Networks, CNNs, supervised vs unsupervised learning basics)

---

## Table of Contents

1. [What Is a Generative Model?](#1-what-is-a-generative-model)
2. [Generative vs Discriminative — A Story](#2-generative-vs-discriminative--a-story)
3. [Generative vs Discriminative — The Formal View](#3-generative-vs-discriminative--the-formal-view)
4. [Introduction to GANs](#4-introduction-to-gans)
5. [The Generator: Turning Random Numbers into Objects](#5-the-generator-turning-random-numbers-into-objects)
6. [The Discriminator: A Judge, Not a Creator](#6-the-discriminator-a-judge-not-a-creator)
7. [Generator + Discriminator Working Together](#7-generator--discriminator-working-together)
8. [Training Algorithm Overview](#8-training-algorithm-overview)
9. [Step-by-Step Training Procedure (The Math)](#9-step-by-step-training-procedure-the-math)
10. [Worked Example: Reading the Manga Generation Results Across Epochs](#10-worked-example-reading-the-manga-generation-results-across-epochs)
11. [GAN as Structured Learning](#11-gan-as-structured-learning)
12. [Why Structured Learning Is Challenging](#12-why-structured-learning-is-challenging)
13. [GAN as a Solution to Structured Learning](#13-gan-as-a-solution-to-structured-learning)
14. [Can the Generator Learn Alone? (Auto-Encoders)](#14-can-the-generator-learn-alone-auto-encoders)
15. [Worked Example: Reading the Pixel-Error Diagram](#15-worked-example-reading-the-pixel-error-diagram)
16. [Can the Discriminator Learn Alone?](#16-can-the-discriminator-learn-alone)
17. [Pros and Cons: Generative vs Discriminative](#17-pros-and-cons-generative-vs-discriminative)
18. [The Benefit of Combining Both (Why GAN Works)](#18-the-benefit-of-combining-both-why-gan-works)
19. [Conditional Generation by GAN](#19-conditional-generation-by-gan)
20. [Worked Example: Reading Matched vs Unmatched Pairs](#20-worked-example-reading-matched-vs-unmatched-pairs)
21. [Image-to-Image Translation with GAN](#21-image-to-image-translation-with-gan)
22. [Other Conditional GAN Applications](#22-other-conditional-gan-applications)
23. [Unsupervised Conditional Generation (No Paired Data)](#23-unsupervised-conditional-generation-no-paired-data)
24. [CycleGAN: Solving the "Ignored Input" Problem](#24-cyclegan-solving-the-ignored-input-problem)
25. [Summary: All Models Side by Side](#25-summary-all-models-side-by-side)
26. [Key Takeaways](#26-key-takeaways)
27. [Glossary](#27-glossary)
28. [Study Tips for Week 10](#28-study-tips-for-week-10)

---

## 1. What Is a Generative Model?

Before GANs make sense, we need to place them on the map of machine learning.

| | **Supervised Learning** | **Unsupervised Learning** |
|---|---|---|
| **Data** | Has both input $x$ and label $y$ | Only has input $x$, no label |
| **Goal** | Learn a function that maps $x \to y$ | Discover hidden structure in the data |
| **Examples** | Regression, classification, object detection | Clustering, dimensionality reduction, feature learning |

A **Generative Model** sits inside unsupervised learning. Its job is to **learn the true data distribution** of a training set well enough that it can **generate brand-new data points** that look like they belong to that same set — with some natural variation.

$$
\boxed{\text{Generative Model} = \text{Learn } p(x) \Rightarrow \text{Sample new, realistic } x}
$$

> **Analogy:** Think of a supervised model as a **student who memorises answer keys** — given a question ($x$), they learn to spit out the matching answer ($y$). A generative model is more like a **student who studies thousands of paintings so thoroughly** that they can eventually **paint an entirely new picture** in the same style, without copying any single one exactly. They're not answering a question — they're learning the *style/pattern itself* well enough to produce fresh examples of it.

---

## 2. Generative vs Discriminative — A Story

A simple story to separate two ways of "knowing" something. **A father has two kids, Sharon and Ivan.** Sharon learns *everything in depth* about what she sees. Ivan only learns the *differences* between things.

One day the father takes them to a tiny zoo with just **lions** and **elephants**. Afterwards, he points to an animal and asks: *"Is this a lion or an elephant?"*

- **Sharon** mentally re-draws both a lion and an elephant from memory, compares the animal in front of her to her two drawings, and picks whichever drawing is the closest match. She answers **"Lion"** — this is a **Generative Model**: she has learned what each class *actually looks like* (he can "regenerate" it), and classifies by comparing to his own generated version.
- **Ivan** never bothered learning what a full lion or elephant looks like — he only picked up a handful of *distinguishing features* (trunk vs. mane, size, ears) and uses those differences directly. He also answers **"Lion"** — this is a **Discriminative Model**: he draws a decision boundary between classes without ever needing to know what a "complete" lion looks like.

Both of them get the same right answer, but they got there in **completely different ways**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/43248135-890f-458c-b061-3c46a956f97b">
</p>

<p align="center">
  <em>Sharon (left image) vs Ivan (right image)</em>
</p>

> **Analogy in simpler terms:** Imagine two ways to tell a fake $50 note from a real one. Kid A (generative) has studied real $50 notes so closely that he could **draw one from memory** — he spots a fake because it doesn't match his mental "template." Kid B (discriminative) never learned to draw a note at all — he just memorised 3–4 quick tells (paper texture, watermark angle, colour-shifting ink) and checks only those. Kid B is faster and often just as accurate, but he'd be **useless if you asked him to draw a $50 note from scratch** — he genuinely doesn't know what the "whole" note looks like, only the differences that matter for telling real from fake.

---

## 3. Generative vs Discriminative — The Formal View

| | **Generative Model** | **Discriminative Model** |
|---|---|---|
| **Model representation** |<p align="center"><img src="https://github.com/user-attachments/assets/f71053b8-d055-4c33-8585-3474add02d70" width=600></p> | <p align="center"><img src="https://github.com/user-attachments/assets/7fe43266-20cf-4122-8b04-4eef92721175" width=600></p> |
| **Informally** | Can *generate* new data instances | *Discriminates* between kinds of data instances |
| **Formally** | Captures the **joint probability** $p(x, y)$, or just $p(x)$ if there are no labels | Captures the **conditional probability** $p(y \mid x)$ |
| **What it "knows"** | The full shape of each class (how the data is distributed) | Only the boundary that separates classes |

$$
\text{Generative: } p(x, y) \qquad \text{vs.} \qquad \text{Discriminative: } p(y \mid x)
$$

> **Analogy:** $p(x, y)$ is like knowing **everything about how a town is laid out** — where every house, shop, and road is — so you could redraw the whole map from memory. $p(y \mid x)$ is like only knowing **which side of a single dividing river** each address falls on — enough to answer "north side or south side?" instantly, but not enough to draw the town itself.

---

## 4. Introduction to GANs

A **Generative Adversarial Network (GAN)** is a generative model made of **two neural networks that compete against each other**:

- A **Generator (G)** that tries to create fake data realistic enough to pass as real.
- A **Discriminator (D)** that tries to tell real data apart from the Generator's fakes.

> **Analogy:** This is the classic **art forger vs. detective** setup. The **forger (Generator)** starts out painting clumsy, obviously fake artworks. A **detective (Discriminator)** looks at both real paintings and the forger's fakes and learns to spot the difference. Every time the detective catches a fake, the forger studies *why* it got caught and improves. Every time the forger produces something convincing, the detective has to get sharper to keep catching it. Round after round, **both** get better — until, ideally, the forger's paintings are good enough to fool even a very sharp detective.

---

## 5. The Generator: Turning Random Numbers into Objects

The **Generator** is simply a neural network (or function) that takes in a **vector of random numbers** and outputs an object — an image, a sentence, audio, and so on.

$$
G: z \rightarrow x, \qquad z \sim \text{(some simple distribution, e.g. Normal)}
$$

- **Image generation example:** feed a vector like $[0.5, -0.3, \dots, 0.1]$ into the Generator, and it outputs a full anime-style face image.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7fe43266-20cf-4122-8b04-4eef92721175" width=600>
</p>

- **Sentence generation example:** feed the *same kind* of random vector in, and instead of pixels, the Generator outputs a sentence like *"Good morning"* or *"Thank you very much."*

<p align="center">
  <img src="https://github.com/user-attachments/assets/e511bb7c-1528-4e98-b04b-bec95d8c59c0" width=600>
</p>

**Each dimension of the input vector can end up controlling a different characteristic of the output** — for example, one dimension might end up controlling *hair texture*, another might control *how open the mouth is*, purely because that's the structure the Generator learned to be useful during training.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d4e251f1-ee37-413f-97f3-951894c7dba2" width=900>
</p>

> **Analogy:** Think of the input vector as a set of **sliders on a character-creation screen** in a video game (like adjusting hair length, eye colour, mouth shape). You didn't design those sliders yourself — the Generator "discovered," through training, that certain slider combinations reliably produce certain visual features. Turning one slider slightly (say, from 0.2 to 0.8 in the "hair texture" dimension) smoothly changes that one feature in the output face, without necessarily touching anything else.

---

## 6. The Discriminator: A Judge, Not a Creator

The **Discriminator** is also just a neural network (or function), but its job is the opposite of the Generator's: instead of creating something, it **judges** something.

$$
D: x \rightarrow \mathbb{R} \quad \text{(a scalar score)}
$$

- **Input:** an object $x$ (e.g. an image).
- **Output:** a single number — **larger means "looks real,"** **smaller means "looks fake."**

For example, a real, crisp anime face might get a score of **1.0**, while an early, blurry, noisy Generator output might get a score of **0.1**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4ceba91-1ded-4c2a-994c-3640c2d4bf19" width=600>
</p>

> **Analogy:** The Discriminator is like a **restaurant food critic** who doesn't cook anything themselves — they just taste a dish and give it a score out of 10. A perfectly plated, authentic dish scores near the top; a clumsy imitation with the wrong texture and taste scores near the bottom. The critic never needs to know *how* to cook to be good at spotting a bad imitation.

---

## 7. Generator + Discriminator Working Together

When trained together over many rounds, the Generator's outputs steadily improve because it's constantly getting feedback from an ever-improving Discriminator. **Manga (anime face) generation** example: a new Generator (V1) produces images, a Discriminator (V1) judges them; then the Discriminator is *upgraded* (V2, V3...) as the Generator's fakes get better, so it never falls behind.

<p align="center"><img src="https://github.com/user-attachments/assets/30e19108-e0a5-4279-bfc4-91528bc35c3a" width=800></p>

> **Analogy:** Picture a video game where a **boss (Discriminator) levels up every time the player (Generator) beats it**. If the boss never got stronger, the player would stop improving once they found one easy trick. Because the boss keeps getting tougher, the player is constantly pushed to develop **genuinely better** strategies — not just tricks that work against a weak, outdated opponent.

---

## 8. Training Algorithm Overview

GAN training alternates between two steps, over and over, in every iteration:

**Step 1: Fix the Generator, update the Discriminator.**
- Sample real objects from the database and label them **1** (real object).
- Sample random vectors, run them through the (frozen) Generator to get fake objects, and label them **0** (fake/generated object).
- Update $D$ so it gets better at telling these two groups apart.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b65f99b1-9f83-4c4d-a8f6-341d8d054fcc" width=600>
</p>

**Step 2: Fix the Discriminator, update the Generator.**
- Sample new random vectors, run them through the Generator to get fake objects.
- Feed those fakes into the (frozen) Discriminator.
- Update $G$ so that the Discriminator's score on these fakes goes **up** — i.e. the Generator learns to "fool" the Discriminator. This is done via **gradient ascent**, since $G$ and $D$ are actually one connected network end-to-end during this step.

> **Analogy:** This is exactly like **taking turns in a game of tag** where the roles alternate. In Step 1, the detective (Discriminator) studies today's batch of real paintings and today's forgeries and sharpens their eye — the forger just stands still. In Step 2, the forger (Generator) tries out new brush techniques specifically aimed at whatever the detective is currently good at catching — the detective just stands still and grades. They never train **at the exact same time**; they take turns, one improving while the other holds steady, which keeps the competition fair and stable.

---

## 9. Step-by-Step Training Procedure (The Math)

Initialise parameters $\theta_d$ for $D$ and $\theta_g$ for $G$. In each iteration:

**Learning D** (maximise $\tilde{V}$ — we *want* $D(\text{real objects}) \to 1$ and $D(\text{fake objects}) \to 0$):

$$
\tilde{V} = \frac{1}{m}\sum_{i=1}^m \log D(x^{(i)}) + \frac{1}{m}\sum_{i=1}^m \log\left(1 - D(\tilde{x}^{(i)})\right), \qquad \tilde{x}^{(i)} = G(z^{(i)})
$$

$$
\theta_d \leftarrow \theta_d + \eta \nabla \tilde{V}(\theta_d)
$$

**Learning G** (maximise $\tilde{V}$ — we want the Discriminator to score fakes highly):

$$
\tilde{V} = \frac{1}{m}\sum_{i=1}^m \log\left(D\big(G(z^{(i)})\big)\right)
$$

$$
\theta_g \leftarrow \theta_g + \eta \nabla \tilde{V}(\theta_g)
$$

**How to read this, plainly:** the plus sign (`+`) in both update rules is there because we're doing **gradient ascent** (climbing *up* toward a higher score), not the usual gradient *descent* (rolling *down* toward a lower loss) you may be used to from ordinary classification training. That's because both terms are things we want to **maximise** — $D$ wants to maximise its accuracy at telling real from fake, and $G$ wants to maximise how "real" the Discriminator thinks its fakes are.

> **Analogy:** Normal training is like **walking downhill to the bottom of a valley** (minimising error). Here, both networks are climbing **uphill toward a better score** — $D$ climbing toward "I can always tell real from fake," and $G$ climbing toward "I can always fool the Discriminator." Same mountain, opposite peaks — and since they share the same connected network in Step 2, moving $G$ up its hill directly depends on how the terrain looks from $D$'s current position.

---

## 10. Worked Example: Reading the Manga Generation Results Across Epochs

The *same* GAN's output is shown at four checkpoints: **epoch 1, epoch 10, epoch 200, and epoch 300**. This is a great habit to build: whenever you're shown a "before/after training" grid of generated images, read it the same deliberate way you'd read any other performance chart.

| Epoch | Figure | What you'd typically see | How to interpret it |
|---|---|---|---|
| **1st epoch** | <p align="center"><img src="https://github.com/user-attachments/assets/c84ff4f5-5236-460c-a1b2-5891218ed045" width=500></p> | A grid of pink/purple static-like blobs — no recognisable faces at all | The Generator is still close to random; it hasn't learned any real structure yet, and the Discriminator is easily telling these apart from real faces |
| **10th epoch** | <p align="center"><img src="https://github.com/user-attachments/assets/df497c90-1c67-47d6-87d9-3a72f1802126" width=500></p> | Blurry but face-*shaped* blobs — eyes and hair regions start to emerge | The Generator has picked up on the **coarse, big-picture structure** (a face has eyes near the top, hair around the edges), but fine detail is still missing |
| **200th epoch** | <p align="center"><img src="https://github.com/user-attachments/assets/9f3c0025-c3f7-464f-a1fd-9d8ee1292e43" width=500></p> | Recognisable anime faces, but colours and details are sometimes a bit off or inconsistent | The Generator now nails the overall structure and is refining finer details, textures, and colour consistency |
| **300th epoch** | <p align="center"><img src="https://github.com/user-attachments/assets/ea2e4ce0-57c8-476f-a942-fc7bf994e511" width=500></p> | Sharper, more consistent, more varied faces | Both networks have had time to co-evolve — G produces convincing detail, and D is strict enough to have forced that improvement |

**How to read this kind of grid in general:** don't just ask "does it look good?" — ask **what specific level of detail changed** between checkpoints. Early epochs usually fix **global structure** first (is there a face-shaped blob at all?) before **local detail** (are the eyes sharp? is the colour right?) improves later. This mirrors how the Discriminator itself improves — early on it can only tell "definitely fake blob" from "real face," so that's the only signal the Generator gets; later, once the big picture is solved, the Discriminator starts penalising finer flaws, which is the only way the Generator can be pushed to fix them.

> **Analogy:** This is like watching a student learn to sketch portraits over months. In week 1, they can barely draw an oval with two dots (epoch 1). By week 2 they've got the proportions of a face roughly right, even if it's messy (epoch 10). A few months in, the face is clearly recognisable, though some details are still slightly off (epoch 200). After sustained practice, the sketches become genuinely convincing (epoch 300). At every stage, the *feedback* they're getting (from a teacher who is also getting better at spotting flaws) determines what they fix next — big proportions before fine shading.

---

## 11. GAN as Structured Learning

Machine learning, at its core, is about finding a function $f: x \rightarrow y$. But the *shape* of $y$ changes what kind of problem you're solving:

| Task | Output $y$ |
|---|---|
| **Regression** | A single scalar (a number) |
| **Classification** | A "class" — a one-hot vector |
| **Structured Learning / Prediction** | A **sequence, matrix, graph, or tree** — output made of components **with dependencies** on each other |

GAN-style generation (images, sentences) falls squarely into **structured learning**, because the output isn't just one number or one label — it's a whole *composed object* where every part needs to make sense **together**.

**Examples of structured output:**
- **Machine Translation:** $x = \text{"She very loves cooking"}$ → $y = \text{a grammatically coherent Japanese sentence}$
- **Speech Recognition:** $x = \text{an audio waveform}$ → $y = \text{a full transcribed sentence}$
- **Chat-bot:** $x = \text{"Hello, how are you?"}$ → $y = \text{"Thank you, I am fine."}$
- **Image-to-Image translation:** turning building labels into a photorealistic facade, black-and-white photos into colour, day scenes into night scenes, or sketches into photos.
  
  <p align="center"><img src="https://github.com/user-attachments/assets/0b0842f9-e5ba-49cd-a21d-89488268c22b" width=600></p>
  
- **Text-to-Image translation:** turning a sentence description into a matching picture.

  <p align="center"><img src="https://github.com/user-attachments/assets/4ceb779c-2c97-4e09-b83d-a1b863c573ec" width=600></p>

> **Analogy:** Predicting a single class label is like answering **"true or false"** on a quiz — one clean answer, done. Structured learning is like being asked to **write an entire essay** — every sentence has to individually make sense **and** connect logically to the sentences around it. You can't just generate word 1, then word 2, then word 3 completely independently and hope they add up to something coherent; the *whole thing* has to hang together.

---

## 12. Why Structured Learning Is Challenging

Two specific reasons make structured learning hard:

**1. One-shot / Zero-shot Learning problem.** In ordinary classification, each class usually has plenty of training examples. But in structured learning, if you treated **every possible output** (every possible sentence, every possible image) as its own "class," the output space would be **astronomically huge** — most possible outputs simply have **zero** training examples. The model has to **create new, never-seen-before outputs** at test time, which demands more "intelligence" than simply picking the closest matching example it memorised.

**2. The planning problem.** The model typically generates the output **piece by piece** (pixel by pixel, word by word), but it needs to have a **big-picture plan in mind** the whole time, because the individual pieces are **not independent** — they depend on each other. If each piece is generated without considering the others, the result can look locally fine but be globally broken.

> **Analogy for the zero-shot problem:** Ordinary classification is like a multiple-choice exam where you just pick from options **A, B, C, or D** — you've seen similar options in your revision. Structured generation is like being asked to **compose an entirely new song** — there's no fixed, finite list of "correct songs" you could have memorised in advance; you have to generate something new that still *sounds* musically coherent.

> **Analogy for the planning problem:** Imagine writing a mystery novel **one word at a time, in order, with total amnesia about your outline** — you'd risk introducing a murder weapon in chapter 10 that was never mentioned earlier, or giving a character two different names. A good writer keeps the **whole plot in mind** while writing any single sentence, even though the sentences are still produced one after another.

---

## 13. GAN as a Solution to Structured Learning

GANs address both challenges above by combining **two complementary viewpoints**:

| Component | Direction | Role |
|---|---|---|
| **Generator** | **Bottom-up** | Learns to generate the object piece by piece, at the component level |
| **Discriminator** | **Top-down** | Evaluates the **whole** object at once, and can therefore judge whether all the pieces fit together |

<p align="center">
  <img src="https://github.com/user-attachments/assets/72f3a1fe-c0b7-4245-9327-e2b03ffe127d" width=800>
</p>

> **Analogy:** This is like a construction site with **two different roles**. The **bricklayer (Generator)** works bottom-up, placing one brick at a time. Left alone, a bricklayer without an overall blueprint might build a wall that's locally neat but globally the wrong shape. The **building inspector (Discriminator)** works top-down — they don't lay a single brick themselves, but they walk around the *finished* structure and judge whether it looks like a real, structurally sound building overall. By feeding the inspector's top-down verdict back to the bricklayer, the bricklayer starts placing bricks that **add up to something coherent**, not just locally plausible.

---

## 14. Can the Generator Learn Alone? (Auto-Encoders)

A natural question: could the Generator just learn by itself, without a Discriminator at all?

**The problem:** the Generator needs matching pairs of (code vector → correct output) to learn from directly — but where would those "correct" random codes come from? One classic answer is the **auto-encoder**.

An auto-encoder is two networks trained **together**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/de9756ef-4b64-4ca1-b223-878e6bfb6d9a" width=600>
</p> 

- An **Encoder** compresses a high-dimensional input (e.g. an image) down into a compact, low-dimensional **code**.
- A **Decoder** takes that code and tries to **reconstruct** the original input as closely as possible.

$$
\text{Encoder}(x) = c \quad \longrightarrow \quad \text{Decoder}(c) \approx x
$$

Once trained, **the Decoder alone is a Generator** — feed it any code vector, and it will produce a plausible-looking output, without needing a Discriminator at all.

> **Analogy:** Think of the auto-encoder like learning **shorthand note-taking and then reading your notes back out loud**. The Encoder is you, in a lecture, **compressing** a long spoken sentence down into a handful of quick shorthand symbols. The Decoder is you, later, **expanding** those symbols back into a full sentence. If you get good enough at both directions, your "expansion" skill (the Decoder) can even be reused on shorthand notes you never actually took — i.e. it becomes a general-purpose generator of sentences from symbols.

---

## 15. Worked Example: Reading the Pixel-Error Diagram

Auto-encoders sound like a clean solution — so why do we still need a Discriminator? The core weakness with a simple worked example is illustrated, using an image of a hand-drawn digit "2."

**Setup:** the Generator produces an image that's supposed to match a target digit "2." The training signal is a pixel-by-pixel loss: "make the generated image **as close as possible**, pixel-by-pixel, to the target."

<p align="center">
  <img src="https://github.com/user-attachments/assets/08d27e52-5eb8-4756-a81f-e13bdfbc4fd1" width=1000>
</p>

| Error type | Description | Pixel-wise loss | Does it *look* like a real "2"? |
|---|---|---|---|
| **1 pixel error (Case A)** | One random stray pixel flipped somewhere in empty space | Small (1 pixel wrong) | Yes — still clearly a "2" |
| **1 pixel error (Case B)** | One pixel flipped, but it's inside a critical stroke gap | Small (1 pixel wrong) | Could break the shape, depending on where it lands |
| **6 pixels error** | Several pixels flipped together, forming a small extra blob/stroke in one corner | Large (6 pixels wrong) | Might still look totally fine as a "2" — the blob is a minor smudge |

**The punchline:** a plain pixel-by-pixel loss treats **all errors of the same count as equally bad**, regardless of *where* they are or whether they actually break the recognisable shape of the digit. A "1 pixel error" that happens to fall in a critical spot can be visually *worse* than a "6 pixel error" that's just a harmless smudge — but a naive per-pixel loss can't tell the difference, because it only counts **how many pixels differ**, not **whether the overall shape/relationships between components still make sense**.

**Why does this happen?** The relationship *between* components (pixels) is what actually matters — a shallow, per-pixel comparison misses whether neighbouring or spatially distant pixels are **correlated** in a way that matters for the whole shape to look right. This is exactly the "planning" problem from Section 12 showing up again.

> **Analogy:** Imagine grading two students' short essays purely by **counting how many words differ** from a model answer. Student A changes exactly 1 word — but it happens to be the word "not," flipping the entire meaning of the sentence into nonsense. Student B changes 6 words, but they're all minor synonym swaps that don't change the meaning at all. A word-count-based grading system would rank Student A as "better" (fewer edits) even though their essay is now **factually wrong**, while Student B's larger word-count differs but is still a genuinely good essay. Counting differences isn't the same as judging overall coherence — which is exactly why we need a Discriminator (a "reader" who judges the *whole* essay, not just a word-diff count) instead of relying on the Generator's own pixel-matching alone.

---

## 16. Can the Discriminator Learn Alone?

The flip side of Section 14: could the **Discriminator** learn by itself, without needing a Generator to supply fake examples?

**The Discriminator is a function $D: x \rightarrow \mathbb{R}$** that outputs a "how good/real is this?" score. Compared to the Generator's bottom-up, pixel-by-pixel approach, the Discriminator can more naturally catch relationships between components through **top-down evaluation** — for instance, a small learnable CNN filter can be trained specifically to check for **isolated, out-of-place pixels**, which is exactly the kind of relationship a pure pixel-by-pixel loss (Section 15) misses.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c403125d-8d6a-4744-b504-dd3d0f53540b" width=1000>
</p>

**The catch:** a Discriminator trained only on **real images** will happily learn to just output "1" (real) for everything, since it's never been shown anything to contrast against.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b9b3f2f4-4612-4d0c-9739-d45830fffc0b" width=1000>
</p>

$$
\boxed{\text{Discriminator training needs negative (fake) samples to be meaningful}}
$$

**A tempting general algorithm:**
1. Start with a set of **positive examples** (real data) and some randomly generated **negative examples**.
2. In each iteration: train $D$ to separate positives from negatives, then **generate new, harder negative examples** by solving

$$
\tilde{x} = \arg\max_{x \in X} D(x)
$$

— i.e., search the entire space of possible objects for whatever currently fools $D$ the most, and use *that* as the next round's negative example.

**Why this doesn't work cleanly in practice:** solving that $\arg\max$ search directly is **far too complicated** over a huge, unconstrained space (e.g., every possible arrangement of pixels). Constraining the search space to make it tractable would also limit how expressive/powerful the model can be. This is precisely **why** it's easier, in practice, to replace "solving $\arg\max D(x)$ directly" with a **separate Generator network** that is *trained* to approximate that search — which is exactly what a GAN's Generator does.

> **Analogy:** Imagine a security guard (Discriminator) trained only by looking at **genuine ID cards**, never a single fake. They'll quickly learn to stamp "APPROVED" on anything, because they've never had to practise catching a forgery. To actually train them well, you need a **stream of realistic fake IDs** to test them against. But manually searching through "every conceivable way to draw a fake ID" to find the single most convincing fake (the $\arg\max$ approach) would take forever — it's far more practical to hire an **actual forger (the Generator)** whose entire job is to keep producing increasingly convincing fakes for the guard to practise on.

---

## 17. Pros and Cons: Generative vs Discriminative

| | **Generative Model** | **Discriminative Model** |
|---|---|---|
| **Pros** | Easy to generate new samples, especially with a deep model | Naturally considers the **big picture** (top-down), so it catches component relationships well |
| **Cons** | Tends to just **imitate appearance** (e.g., pixel-matching) and struggles to learn correlations **between** components (Section 15) | Generation isn't always feasible on its own; **how do you get good negative samples** to train against? (Section 16) |

> **Analogy:** The Generator alone is like an artist who can **copy a photo stroke-by-stroke** very well but doesn't truly understand composition — they might nail every individual brushstroke while still producing a portrait where the eyes are subtly misaligned, because they never step back to check the *whole face*. The Discriminator alone is like an art critic with **excellent judgement** about what looks right or wrong overall, but who has **never picked up a paintbrush** and has no idea how to actually produce a painting from scratch, nor an efficient way to conjure a believable-looking "bad" painting to sharpen their own judgement against.

---

## 18. The Benefit of Combining Both (Why GAN Works)

A GAN elegantly resolves both weaknesses at once by letting each network do **what it's naturally good at**, and using each to patch the other's blind spot:

$$
G \rightarrow \tilde{x} \qquad \Longleftrightarrow \qquad \tilde{x} = \arg\max_{x \in X} D(x)
$$

- **From the Discriminator's point of view:** instead of solving an impossibly complex search by itself, it simply uses the **Generator** to supply a constant stream of realistic negative samples to train against.
- **From the Generator's point of view:** it still generates the object **component-by-component** (bottom-up, as always) — but now it's being guided by a Discriminator that judges with a **global, top-down** view, so the Generator implicitly learns to respect relationships between components too.

> **Analogy:** This is the forger-and-detective story from Section 4, now explained from *both* sides at once. The **detective** no longer needs to imagine every conceivable fake painting themselves — they simply study whatever the **forger** brings them next. The **forger** still paints one brushstroke at a time, but because a sharp-eyed detective keeps rejecting paintings where the brushstrokes don't add up to a believable whole, the forger is forced to start thinking about the **overall composition**, not just individual strokes — even though composition was never explicitly taught to them.

---

## 19. Conditional Generation by GAN

**A limitation of plain GANs:** the Generator produces a somewhat **random** image from the learned domain. The relationship between a specific input vector and the specific image it produces is complex and hard to control — you can't easily say "generate *this specific kind* of image."

Some datasets come with extra information — like a class label or a text description — and it's very useful to make use of it, for two reasons:
1. **Improving the GAN** overall (extra information can act as a helpful training signal).
2. **Targeted / controllable generation** — e.g., typing a caption and getting an image that actually matches it.

**Why not just use plain supervised learning for this (e.g. text $\rightarrow$ image)?** Because a network trained with a simple "get as close as possible" loss to a single target image tends to produce a **blurry average** of many plausible outputs. For example, given the word "Train," many different real train photos are all valid targets — a supervised network trying to satisfy all of them simultaneously ends up outputting something like the **statistical average** of all those trains, which looks like a vague, blurry blob rather than any single, sharp, realistic photo.

<p align="center">
  <img src="https://github.com/user-attachments/assets/259c30a0-42dc-461f-9a56-f876002a1e3d" width=1000>
</p>

> **Analogy:** Imagine asking 20 different artists to each draw "a train," then **overlaying all 20 drawings on top of each other** and averaging the ink density at every point. The result wouldn't look like any single train — it'd be a smudgy blur, because different artists put the chimney, windows, and wheels in slightly different spots, and averaging washes all those specific, sharp details away. That's exactly what happens when a network is trained to minimise *average* pixel distance to many valid answers at once.

### Conditional GAN (cGAN) — Fixing This with a Smarter Discriminator

In a **Conditional GAN**, the Generator takes in **both** a condition $c$ (e.g. the text "Train") **and** a random noise vector $z$:

$$
\tilde{x} = G(c, z)
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/ff658449-ee32-447c-abeb-ee2e60d476f6" width=800>
</p>

If the Discriminator only checks "is $x$ real or not," the Generator can get away with producing **realistic-looking images that completely ignore the condition** $c$ — because the Discriminator was never taught to check whether the image actually *matches* the given text.

**The fix:** feed the Discriminator **both** $c$ and $x$ together, and train it to output a score representing **both** "is $x$ real?" **and** "does $x$ actually match $c$?"

$$
D(c, x) \rightarrow \text{scalar}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/5aed4c8a-a27f-40bd-8ae8-3b10cb6e50e5">
</p>

| Pair type | Example | Label |
|---|---|---|
| **True pair** | (`"train"`, a real photo of a train) | 1 |
| **False pair — wrong label** | (`"cat"`, a real photo of a train) | 0 |
| **False pair — fake image** | (`"train"`, a Generator-produced image) | 0 |

> **Analogy:** A plain Discriminator is like a bouncer who only checks **"is this a real photo or a doodle?"** — they'd happily wave through a beautifully rendered photo of a *cat*, even if you asked for a train. A **Conditional GAN's Discriminator** is a stricter bouncer who checks **two things at the door**: "is this photo real?" **and** "does it actually match the ticket description (the caption) you were given?" Only photos that pass *both* checks are approved — which is exactly the pressure that forces the Generator to actually pay attention to the condition instead of ignoring it.

---

## 20. Worked Example: Reading Matched vs Unmatched Pairs

The conditional GAN's step-by-step training procedure explicitly builds **three types of samples** into every training batch for the Discriminator:

1. **Real pairs** — $(c^{(i)}, x^{(i)})$ straight from the database → should be scored **high**.
2. **Generator's fakes** — $(c^{(i)}, \tilde{x}^{(i)})$ where $\tilde{x}^{(i)} = G(c^{(i)}, z^{(i)})$ → should be scored **low** (the image itself is fake).
3. **Unmatched real pairs** — $(c^{(i)}, \hat{x}^{(i)})$, where $\hat{x}^{(i)}$ is a **real** image from the database, but deliberately paired with the **wrong** condition → should *also* be scored **low** (the image is real, but it doesn't match the caption).

$$
\tilde{V} = \frac{1}{m}\sum_{i=1}^m \log D(c^{(i)}, x^{(i)}) + \frac{1}{m}\sum_{i=1}^m \log\left(1 - D\big(c^{(i)}, \tilde{x}^{(i)}\big)\right) + \frac{1}{m}\sum_{i=1}^m \log\left(1 - D\big(c^{(i)}, \hat{x}^{(i)}\big)\right)
$$

**How to read this kind of setup in general:** whenever you see a Discriminator being fed a mix of "real-but-mismatched" pairs alongside the usual real and fake examples, its purpose is to teach the Discriminator that **"real" and "matches the condition" are two separate things it must check independently** — a photo isn't automatically approved just because it's a genuine, un-doctored photo; it also has to be the *right* photo for the given condition.

> **Analogy:** This is like training airport check-in staff with **three kinds of test cases**, not just two: (1) a genuine passport that matches the boarding pass — **approve**; (2) an obviously forged passport — **reject**; and (3) a *completely genuine, valid* passport that simply belongs to **someone else**, mismatched with this boarding pass — also **reject**. Without training on case (3) specifically, staff might only ever learn to check "is this passport real?" and never learn to check "does it actually match this ticket?"

---

## 21. Image-to-Image Translation with GAN

**Image-to-Image translation** takes an input image and outputs a corresponding image in a different domain — e.g., turning a segmentation-label map into a photorealistic street scene, a black-and-white photo into colour, a day photo into a night photo, an aerial photo into a map, or a hand-drawn sketch into a photo of a bag or shoe.

<p align="center">
  <img src="https://github.com/user-attachments/assets/78ffe676-5111-4fcc-b231-390d3ee28938" width=600>
</p>

Just like text-to-image, a **plain supervised approach** (train a network to minimise pixel distance to one target photo) produces **blurry** results, for the same averaging reason as before.

**Using a GAN instead:** the Generator takes the input image (plus optionally some noise $z$) and produces an output image; the Discriminator judges whether that output image is a **real, convincing example of the target domain** (and, in the conditional-GAN style, whether it's a good *match* for the input). Adding a small extra "as close as possible" term on top of the adversarial loss (**GAN + close**) can further nudge the output to stay faithful to the specific input, rather than just "any realistic-looking output."

<p align="center">
  <img src="https://github.com/user-attachments/assets/31ff24cf-6d21-467f-a11c-c246087eabf5" width=600>
</p>

| Approach | Typical Result |
|---|---|
| **Traditional supervised (pixel loss only)** | Blurry — an average of many plausible outputs |
| **GAN only** | Sharp and realistic, but might drift from matching the specific input closely |
| **GAN + closeness term** | Sharp **and** faithful to the input |

> **Analogy:** Think of colourising an old black-and-white photo. A purely pixel-matching network is like a colourist who's scared of committing to any one specific colour choice, so they paint every wall a **washed-out, indecisive beige** that's "safe" on average. A GAN-based colourist commits to **bold, specific, realistic colours** because a strict art critic (the Discriminator) would reject a wishy-washy, undecided colouring as clearly fake. Adding a "closeness" term on top is like also telling that colourist, "and make sure it's still recognisably *this exact* photo, not just *some* colourful, realistic photo."

---

## 22. Other Conditional GAN Applications

Conditional GANs generalise well beyond images:

- **Speech Enhancement:** the Generator takes a **noisy** speech spectrogram and produces a **cleaned-up** version; the Discriminator is shown pairs of (output, noisy input) or (clean, noisy input) and judges whether the *pairing* looks like a genuine "cleaned this specific noisy clip" relationship or a fake one.

<p align="center">
  <img src="https://github.com/user-attachments/assets/31ff24cf-6d21-467f-a11c-c246087eabf5" width=1000>
</p>

- **Video Generation:** the Generator takes in the last several frames $t_{1:n}$ of a video and predicts the **next frame** $t_{n+1}$; the Discriminator is shown the full sequence $t_{1:n+1}$ (with either the *real* next frame or the *generated* one appended) and judges whether that **last frame** looks like a real, natural continuation or a generated/fake one.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a0b5f8d1-92a9-4d5e-a0e5-090bc58ea6d1" width=1000>
</p>

> **Analogy:** Speech enhancement with a conditional GAN is like a noise-cancelling audio engineer whose work is graded not just on "does the output sound clean?" but on **"does this clean output still sound like the same voice/sentence as the noisy recording it came from?"** Video-frame generation is like being shown a short video clip with the last second **cut off**, and being asked to guess what happens next in a way that feels like a natural, un-jarring continuation — a good guess flows smoothly; a bad one feels like an abrupt jump-cut.

---

## 23. Unsupervised Conditional Generation (No Paired Data)

So far, conditional GANs assumed we have **paired data** — e.g., a black-and-white photo *and* its correct colour version, side by side. But often we only have **two separate, unpaired collections**: photos in Domain $X$ (e.g. real photographs) and images in Domain $Y$ (e.g. Monet-style paintings), with **no example that says "this exact photo corresponds to this exact painting."** This is the setting for **style transfer**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3a6f9368-3f84-4f08-b8cd-ee485ad90f73" width=600>
</p>

**The naive "direct transformation" approach:** train a Generator $G_{X \to Y}$ to take an image from $X$ and output something that a Discriminator $D_Y$ believes belongs to domain $Y$.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8d86fdc4-c78a-4364-9e92-27420dd52b73" width=600>
</p>

**The problem this runs into:** the Discriminator $D_Y$ only ever checks **"does this output look like it belongs to domain Y?"** — it never checks whether the output still has anything to do with the **specific input image** that was fed in. This opens the door for the Generator to essentially **ignore the input entirely** and just learn to output *some* generic, convincing Domain-Y-looking image every time, regardless of what photo it was given.

<p align="center">
  <img src="https://github.com/user-attachments/assets/161b295c-8921-4ae5-baa1-c516790acf47" width=600>
</p>

$$
\boxed{\text{Input might get ignored} \;\Rightarrow\; \text{output looks like domain } Y \text{, but bears no relation to the specific input}}
$$

**Two partial fixes:**
1. **Simpler generator network design** — a Generator with less capacity/flexibility is naturally forced to keep the output more closely tied to the input, since it doesn't have the "room" to fabricate something unrelated.
2. **A pre-trained encoder as a feature-preservation check** — feed *both* the original input and the Generator's output through the same pre-trained encoder network, and add a loss term that keeps their encoded features **as close as possible**. This explicitly forces the output to preserve the input's core content, even while changing its *style*.

<p align="center">
  <img src="https://github.com/user-attachments/assets/815a70ce-1a61-458e-8409-8167bdbe65c8" width=1500>
</p>

> **Analogy:** Imagine hiring a translator to translate your diary from English into French, but you can **only check the French output on its own** — you never compare it back to your original diary. A lazy (or dishonest) translator could hand you back **any well-written French text about their day**, and you'd have no way to catch that it has nothing to do with what you actually wrote — as long as it *reads like fluent French*, it passes the check. That's the "input might get ignored" trap: judging only the output's *style* (fluent French / Domain Y) without ever verifying it's still faithful to the *content* of the specific input.

---

## 24. CycleGAN: Solving the "Ignored Input" Problem

**CycleGAN** solves the "ignored input" problem elegantly, using an idea called **cycle consistency** — and crucially, it does this **without needing any paired data**, and without needing a separate pre-trained encoder.

**The idea:** train **two** Generators, one for each direction:
- $G_{X \to Y}$: translates Domain $X \to$ Domain $Y$.
- $G_{Y \to X}$: translates Domain $Y \to$ Domain $X$.

Take an image from Domain $X$, translate it to Domain $Y$ using $G_{X \to Y}$, then **translate it back** to Domain $X$ using $G_{Y \to X}$. The result should be **as close as possible** to the *original* starting image.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4fab09f9-36b9-4d2e-a28a-2006573691d7" width=600>
</p>

The same check is also done in the opposite direction, starting from a real Domain $Y$ image, translating to $X$ and back to $Y$. On top of this, ordinary Discriminators $D_X$ and $D_Y$ still check that each translated image looks convincingly like it belongs to its target domain.

<p align="center">
  <img src="https://github.com/user-attachments/assets/53769025-0bcb-49bc-b109-aec0a9af9985" width=600>
</p>

**Why this fixes the "ignoring the input" problem:** if $G_{X \to Y}$ threw away all the specific information about the input photo and just output *some generic* Domain-Y-looking image every time, then $G_{Y \to X}$ would have **no way to reliably reconstruct the original input** from that generic output — because all the input-specific details needed for a faithful round trip would already be lost. The cycle-consistency loss directly **punishes** that information loss, forcing both Generators to preserve the input's core content while still changing its style.

> **Analogy:** This is exactly like the classic **translation round-trip check**: translate an English sentence into French, then translate that French sentence **back** into English, and see if you land back on (roughly) the same sentence you started with. If the first translator was lazy and produced generic, unrelated French text instead of an actual translation, the round-trip would come back as complete nonsense compared to the original — instantly exposing the problem. Requiring a **faithful round trip** forces both translators to actually preserve the *meaning* (content) of the original sentence, even while changing its *language* (style) — which is precisely what CycleGAN forces its two Generators to do with images.

---

## 25. Summary: All Models Side by Side

| Model / Concept | Setting | Needs paired data? | Key idea |
|---|---|---|---|
| **Vanilla GAN** | Unconditional generation | No labels needed | Generator (bottom-up) vs. Discriminator (top-down), trained adversarially |
| **Auto-encoder as Generator** | Generator-only, no Discriminator | N/A | Encoder compresses, Decoder reconstructs; Decoder alone becomes a Generator — but suffers from the pixel-error problem (Section 15) |
| **Conditional GAN** | Text/label $\to$ image, or general supervised-style generation | **Yes** (matched condition–output pairs) | Discriminator checks both "is it real?" **and** "does it match the condition?" |
| **Image-to-Image GAN** | Photo domain $\to$ photo domain (paired) | **Yes** | Same as Conditional GAN, applied to image pairs (e.g. labels → facade) |
| **Direct unsupervised transformation** | Style transfer, unpaired domains | No | Simple $G_{X\to Y}$ + $D_Y$ — but risks the Generator ignoring the input entirely |
| **CycleGAN** | Style transfer, unpaired domains | No | Adds a **round-trip / cycle-consistency** check with two Generators, preventing input from being ignored |

> **Analogy for the whole family tree:** A **vanilla GAN** is a forger who paints *anything* convincing, with no constraints. A **Conditional GAN** is a forger given a **specific commission** ("paint me a train") and graded on following the brief, not just painting *something* convincing. An **Image-to-Image GAN** is the same forger, but the "commission" is itself a full reference photo they must adapt into a new style. A **direct unsupervised transformation** is a forger asked to paint "in the style of Monet" with **no reference photo required to match against** — risky, because they might just paint *any* Monet-style scene and call it done. **CycleGAN** fixes that by also requiring the forger to **describe the original scene back to you afterward** well enough that you'd recognise it — forcing them to actually pay attention to what they started with.

---

## 26. Key Takeaways

> [!IMPORTANT]
> **The core things to remember from Week 10:**

1. A **Generative Model** learns the data distribution $p(x)$ (or joint $p(x,y)$) well enough to **generate new, realistic samples** — unlike a **Discriminative Model**, which only learns $p(y \mid x)$, a boundary between classes.
2. A **GAN** pits a **Generator** (creates fake data from random noise) against a **Discriminator** (scores real vs. fake) in an adversarial, alternating training loop.
3. Training alternates: **Step 1** fixes $G$ and updates $D$ to better separate real from fake; **Step 2** fixes $D$ and updates $G$ (via gradient ascent) to better fool $D$.
4. GAN-style generation is a form of **structured learning** — output is a *composed* object (image, sentence) whose parts have **dependencies**, which is why it's fundamentally harder than plain classification or regression.
5. The **Generator alone** (via an auto-encoder) tends to just **imitate appearance** using a pixel-by-pixel loss, which can't distinguish a harmless smudge from a shape-breaking error (Section 15).
6. The **Discriminator alone** struggles because it needs realistic **negative samples** to train against, and directly searching for the best fake ($\arg\max D(x)$) is computationally intractable.
7. **Combining both** works because the Generator supplies the Discriminator with a constant stream of useful negatives, while the Discriminator's **top-down, whole-object judgement** teaches the Generator to respect relationships between components.
8. Plain supervised text-to-image / image-to-image models tend to produce **blurry outputs**, because they average over many valid targets; GANs avoid this by rewarding *sharp, realistic, committed* outputs instead.
9. A **Conditional GAN** needs its Discriminator to check **both** realism **and** whether the output actually matches the given condition $c$ — otherwise the Generator can learn to simply ignore $c$.
10. In **unsupervised** (unpaired) style transfer, a naive Generator can also **ignore the input entirely**, since the Discriminator only checks "does this look like domain Y?" and never checks faithfulness to the specific input.
11. **CycleGAN** fixes this using **cycle consistency**: translating $X \to Y \to X$ (or $Y \to X \to Y$) should return (approximately) the original image, which forces the Generators to preserve the input's core content.

---

## 27. Glossary

| Term | Definition |
|------|-----------|
| **Generative Model** | A model that learns the underlying data distribution $p(x)$ so it can generate new, realistic samples |
| **Discriminative Model** | A model that learns the conditional probability $p(y\mid x)$, i.e. a decision boundary between classes |
| **Generator (G)** | A network that maps a random vector (and optionally a condition) to a generated object |
| **Discriminator (D)** | A network that scores an object (and optionally a condition) as real or fake |
| **Latent Vector / Code (z)** | The random input vector fed to the Generator; each dimension can end up controlling some characteristic of the output |
| **Adversarial Training** | Training two networks in competition, where one improves at the other's expense |
| **Gradient Ascent** | Updating parameters to **increase** a target score (used for Generator training), as opposed to gradient descent, which decreases a loss |
| **Structured Learning / Prediction** | Predicting outputs with internal structure and dependencies (sequences, matrices, graphs) rather than a single scalar or class label |
| **Zero-shot / One-shot problem** | The challenge that, in structured learning, the output space is so huge that most possible outputs have no training examples |
| **Planning problem** | The challenge of generating an output component-by-component while still keeping the "big picture" consistent |
| **Auto-encoder** | A pair of networks (Encoder + Decoder) trained together to compress and reconstruct data; the Decoder alone can act as a Generator |
| **Encoder** | Compresses a high-dimensional input into a compact code |
| **Decoder** | Reconstructs the original input from a compact code |
| **$\arg\max_x D(x)$** | The (intractable) idea of directly searching for the input that maximises the Discriminator's score, to use as a hard negative example |
| **Conditional GAN (cGAN)** | A GAN where the Generator and Discriminator both take an extra condition $c$ (e.g. text, label, or another image) as input |
| **True pair / False pair** | In cGAN training: a matching (condition, real image) pair vs. a mismatched or fake pair, both used to train $D$ |
| **Image-to-Image Translation** | Mapping an image in one domain (e.g. sketches) to a corresponding image in another domain (e.g. photos) |
| **Style Transfer** | Transforming an object's style (e.g. photo → painting) while (ideally) preserving its content |
| **Unpaired / Unsupervised Conditional Generation** | Learning to translate between two domains without example pairs that explicitly correspond to each other |
| **Direct Transformation** | The naive unpaired approach: $G_{X\to Y}$ + $D_Y$ only, which risks the Generator ignoring the input |
| **CycleGAN** | An unpaired image-to-image translation method using two Generators and a cycle-consistency loss |
| **Cycle Consistency** | The requirement that translating $X \to Y \to X$ should reconstruct (approximately) the original input |

---

> [!TIP]
> **Study tip for Week 10:** Make sure you can:
> 1. Explain, in your own words, the difference between a **generative** and a **discriminative** model, using the lion-and-elephant (or fake-banknote) analogy.
> 2. Walk through the **GAN training loop** step by step — explain why Step 1 and Step 2 alternate, and why gradient *ascent* (not descent) is used when updating the Generator.
> 3. Explain, using the **pixel-error worked example**, why a plain pixel-by-pixel loss can rate a shape-breaking error as "better" than a harmless smudge — and why this shows the Generator can't easily learn structure alone.
> 4. Explain why a Discriminator can't easily train itself without a Generator, and what the $\arg\max_x D(x)$ idea represents (and why it's impractical).
> 5. Explain how a **Conditional GAN's Discriminator** differs from a plain GAN's Discriminator (hint: it checks two things, not one), and why that prevents the "ignoring the condition" failure mode.
> 6. Explain why plain supervised text-to-image / image-to-image models tend to produce **blurry** results, and why GANs don't have this problem.
> 7. Explain the **"input might get ignored"** problem in unsupervised (unpaired) style transfer, and how **CycleGAN's cycle-consistency loss** fixes it.
> 8. Be able to read a **"results across training epochs" grid** (like the manga generation example) and explain what generally improves first (global structure) versus what improves later (fine detail).
