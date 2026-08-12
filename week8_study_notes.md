# 📗 Week 8 Study Notes: Face Recognition Using Deep Learning

> **Course:** COS30082 — Applied Machine Learning
> **Topic:** Face Recognition Using DL
> **Prerequisite:** Week 5–7 (Convolutional Neural Networks, Transfer Learning, Object Detection)

---

## Table of Contents

1. [What Is Face Recognition?](#1-what-is-face-recognition)
2. [Face Identification vs Face Verification](#2-face-identification-vs-face-verification)
3. [Why Face Detection Matters](#3-why-face-detection-matters)
4. [Classical (Pre-Deep-Learning) Face Recognition](#4-classical-pre-deep-learning-face-recognition)
5. [Why Classical Methods Struggle at Scale](#5-why-classical-methods-struggle-at-scale)
6. [Worked Example: Reading the Milestones-of-Face-Representation Chart](#6-worked-example-reading-the-milestones-of-face-representation-chart)
7. [One-Shot Learning](#7-one-shot-learning)
8. [Face Recognition as a One-Shot Learning Problem](#8-face-recognition-as-a-one-shot-learning-problem)
9. [Why a Standard CNN Classifier Fails Here](#9-why-a-standard-cnn-classifier-fails-here)
10. [Reframing the Problem: Learning a Similarity Function](#10-reframing-the-problem-learning-a-similarity-function)
11. [Worked Example: Reading a Similarity-Score Diagram](#11-worked-example-reading-a-similarity-score-diagram)
12. [The Siamese Network](#12-the-siamese-network)
13. [The Learning Objective: What Makes a Good Encoding?](#13-the-learning-objective-what-makes-a-good-encoding)
14. [Triplet Loss](#14-triplet-loss)
15. [The Learning Objective (With a Margin)](#15-the-learning-objective-with-a-margin)
16. [The Triplet Loss Function](#16-the-triplet-loss-function)
17. [Choosing Good Triplets](#17-choosing-good-triplets)
18. [Triplet Mining: Easy, Hard, and Semi-Hard](#18-triplet-mining-easy-hard-and-semi-hard)
19. [Worked Example: Calculating the Triplet Loss by Hand](#19-worked-example-calculating-the-triplet-loss-by-hand)
20. [Offline vs Online Triplet Mining](#20-offline-vs-online-triplet-mining)
21. [Deploying the Network as an Embedding Generator](#21-deploying-the-network-as-an-embedding-generator)
22. [Similarity Metrics: How Do We Measure "Distance"?](#22-similarity-metrics-how-do-we-measure-distance)
23. [Face Verification as Binary Classification](#23-face-verification-as-binary-classification)
24. [Worked Example: Interpreting a Verification Dataset Table](#24-worked-example-interpreting-a-verification-dataset-table)
25. [Key Takeaways](#25-key-takeaways)
26. [Glossary](#26-glossary)
27. [Study Tips for Week 8](#27-study-tips-for-week-8)

---

## 1. What Is Face Recognition?

**Face Recognition** is really an umbrella term for **two separate tasks**:

| Task | Question it answers |
|---|---|
| **Face Identification** | "Given a face, and a database of known people, **whose** face is it?" |
| **Face Verification** | "Given a face and a **claimed** identity, **is** it really that person?" |

$$
\boxed{\text{Face Recognition} = \text{Identification (who is this?)} + \text{Verification (is this who they claim to be?)}}
$$

> **Analogy:** Imagine a members-only club. **Identification** is like a staff member looking at someone walking in and, without them saying a word, figuring out from a big photo album "that's member #482, Sarah." **Verification** is like someone swiping a membership card that says "I am Sarah" at the door, and the staff member just checks "does this face actually match Sarah's photo on file?" — a much narrower, one-person check.

---

## 2. Face Identification vs Face Verification

### 2.1 Face Verification — a "One-to-One" Problem

- **Input:** an image of a person **+** a claimed name/ID
- **Output:** yes/no — is the input image really that claimed person?
- Only **one** comparison is made: the input face vs the **one** face on file for that claimed identity.

### 2.2 Face Identification — a "One-to-Many" Problem

- The system holds a **database of K people**.
- **Input:** a new face image (no claimed identity given).
- **Output:** the matching person's ID — **or** "not recognised" if nobody in the database matches.
- The input face must be compared against **all K** people in the database.

| | Verification | Identification |
|---|---|---|
| **Comparisons per query** | 1 (input vs claimed identity) | K (input vs everyone in the database) |
| **Problem type** | One-to-one | One-to-many |
| **Typical use case** | Unlocking your phone, boarding-gate check | Airport watch-list screening, "tag this friend" in photos |
| **Possible outputs** | "Match" / "No match" | A specific person's ID, or "hasn't seen them before" |

> **Analogy:** Verification is like a bouncer checking **one** ID card against **one** face — "does this photo match this person, yes or no?" Identification is like a detective walking into a room full of suspects with **only a photo** and having to figure out, by comparing it against **every single person** in the room, which one (if any) is the match. The detective's job is fundamentally harder because the search space is much bigger — the more people (K) in the room, the more comparisons are needed, and the more chances there are for mistaken identity.

---

## 3. Why Face Detection Matters

Before we can *recognise* a face, we first need to **find** it in the image. **Face Detection** is the "forefront module" — the very first step — of any automatic face recognition pipeline.

It's also the entry point for many other systems:

- **Human–Computer Interaction:** expression recognition, cognitive/emotional state recognition, biometric security
- **Surveillance:** tracking, generic object/target recognition, human behaviour analysis

> **Analogy:** Face detection is like the **front door** of a building. No matter what happens deeper inside (recognition, expression analysis, tracking), everyone and everything has to pass through that front door first. If the front door (detection) fails to notice someone walking in, nothing downstream — recognition, security checks, tracking — ever gets a chance to run at all.

---

## 4. Classical (Pre-Deep-Learning) Face Recognition

Before deep learning took over, face recognition relied on **hand-designed** feature-extraction techniques, split into two broad categories.

### 4.1 Holistic Features — Eigenfaces (PCA)

Eigenfaces treats a whole face image as one long vector and uses **Principal Component Analysis (PCA)** to find the small set of "basis faces" that best explain variation across a training set of faces. Any new face can then be approximately reconstructed as a **weighted combination** of these "eigenfaces."

| Advantage | Disadvantage |
|---|---|
| Easy to implement; needs no special facial-feature knowledge beyond the face ID | Sensitive to lighting, shadows, and the scale of the face in the image |

> **Analogy:** Imagine you could describe **anyone's face** using just a handful of sliders — "more/less round," "more/less angular jaw," "wider/narrower eyes" — instead of describing every pixel. Eigenfaces finds the small number of sliders that, combined, can approximately recreate most faces in your training set. It's efficient, but if the lighting changes dramatically, the "slider readings" it estimates can be thrown off — a shadow across half the face can look, mathematically, like a completely different combination of sliders.

### 4.2 Local-Based Features — LBP, SIFT

Instead of treating the whole face as one blob, **Local Binary Pattern (LBP)** and **SIFT** look at small local neighbourhoods of pixels, encode the local texture pattern around each pixel, divide the face into a grid of regions, build a **histogram** for each region, and then **concatenate** all the histograms into one final feature vector.

| Advantage | Disadvantage |
|---|---|
| Robust features for frontal faces; invariant to translations and rotations | Sensitive to noise |

> **Analogy:** If Eigenfaces is like describing a face with a few global sliders, LBP is like a detective who examines a face **patch by patch** — "this patch of skin near the eye has this specific micro-texture pattern, this patch near the mouth has that pattern" — and builds up a detailed local fingerprint. Because it's built from many small, independent patches, LBP doesn't get thrown off as easily if the face shifts slightly or rotates a little — but a patch of random noise (e.g. sensor grain in a dark photo) can still confuse an individual local reading.

---

## 5. Why Classical Methods Struggle at Scale

- Classical approaches **don't scale well** to very large face datasets, and they **limit** how rich and robust the learned face representations can become — they rely on features a human decided were important, not features the data itself reveals.
- The recent availability of **affordable, powerful GPUs** and the creation of **huge public face databases** shifted the research focus toward **deep neural networks** that handle *every* stage of the pipeline — detection, preprocessing, feature representation, **and** classification — for both verification and identification.

> **Analogy:** Classical, hand-designed features are like navigating a city using a **paper map someone drew by hand decades ago** — useful for the roads that existed when it was drawn, but it doesn't automatically update itself as new roads are built. Deep learning is like switching to a **live GPS system** that continuously learns and updates its own map directly from real traffic data (huge datasets), rather than relying on someone's fixed, hand-drawn version.

---

## 6. Worked Example: Reading the Milestones-of-Face-Representation Chart

The slides show a timeline chart plotting **face-representation quality** (as accuracy on the LFW benchmark) against **time**, tracing four "eras" of feature representation stacked on top of each other:

```text
Representation
  ▲                                                          Deepface
  │                                                          (LFW > 97%)
  │                                              ● Deepfaces, VGG, Facenet
Deep                                            /
learning                                       /  LE
  │                                            /   (LFW > 82%)
Shallow                                    ●──/
learning                        Gabor, LBP /
  │                            (LFW > 70%)/
Local                          ●─────────/
handcraft            Eigenface/
  │                  (LFW~60%)
Holistic       ●────/
learning       │
  └────────────┴──────────┴──────────┴──────────┴────────▶ Time
             1991        1997       2010       2014
```

**How to read a chart like this:**

| Era | Approx. year | Approx. accuracy (LFW) | What it represents |
|---|---|---|---|
| **Holistic learning** | 1991 | ~60% | Whole-face statistical methods (e.g. Eigenface) |
| **Local handcraft** | 1997 | > 70% | Hand-designed local descriptors (Gabor, LBP) |
| **Shallow learning** | 2010 | > 82% | Learned (but still relatively simple/"shallow") local encodings |
| **Deep learning** | 2014 | > 97% | Deep CNN-based representations (DeepFace, VGG, FaceNet) |

**The key trend to notice:** each new "era" isn't just a **different** method — it's a method that consistently **pushes accuracy higher**, and the *rate* of improvement accelerates sharply once deep learning arrives around 2012–2014 — the curve gets noticeably steeper in a short span of years, compared to the slower climb from 1991 to 2010.

> **Analogy:** Reading this chart is like looking at a graph of how fast people could travel across a country over history — walking (holistic), then horse and cart (local handcraft), then early cars (shallow learning), then commercial jets (deep learning). What matters isn't just that each new method is "a bit better" — it's that the **jump in capability accelerates** with each generation, and the biggest leap by far happens in the most recent, shortest time window. When you see a chart like this in general, always check **both** axes together: not just "did it improve," but "how much time did that improvement take," since a steep, short jump tells a very different story than a slow, gradual climb.

---

## 7. One-Shot Learning

**One-shot learning** is a classification task where a model must learn to make correct predictions about **many future examples**, having seen **only one example** (or very few) from each class during training/reference.

$$
\boxed{\text{One-shot learning} = \text{Learn from } \approx 1 \text{ example per class} \rightarrow \text{Generalise to many future, unseen examples}}
$$

> **Analogy:** Imagine meeting someone **once** at a party, for thirty seconds, and then being expected to recognise them confidently a week later at a train station — in different lighting, wearing different clothes, maybe with a new haircut. Most machine learning approaches are the opposite: they want to study a person's face from **hundreds** of angles and lighting conditions before they feel confident. One-shot learning is specifically about training a system to do the "party" version — get it right from a **single** glimpse.

---

## 8. Face Recognition as a One-Shot Learning Problem

Face recognition is a **textbook real-world example** of one-shot learning:

- **Face identification:** a system might have only **one or a few** reference photos of a person's face on file, yet must correctly recognise them in new photos — despite changes in expression, hairstyle, lighting, accessories, and more.
- **Face verification:** a system might have only **one** reference photo of a person, yet must correctly verify new photos of them, potentially **every single day** (e.g. unlocking a phone).

> **Analogy:** Think of a company ID badge system set up on someone's very first day. HR takes **exactly one** photo for the badge. From that day forward, the security camera at the door has to recognise that same employee **every morning**, even though they might be wearing glasses one day, a hat the next, or have a new haircut after a holiday. The system never gets a second "official" reference photo to compare against — it has to work from that single original image, indefinitely. That constraint — "just one reference photo, forever" — is exactly what makes face recognition a natural example of one-shot learning.

---

## 9. Why a Standard CNN Classifier Fails Here

It's tempting to just train a normal CNN classifier — feed in face images, and let a softmax layer predict "whose face is this?" out of K known people. **This does not work well.** Here's why:

```text
Known data
Face ID: 1 ┐
Face ID: 2 │
Face ID: 3 ├──▶  Classic CNN model  ──▶  Softmax loss
Face ID: 4 ┘
```

| Problem | Explanation |
|---|---|
| **Training data is too small** | With only one (or a few) images per person, there isn't nearly enough data to train a robust CNN classifier from scratch |
| **The output classes aren't fixed** | Every time a new person is registered, the number of output classes (K) **changes** — the final classification layer has to be **redesigned and retrained** from scratch |

```text
N new people  +  K known people  ──▶  Classic CNN model  ──▶  Softmax loss over (K + N + 1) classes
```

> **Analogy:** This is like trying to build a school's "who's who" yearbook quiz app by training it as a strict multiple-choice test, where the number of possible answer choices is literally the total number of students. The moment **one new student enrols**, you'd have to throw out the entire quiz and rebuild it with one more answer choice added to every single question — completely impractical for a school (or company, or app) where new people join all the time. We need an approach that doesn't require rebuilding the whole system every time someone new shows up.

---

## 10. Reframing the Problem: Learning a Similarity Function

Instead of treating face recognition as "classify this face into one of K fixed classes," one-shot learning reframes it as a **difference-evaluation problem**: learn a function that measures **how different** two face images are.

$$
d(img1, img2) = \text{degree of difference between two images}
$$

$$
\text{If } d(img1, img2) \leq \tau \rightarrow \text{same person} \qquad \text{If } d(img1, img2) > \tau \rightarrow \text{different people}
$$

where $\tau$ (tau) is a chosen **threshold**.

> **Analogy:** Instead of asking "which **one** of these 1,000 named boxes does this face belong in?" (classification), we instead ask a much simpler, reusable question: "**how far apart** are these two faces?" (similarity). It's the difference between a librarian who must file a book into **one specific, pre-labelled shelf** out of thousands (classification — breaks the moment a new shelf is needed) versus a librarian who simply asks "are these two books similar enough to sit next to each other?" (similarity) — a question that works identically well **no matter how many books the library eventually holds**.

---

## 11. Worked Example: Reading a Similarity-Score Diagram

The slides show one **reference face** compared against **four other faces**, each connected by an arrow labelled with a distance score $d(img1, img2)$:

```text
                         d(img1, img2)
                    16 ──────────────▶  Face A
Reference face  ──── 9 ──────────────▶  Face B
   (query)     ──── 0.1 ─────────────▶  Face C   ← smallest distance
                    25 ──────────────▶  Face D
```

**How to read this diagram:**

| Comparison | Distance score | Interpretation |
|---|---|---|
| Reference vs Face A | 16 | Large distance → **different** person |
| Reference vs Face B | 9 | Large distance → **different** person |
| Reference vs Face C | **0.1** | Very small distance → **same** person |
| Reference vs Face D | 25 | Largest distance → clearly **different** person |

**The rule to apply:** whichever comparison produces the **smallest** distance score is the model's best guess for "who this person actually is" — here, Face C, with a distance of just 0.1, is dramatically smaller than the other three scores, so the model confidently identifies the reference face as belonging to the same person as Face C.

> **Analogy:** This is like a metal detector at airport security that beeps **louder** the closer you get to a hidden metal object. If you wave the wand near four different pockets and get readings of 16, 9, 0.1, and 25 (in some made-up "beep intensity" units), you don't need a strict fixed threshold to know where to look first — you simply walk toward whichever pocket produced the **quietest, closest-to-zero reading**, because in this kind of "distance" system, **the smallest number always signals the closest match**, regardless of the exact numeric scale being used.

---

## 12. The Siamese Network

The key architecture behind one-shot learning is called the **Siamese neural network** (from Gregory Koch et al.'s 2015 paper, *"Siamese Neural Networks for One-Shot Image Recognition"*). It became especially popular once **deep convolutional neural networks** were used to process image inputs **in parallel**.

A Siamese network is **not fundamentally different** from any other CNN — it still takes an image as input and encodes its features into a set of numbers. **The difference is in how the output is used.**

```text
x⁽¹⁾ ──▶ [ CNN layers ] ──▶ [ FC layers ] ──▶ f(x⁽¹⁾)   ─┐
                                                          ├──▶ Compare the two encodings
x⁽²⁾ ──▶ [ CNN layers ] ──▶ [ FC layers ] ──▶ f(x⁽²⁾)   ─┘   (both branches share the SAME weights)
```

Two images are each pushed through **identical, weight-sharing** CNN branches, producing two **encoding vectors** — e.g. 128 numbers each — that represent the two faces in a shared feature space.

> **Analogy:** "Siamese" refers to **identical twins** — and that's exactly the idea here. Instead of building two different judges to evaluate two different photos, you build **one judge** (one set of weights) and simply show them **two photos, one after another**, using the exact same criteria both times. Because it's the *same* judge with the *same* opinions both times, their two verdicts (the two encoding vectors) can be **fairly compared** against each other — if you'd used two different judges with different opinions, comparing their scores wouldn't mean anything.

---

## 13. The Learning Objective: What Makes a Good Encoding?

During training, a classic CNN tunes its parameters to associate each image with the **correct class label**. A Siamese network instead tunes its parameters to produce encodings such that the **distance** between two encodings reflects whether the two faces belong to the same person:

$$
\big\lVert f(x^{(1)}) - f(x^{(2)}) \big\rVert_2^2
$$

**The goal**, for parameters that define the encoding $f(x)$:

$$
\text{If } x^{(i)}, x^{(j)} \text{ are the SAME person:} \quad \big\lVert f(x^{(i)}) - f(x^{(j)}) \big\rVert_2^2 \text{ is SMALL}
$$

$$
\text{If } x^{(i)}, x^{(j)} \text{ are DIFFERENT people:} \quad \big\lVert f(x^{(i)}) - f(x^{(j)}) \big\rVert_2^2 \text{ is LARGE}
$$

> **How can we design an objective function to actually enforce this?** — this is exactly the question **triplet loss** answers next.

> **Analogy:** Picture a huge open field where every person's face gets "planted" as a single point, based on the numbers the network outputs for them (the encoding). A good encoding function is like a groundskeeper whose job is to make sure that **photos of the same person** always get planted **close together** in one cluster, while **photos of different people** get planted **far apart** in separate patches of the field. The network's parameters are literally the groundskeeper's rulebook for deciding where each new photo gets planted.

---

## 14. Triplet Loss

To train the network toward that goal, we use a loss function called **triplet loss**. It trains the network using **three images at once**:

| Role | Meaning |
|---|---|
| **Anchor (A)** | A reference photo of a specific person |
| **Positive (P)** | A **different** photo of the **same** person as the anchor |
| **Negative (N)** | A photo of a **different** person |

```text
Before training:                          After training:

Anchor ●───▶ ● Negative (close)            Anchor ●─────────────▶ ● Negative (far)
       │                                          │
       └───▶ ● Positive (also close)              └──▶ ● Positive (very close)
```

After training, encodings of the **positive** example should sit **close** to the anchor, while encodings of the **negative** example should sit **farther away**.

> **Analogy:** Think of triplet loss like a "one-on-one-on-one" comparison exercise you might use to teach a child what "similar" means — you show them a photo of their mum (anchor), a **different** photo of their mum (positive), and a photo of a stranger (negative), and ask them to physically place the three photos on a table so the two "mum" photos are close together and the stranger's photo is pushed away. Do this exercise over and over with thousands of different triplets, and the child (the network) gradually develops a robust internal sense of "what makes two photos the same person" — without ever needing to memorise a fixed list of names.

---

## 15. The Learning Objective (With a Margin)

Using the anchor/positive/negative pair, the goal is:

$$
d(A,P) = \lVert f(A) - f(P) \rVert^2 \;\; \leq \;\; d(A,N) = \lVert f(A) - f(N) \rVert^2
$$

$$
\Rightarrow \quad \lVert f(A)-f(P) \rVert^2 - \lVert f(A)-f(N) \rVert^2 \leq 0
$$

**The problem:** if we stop here, the network could "cheat" by simply outputting **0** for every single encoding (or the exact same encoding for every image) — then the inequality above is trivially satisfied ($0 - 0 \le 0$) without the network having learned anything useful at all.

**The fix:** add a **margin** ($\alpha$) that forces a real gap between the two distances:

$$
\lVert f(A)-f(P) \rVert^2 - \lVert f(A)-f(N) \rVert^2 + margin \;\leq\; 0
$$

The margin defines **how far apart** the dissimilar pair must be, relative to the similar pair, in order to properly distinguish the two.

> **Analogy:** Without a margin, a lazy grader could give **every single answer a score of exactly zero** and technically satisfy "the wrong answers shouldn't score higher than the right ones" — because 0 is never higher than 0! Adding a margin is like the teacher setting a **minimum required gap**: "the correct answer must score at least 20 points higher than the wrong answer, not just *any* amount higher." That extra requirement forces the model to actually spread things apart meaningfully, rather than collapsing everything into the same lazy, uninformative point.

---

## 16. The Triplet Loss Function

Given three images $A$, $P$, and $N$, the triplet loss is:

$$
L(A, P, N) = \max\Big(\lVert f(A)-f(P) \rVert^2 - \lVert f(A)-f(N) \rVert^2 + \alpha,\; 0\Big)
$$

- **Sample training set example:** 10,000 pictures of 1,000 people.
- To train the model, we need **pairs of A and P** — i.e. pairs of pictures of the **same** person — which is why the training set needs **multiple photos per identity**, even though the *final deployed* model only needs a **single** reference photo per person (one-shot learning) once it's trained.

> **Analogy:** The `max(..., 0)` at the end works like a **"floor" on a scoreboard that can never go negative.** If the network is already doing a great job — the negative is comfortably farther away than the positive, even after accounting for the margin — the "penalty" calculation would come out negative, but the `max` function simply clips it to **zero**: "no penalty needed, you're already doing fine, don't waste training effort here." Only when the network is genuinely getting it wrong (or not by *enough* of a margin) does the loss become a positive number that actually pushes learning forward.

---

## 17. Choosing Good Triplets

If $A$, $P$, and $N$ are chosen **completely randomly** from a large dataset, the inequality

$$
d(A,P) + \alpha \leq d(A,N)
$$

is **easily satisfied** almost automatically — a randomly picked negative is very likely to already look nothing like the anchor, so the model gets **little useful learning signal** from most random triplets.

**Solution:** deliberately choose triplets that are **"hard" to train on**, to increase learning efficiency — specifically, choose examples where $d(A,P)$ is **close to** $d(A,N)$. *(Refer to the FaceNet paper: Schroff, Kalenichenko & Philbin, 2015.)*

> **Analogy:** Randomly quizzing a geography student with "Is Antarctica cold, or is Antarctica the same as the Sahara Desert?" teaches them almost nothing — the answer is obvious even before thinking. A genuinely useful quiz question picks two things that are **actually close and easy to confuse** — like "Is this Portugal or Spain?" Triplet selection works the same way: to actually sharpen the model, we need to specifically hunt for triplets where the "wrong" answer is deceptively close to the "right" one, not ones where the answer is already painfully obvious.

---

## 18. Triplet Mining: Easy, Hard, and Semi-Hard

Based on the definition of the loss, every triplet falls into one of **three categories**:

| Category | Condition | What it means |
|---|---|---|
| **Easy triplet** | $d(A,P) + margin < d(A,N)$ | Loss is **0** — the negative is already comfortably far away; nothing to learn here |
| **Hard triplet** | $d(A,N) < d(A,P)$ | The negative is actually **closer** to the anchor than the positive is — a serious mistake, big loss |
| **Semi-hard triplet** | $d(A,P) < d(A,N) < d(A,P) + margin$ | The negative isn't closer than the positive, but it's still **too close** — small positive loss |

```text
                Easy negatives
              ╭──────────────────╮
              │  Semi-hard        │
              │  negatives        │
              │  ╭─────────────╮  │
              │  │ Hard        │  │
              │  │ negatives   │  │
              │  │      ●a  ●p │  │◀── margin
              │  ╰─────────────╯  │
              ╰──────────────────╯
        Regions of embedding space, relative to
        the anchor (a) and the positive (p)
```

Each of these three categories depends on **where the negative sits, relative to the anchor and the positive** — so we can extend the same three labels ("hard," "semi-hard," "easy") to describe **negatives** themselves.

> **Analogy:** Picture painting three concentric circles around a dartboard's bullseye (the anchor). Anything **inside the innermost circle** (closer than the positive) is a **"hard negative"** — dangerously close, an easy mistake to make. The **middle ring** (a bit farther than the positive, but still within the margin) holds **"semi-hard negatives"** — close calls that still deserve some training attention. Everything **outside the margin ring** is an **"easy negative"** — so far from the bullseye that there's no realistic chance of confusing it with the real target, so it's not worth spending training time on.

---

## 19. Worked Example: Calculating the Triplet Loss by Hand

Let's use a fixed **margin ($\alpha$) of 0.2** and walk through three different triplets, the same way we'd trace through any formula step by step.

| Case | $d(A,P)$ | $d(A,N)$ | Category (using Section 18's rules) | $L = \max(d(A,P)-d(A,N)+\alpha,\ 0)$ | Result |
|---|---|---|---|---|---|
| **1** | 0.3 | 0.9 | $0.3+0.2=0.5 < 0.9$ → **Easy** | $\max(0.3-0.9+0.2,\ 0)=\max(-0.4,\,0)$ | **0** (no learning signal) |
| **2** | 0.7 | 0.3 | $0.3 < 0.7$ → **Hard** | $\max(0.7-0.3+0.2,\ 0)=\max(0.6,\,0)$ | **0.6** (big penalty) |
| **3** | 0.3 | 0.4 | $0.3 < 0.4 < 0.5$ → **Semi-hard** | $\max(0.3-0.4+0.2,\ 0)=\max(0.1,\,0)$ | **0.1** (small penalty) |

**How to read this table in general:** always compute $d(A,P) - d(A,N) + \alpha$ first — if it comes out **negative**, the `max(..., 0)` clips the loss to zero and the network gets **no gradient signal** from that triplet (Case 1). If it comes out **positive**, that positive value **is** the loss, and it's the network's cue to push the negative farther away and/or pull the positive closer — the **larger** the leftover positive value, the **stronger** the correction (compare Case 2's large 0.6 penalty against Case 3's mild 0.1 penalty).

> **Analogy:** Think of this like a shooting-range coach reviewing three attempts. Case 1 is a shot that already **landed well within the target zone** — the coach says "nothing to correct, well done" (loss = 0). Case 2 is a shot that **completely missed in the wrong direction** — the coach gives strong, detailed correction (loss = 0.6). Case 3 is a shot that landed **just barely outside** the acceptable zone — the coach gives a small nudge of correction (loss = 0.1), enough to improve, but nowhere near as urgent as Case 2's miss.

---

## 20. Offline vs Online Triplet Mining

Because training with **Easy Triplets** wastes effort (their loss is always 0), an important design decision is **how** to select the hard/semi-hard triplets used for training:

| Strategy | How it works |
|---|---|
| **Offline Triplet Mining** | Generate the triplets **manually, ahead of time**, then feed that fixed set of triplets into the network for training |
| **Online Triplet Mining** | Feed in a **batch** of training images, generate triplets **from all examples already in that batch**, and compute the loss on the fly. This randomises the triplets and increases the chance of finding **high-loss (hard/semi-hard)** triplets, which speeds up training. For a batch of size $N$, up to $N^3$ possible triplets can be generated. |

> **Analogy:** Offline mining is like a teacher who **prepares one fixed worksheet of tricky practice questions the night before class** and hands out the exact same worksheet every session — thorough, but it never adapts to what the class is *currently* struggling with. Online mining is like a teacher who, **during** the lesson, watches which questions students are getting wrong **right now**, and generates fresh, targeted practice questions from the current classroom mix on the spot — more dynamic, and far more likely to keep serving up genuinely useful, challenging questions as the class (the model) improves.

---

## 21. Deploying the Network as an Embedding Generator

Once trained, a Siamese/FaceNet-style CNN can be used purely as an **embedding generator** — it converts any face image into a fixed-length numeric vector (an **embedding**), which can then be compared using a **similarity learning metric** and a **threshold** to produce a final prediction.

```text
Face 1 ──▶ [ CNN ] ──▶ embedding  ─┐
                                    ├──▶ [ Similarity metric ] ──▶ [ Threshold ] ──▶ Prediction
Face 2 ──▶ [ CNN ] ──▶ embedding  ─┘        (e.g. Manhattan distance)
```

This single embedding can power **three different applications**:

| Application | How the embedding is used |
|---|---|
| **Face verification** | Compare **two** embeddings and decide "same" or "different" using a distance metric |
| **Face identification** | Treat the embedding as a feature vector and use it to train a classifier that predicts a name/label |
| **Face clustering** | Group similar embeddings together (e.g. with **K-means**) — exactly how photo apps automatically group photos of the same person, without being told anyone's name in advance |

> **Analogy:** Once trained, the CNN acts like a **universal fingerprint scanner** for faces — it doesn't need to know anyone's name to do its job; it just converts any face into a consistent numeric "fingerprint" (the embedding). What you **do** with that fingerprint afterward depends on the task: compare two fingerprints directly (verification), file a fingerprint under a known name (identification), or sort a big pile of unlabelled fingerprints into natural groups without knowing anyone's name at all (clustering) — one underlying tool, three different downstream uses.

---

## 22. Similarity Metrics: How Do We Measure "Distance"?

There are **three broad families** of similarity/distance metrics used to compare two embedding vectors $x^{(i)}$ and $x^{(j)}$.

### 22.1 Cardinality-Based Metrics

Leverage the **union and intersection** of the two vectors being compared. Example: **Jaccard similarity**.

$$
d(x^{(i)}, x^{(j)}) = \frac{x^{(i)} \cap x^{(i)}}{x^{(i)} \cup x^{(i)}}
$$

> **Analogy:** Like comparing two people's music playlists by asking, "**out of all the songs either of you has**, how many do you **both** have in common?" — the more overlap relative to the combined total, the more similar your taste in music.

### 22.2 Orientation-Based Metrics

Leverage the **angle** between two vectors. Example: **Cosine similarity**.

$$
d(x^{(i)}, x^{(j)}) = \cos(\theta) = \frac{\sum_{k=1}^{n} |x_k^{(i)} \cdot x_k^{(i)}|}{\sqrt{\sum_{k=1}^{n} {x_k^{(i)}}^2} \; \sqrt{\sum_{k=1}^{n} {x_k^{(j)}}^2}}
$$

> **Analogy:** Cosine similarity is like comparing two arrows pointing out from the same origin and asking **only** "do they point in roughly the same direction?" — it doesn't care how *long* each arrow is, only their **angle**. Two very short arrows and two very long arrows pointing the exact same direction are considered equally similar.

### 22.3 Distance-Based Metrics

Leverage the **average distance** between corresponding elements of the two vectors. These include:

| Metric | Formula |
|---|---|
| **Euclidean distance** | $d(x^{(i)}, x^{(j)}) = \sqrt{(x_1^{(i)}-x_1^{(j)})^2+(x_2^{(i)}-x_2^{(j)})^2+\cdots+(x_n^{(i)}-x_n^{(j)})^2}$ |
| **Manhattan distance** | $d(x^{(i)}, x^{(j)}) = \lvert x_1^{(i)}-x_1^{(j)} \rvert + \lvert x_2^{(i)}-x_2^{(j)} \rvert + \cdots + \lvert x_n^{(i)}-x_n^{(j)} \rvert$ |
| **Minkowski distance** | $d(x^{(i)}, x^{(j)}) = \left( \sum_{k=1}^{n} \lvert x_k^{(i)} - x_k^{(i)} \rvert^{p} \right)^{\frac{1}{p}}$ |

> **Analogy:** **Euclidean distance** is "as the crow flies" — a straight diagonal line between two points, like measuring distance with a ruler drawn directly between two dots on a map. **Manhattan distance** is how a taxi actually has to travel through a city's **grid of streets** — you can't cut diagonally through buildings, so you add up the horizontal blocks plus the vertical blocks separately. **Minkowski distance** is a **general "dial"** that can be tuned ($p$) to behave like Manhattan distance at one setting and like Euclidean distance at another — think of it as one adjustable ruler that can be configured to measure either "taxi routes" or "straight lines," depending on how you set it.

---

## 23. Face Verification as Binary Classification

Triplet loss is **one** way to train a face-recognition network. There's a second, simpler way: treat it as a **straight binary classification problem**.

```text
x⁽ⁱ⁾ ──▶ [ CNN layers ] ──▶ f(x⁽ⁱ⁾)  ─┐
                                        ├──▶ [ Sigmoid ] ──▶ ŷ   (0 = different, 1 = same)
x⁽ʲ⁾ ──▶ [ CNN layers ] ──▶ f(x⁽ʲ⁾)  ─┘
     (both branches share the SAME weights — this is still a Siamese network)
```

We compute the embeddings of an **image pair** (e.g. 128-dimensional or more) using a Siamese network, then feed both embeddings into a **logistic regression** unit with a final **sigmoid** layer:

$$
Y' = w^{(i)} * \text{Sigmoid}\big(f(x^{(i)}) - f(x^{(j)})\big) + b
$$

where the subtraction represents the **Manhattan distance** between $f(x^{(i)})$ and $f(x^{(j)})$ (Euclidean and chi-square similarity are also commonly used alternatives).

> **Analogy:** If triplet loss is like teaching someone to judge similarity by **comparing three photos at once** (Section 14), this binary-classification approach is like teaching them with a simpler **"yes/no flashcard" quiz**: show two photos side by side, and the only question is "same person, yes or no?" It's a more direct, arguably simpler training signal — no juggling an anchor, positive, *and* negative simultaneously — just a straightforward pairwise judgment call.

---

## 24. Worked Example: Interpreting a Verification Dataset Table

The slides show a small training table for the binary-classification approach:

| $x$ (image pair) | $y$ (label) | Interpretation |
|---|---|---|
| Photo of Person A, Photo of Person B | **0** | "Different" |
| Photo of Person C, Photo of Person D | **0** | "Different" |
| Photo of Person E, Photo of Person E (different shot) | **1** | "Same" |

**How to read a table like this:** each **row** is one training example — a **pair** of images, not a single image — and the label $y$ isn't a person's name at all; it's simply **1** if both photos in the pair show the **same** person, and **0** if they show **different** people. This is exactly what makes it a genuine **binary classification** problem: no matter how many thousands of different people appear across the whole dataset, every single row only ever needs one of **two** possible labels (0 or 1) — which is precisely the property that let us sidestep the "fixed number of output classes" problem described back in Section 9.

> **Analogy:** This table works like a matching game at a party where you're handed **pairs of name tags** and simply asked "do these two tags belong to the same person, yes or no?" — you're never asked to recall someone's actual name from a long list. Whether the party has 10 guests or 10,000 guests, the question format **never changes**: it's always just "same or different?" — which is exactly why this framing sidesteps the problem of a classification layer that would otherwise need to grow every time a new guest arrives.

---

## 25. Key Takeaways

> [!IMPORTANT]
> **The core things to remember from Week 8:**

1. **Face recognition** covers two tasks: **identification** (one-to-many — "who is this?") and **verification** (one-to-one — "is this really who they claim to be?").
2. **Face detection** is the essential first step of any face-recognition pipeline — nothing downstream works if a face isn't found first.
3. **Classical approaches** (Eigenfaces/PCA for holistic features; LBP/SIFT for local features) don't scale well to huge datasets and rely on hand-designed features rather than features the data reveals on its own.
4. Face recognition is a natural example of **one-shot learning** — systems typically have only one (or a few) reference photos per person, yet must generalise to many future, unseen photos.
5. A **standard CNN classifier fails** at this task because (a) there's too little training data per class, and (b) the number of classes changes every time someone new is registered, forcing constant retraining.
6. The solution is to **reframe** the task from classification into a **similarity/difference-evaluation problem**: learn a distance function $d(img1, img2)$, then compare it against a threshold $\tau$.
7. The **Siamese network** architecture uses two (or more) branches with **shared weights** to produce comparable embeddings for different input images.
8. **Triplet loss** trains the network using an **anchor, positive, and negative** image, pulling positives closer and pushing negatives farther away, with a **margin** to prevent the network from collapsing all encodings to the same trivial point.
9. Randomly chosen triplets are usually too easy to be useful — **triplet mining** (selecting hard/semi-hard triplets) is essential for efficient training, and can be done **offline** (pre-generated) or **online** (generated per batch).
10. Once trained, the network becomes a general-purpose **embedding generator**, whose output can support **verification**, **identification**, **or clustering**.
11. Various **similarity/distance metrics** (cardinality-based, orientation-based, distance-based) can be used to compare embeddings, each with a different notion of "closeness."
12. An **alternative** to triplet loss is to train the Siamese network as a **straight binary classifier** ("same" = 1 / "different" = 0) using a sigmoid output layer over the embedding difference.

---

## 26. Glossary

| Term | Definition |
|------|-----------|
| **Face Detection** | Locating a face within an image (no identity involved) — the first step of any face-recognition pipeline |
| **Face Identification** | A one-to-many task: given a face, determine **whose** face it is out of a known database of people |
| **Face Verification** | A one-to-one task: given a face and a claimed identity, determine whether they match |
| **Eigenfaces** | A classical holistic face-recognition method using PCA to represent faces as combinations of "basis faces" |
| **Local Binary Pattern (LBP)** | A classical local feature-extraction technique that encodes texture patterns around each pixel |
| **One-Shot Learning** | A classification approach that must learn from only one (or very few) examples per class, then generalise to new, unseen examples |
| **Embedding / Encoding** | A fixed-length numeric vector output by a network that represents an image's features in a comparable feature space |
| **Siamese Network** | An architecture with two (or more) identical, weight-sharing branches, used to produce comparable embeddings for different inputs |
| **Anchor (A)** | The reference image in a triplet |
| **Positive (P)** | An image of the **same** person as the anchor |
| **Negative (N)** | An image of a **different** person from the anchor |
| **Triplet Loss** | A loss function trained on (anchor, positive, negative) triplets that pulls positives closer and pushes negatives farther from the anchor |
| **Margin ($\alpha$)** | A required minimum gap between $d(A,N)$ and $d(A,P)$, added to the triplet loss to prevent the network from collapsing all encodings to the same point |
| **Easy Triplet** | A triplet whose loss is already 0 — no useful training signal |
| **Hard Triplet** | A triplet where the negative is actually closer to the anchor than the positive is |
| **Semi-Hard Triplet** | A triplet where the negative is farther than the positive, but still within the margin — still produces a small positive loss |
| **Triplet Mining** | The process of selecting useful (hard/semi-hard) triplets for training, rather than easy, uninformative ones |
| **Offline Triplet Mining** | Generating triplets manually ahead of time, before training begins |
| **Online Triplet Mining** | Generating triplets dynamically from each training batch, on the fly |
| **Face Clustering** | Grouping similar face embeddings together (e.g. via K-means) without needing identity labels |
| **Jaccard Similarity** | A cardinality-based similarity metric using the ratio of intersection to union between two vectors |
| **Cosine Similarity** | An orientation-based similarity metric measuring the angle between two vectors |
| **Euclidean Distance** | A straight-line ("as the crow flies") distance-based similarity metric |
| **Manhattan Distance** | A distance-based similarity metric summing absolute differences along each dimension (like navigating a city grid) |
| **Minkowski Distance** | A generalised distance metric that becomes Manhattan or Euclidean distance depending on the parameter $p$ |
| **Threshold ($\tau$)** | The cutoff distance value used to decide whether two faces are "the same" or "different" |
| **Sigmoid Layer** | A final network layer that squashes an output into a 0–1 range, used to output a "same/different" probability in the binary-classification approach to verification |

---

> [!TIP]
> **Study tip for Week 8:** Make sure you can:
> 1. Explain, in your own words, the difference between face **identification** (one-to-many) and face **verification** (one-to-one), and give a real-world example of each.
> 2. Explain **why** a standard CNN softmax classifier is a poor fit for face recognition — cover **both** reasons (too little data per class, and a constantly changing number of classes).
> 3. Walk through the **triplet loss worked example** (Section 19) and calculate the loss by hand for a new set of $d(A,P)$, $d(A,N)$, and margin values.
> 4. Explain **why** a margin is necessary in the triplet loss objective — what would happen without it?
> 5. Describe the difference between **easy**, **hard**, and **semi-hard** triplets, and explain why training on only easy triplets is a waste of effort.
> 6. Compare **offline** and **online** triplet mining — what's the trade-off between them?
> 7. Name the **three** downstream applications an embedding generator can support (verification, identification, clustering), and briefly explain how the embedding is used differently in each.
> 8. Compare the **triplet loss** approach and the **binary classification (sigmoid)** approach to training a face-recognition network — what do they have in common, and what's different about how each one is trained?
> 9. Be able to describe, in your own words, **at least two** different types of similarity metrics (e.g. Euclidean vs Cosine) and explain what aspect of the two vectors each one actually measures.
