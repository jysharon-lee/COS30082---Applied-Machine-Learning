# 📘 Week 6 Study Notes: Transfer Learning for Computer Vision

> **Course:** COS30082 — Applied Machine Learning  
> **Topic:** Transfer Learning for Computer Vision  
> **Prerequisite:** Week 5 (Convolutional Neural Networks)

---

## Table of Contents

1. [What Is Transfer Learning?](#1-what-is-transfer-learning)
2. [Why Use Transfer Learning? (The Purpose)](#2-why-use-transfer-learning-the-purpose)
3. [Traditional ML vs Transfer Learning](#3-traditional-ml-vs-transfer-learning)
4. [Case Study: Understanding Transfer Learning Through Dog Breeds](#4-case-study-understanding-transfer-learning-through-dog-breeds)
5. [Formal Definition: Domain, Task, and the Transfer Learning Objective](#5-formal-definition-domain-task-and-the-transfer-learning-objective)
6. [What Is a Domain?](#6-what-is-a-domain)
7. [What Is a Task?](#7-what-is-a-task)
8. [What, When, and How to Transfer](#8-what-when-and-how-to-transfer)
9. [Key Takeaway: When Does Transfer Learning Work Well?](#9-key-takeaway-when-does-transfer-learning-work-well)
10. [Motivation for Using Transfer Learning in Deep Learning](#10-motivation-for-using-transfer-learning-in-deep-learning)
11. [What Are Pre-trained Models?](#11-what-are-pre-trained-models)
12. [Two Deep Transfer Learning Strategies](#12-two-deep-transfer-learning-strategies)
13. [Strategy 1 — Off-the-Shelf Pre-trained Models as Feature Extractors](#13-strategy-1--off-the-shelf-pre-trained-models-as-feature-extractors)
14. [Strategy 2 — Fine-Tuning Off-the-Shelf Pre-trained Models](#14-strategy-2--fine-tuning-off-the-shelf-pre-trained-models)
15. [Worked Example: Reading the Yosinski Transferability Plot](#15-worked-example-reading-the-yosinski-transferability-plot)
16. [Popular Pre-trained Models and Source Datasets](#16-popular-pre-trained-models-and-source-datasets)
17. [Types of Deep Transfer Learning: An Overview](#17-types-of-deep-transfer-learning-an-overview)
18. [Domain Adaptation](#18-domain-adaptation)
19. [Multi-task Learning (MTL)](#19-multi-task-learning-mtl)
20. [Zero-shot Learning](#20-zero-shot-learning)
21. [One-shot Learning](#21-one-shot-learning)
22. [Key Takeaways](#22-key-takeaways)
23. [Glossary](#23-glossary)
24. [Study Tips for Week 6](#24-study-tips-for-week-6)

---

## 1. What Is Transfer Learning?

### Reusing Knowledge Instead of Starting From Zero

**Transfer Learning** is the ability of a system to recognise and apply knowledge and skills learned from a **previous task** to a **new (novel) task or domain**, as long as the two share some **commonality**.

$$
\boxed{\text{Transfer Learning} = \text{Reuse what was already learned, instead of learning everything from scratch}}
$$

> **Analogy:** Imagine a chef who has spent years mastering Italian cooking — knife skills, how heat behaves, how to balance salt and acid, how sauces thicken. If that chef is then asked to cook French cuisine, they don't need to relearn how to hold a knife or how a pan behaves on heat. They already have transferable skills, so they only need to pick up what's *new* to French cuisine (different sauces, different techniques). That head start is exactly what transfer learning gives a model.

---

## 2. Why Use Transfer Learning? (The Purpose)

| Purpose | Simple Explanation |
|---|---|
| **Leverage knowledge from previously trained models** | Don't reinvent the wheel — reuse what a model already learned |
| **Accelerate the learning process** | Training starts from a "warm" state instead of random weights, so it converges faster |
| **Improve generalisation** | Knowledge learned broadly (e.g. from millions of images) helps the model avoid overfitting to a small new dataset |
| **Tackle dataset bottlenecks** | Reduces how much labelled data you need for the target task |

> [!NOTE]
> In short: transfer learning exists because collecting **millions of labelled images** for every single new problem is expensive, slow, and often impossible. Transfer learning lets us "borrow" knowledge instead.

---

## 3. Traditional ML vs Transfer Learning

### Two Very Different Philosophies

<div align="center">

```text
Traditional ML                              Transfer Learning

Dataset 1 ──▶ Learning system task 1        Source dataset ──▶ Learning system source task
                                                                        │
Dataset 2 ──▶ Learning system task 2                                   ▼
                                                                    knowledge
                                                                        │
                                                                        ▼
                                             Target dataset ──▶ Learning system target task
```

</div>

| | Traditional ML | Transfer Learning |
|---|---|---|
| **Task handling** | Isolated, single task | Target task's learning **relies on** the source task |
| **Knowledge retention** | Knowledge is **not retained** between tasks | Knowledge is **passed on** from source to target |
| **Awareness of other tasks** | Learning does not take into account knowledge from other tasks | Learning is **built on top of** prior knowledge |
| **Result** | Faster learning process, more accurate, and needs less training data | — |

> **Analogy:** Traditional ML is like a student who studies for every exam as if it's their very first day of school — even if they already learned algebra last semester, they'd have to relearn it from scratch for every new subject. Transfer learning is like a student who realises "I already know algebra from last semester, so learning calculus this semester will be much easier" — they build on top of what they already know.

### Visualising the Difference With Classification Boundaries

<div align="center">

```text
Traditional ML                         Transfer Learning

Blue class ──▶ [boundary learned        Blue class ──▶ [boundary learned]
                independently]                                │
                                                                ▼ (knowledge passed on)
Red class  ──▶ [boundary learned        Red class  ──▶ [boundary learned,
                independently,                          benefiting from blue's
                starting from zero]                      knowledge]
```

</div>

- The circles and squares represent **labelled training samples** from two classes.
- The **size** of a circle/square shows its weight (importance) in determining the boundary.
- The dotted/dashed lines are the **classification boundaries**.
- In transfer learning, the second (red) task's boundary is influenced by what was learned from the first (blue) task — leading to a **better-informed boundary**, especially when the red task has very few samples of its own.

---

## 4. Case Study: Understanding Transfer Learning Through Dog Breeds

### Setting Up the Problem

- **Dataset 1**: Large-breed dogs (e.g. German Shepherd, Rhodesian Ridgeback, Bull Mastiff) → trained into "Learning system task 1"
- **Dataset 2**: Small/toy-breed dogs (e.g. Pomeranian, Pug, Shih Tzu, Yorkshire Terrier) → a **different** but **related** classification problem

### Strategy 1: Train a Brand-New Model From Scratch

If we simply reuse the model trained on Dataset 1 (large dogs) to classify Dataset 2 (small dogs) **without any adaptation**, we usually see:

$$
\boxed{\text{Performance degradation, because the model does not generalise well — it is biased toward its original training data}}
$$

> **Analogy:** It's like asking someone who has only ever driven big trucks to suddenly drive a compact sports car without any adjustment period. Their instincts (built entirely around trucks) will actively work against them at first, because the "training data" of their driving experience doesn't match the new situation.

### Strategy 2: Transfer Learning

Instead of training two completely separate models, we treat Dataset 1 as the **source** and Dataset 2 as the **target**, and **transfer** the useful knowledge from source → target.

| Strategy | Approach |
|---|---|
| **Strategy 1: Retrain new model** | Dataset 1 → Learning system task 1 (independent); Dataset 2 → Learning system task 2 (independent, starts from scratch) |
| **Strategy 2: Transfer learning** | Dataset 1 → Learning system task 1 **(source)** → *knowledge transferred* → Learning system task 2 **(target)**, built using Dataset 2 |

> [!TIP]
> Both datasets in this example are **dog breeds**, so they share a lot of "commonality" — general fur textures, ear shapes, eye positions, body proportions. That shared commonality is exactly what makes them good candidates for transfer learning.

---

## 5. Formal Definition: Domain, Task, and the Transfer Learning Objective

### The Textbook Definition

Given:

- A **source domain** $D_S$ and a corresponding **source task** $T_S$
- A **target domain** $D_T$ and a **target task** $T_T$

The objective of transfer learning is to help us learn the **target conditional probability distribution** $P(Y_T \vert X_T)$ in $D_T$, using information gained from $D_S$ and $T_S$, where:

$$
D_S \neq D_T \quad \text{or} \quad T_S \neq T_T
$$

> **Plain-English translation:** "The source data/task and the target data/task don't have to be identical — as long as they're at least somewhat related, we can use what we learned from the source to help predict labels in the target."

### The Three Variations

| Variation | Meaning | Example |
|---|---|---|
| **Same domain, different task** | Same type of input data, but a different question being asked | Photos of dogs → classify breed (source) vs classify age (target) |
| **Different domain, same task** | Same type of question, but the input data "looks" different | Classify vehicles in daylight photos (source) vs classify vehicles in night-vision photos (target) |
| **Different domain, different task** | Both the input data and the question differ | ImageNet object classification (source) vs medical X-ray disease detection (target) |

---

## 6. What Is a Domain?

### Feature Spaces and Distributions

If two **domains** are different, they may have different **feature spaces** or different **marginal distributions**.

> **Analogy:** Think of "domain" as the *type of photograph* — is it a close-up macro shot, a photo taken outdoors in sunlight, or a photo taken in a dim greenhouse? Two datasets can both be about "bell peppers," yet look completely different simply because of lighting, angle, or background — that's a **domain shift**.

<div align="center">

```text
Dataset A (Bell pepper leaves,          Dataset B (Bell pepper plants,
close-up, grey background)              outdoor, in soil/pots, full plant)

     🍃  🍃                                  🌱      🌱
     🍃  🍃                                  🌱      🌱

Same object (bell pepper) — but a very different "look" (domain)
```

</div>

Even though **Dataset A** and **Dataset B** are both about bell peppers, one is a set of individual leaf close-ups on a plain background, and the other is full plants growing outdoors. Their **feature spaces** (what visual patterns dominate) and **distributions** (how those patterns are spread out) can be quite different — this is what we mean by "different domains."

---

## 7. What Is a Task?

### Label Spaces and Conditional Distributions

If two **tasks** are different, they may have different **label spaces** or different **conditional distributions**.

> **Analogy:** Think of "task" as *the actual question you're asking the model to answer*. Even using the exact same kind of photos (same domain — e.g. dog portraits), asking "which specific breed is this?" (Dataset A: Pomeranian, Affenpinscher, Brussels Griffon...) is a **different task** from asking "which specific breed is this?" using an entirely different list of breeds (Dataset B: Manchester Terrier, Miniature Pinscher, Papillon...) — because the **set of possible answers (label space)** is different.

<div align="center">

```text
Dataset A                     Dataset B
Label space:                  Label space:
{Pomeranian, Affenpinscher,   {Manchester Terrier, Miniature Pinscher,
 Brussels Griffon, ...}        Papillon, Pug, Shih Tzu, ...}

              Same "type" of task (breed classification),
              but the answer options don't match up.
```

</div>

---

## 8. What, When, and How to Transfer

### Three Guiding Questions

| Question | Guidance |
|---|---|
| **What to transfer?** | Identify which knowledge is **source-specific** vs **common** between source and target. Be careful of **negative transfer**. The choice of source data/model is an open problem requiring domain expertise or experience |
| **When to transfer?** | Mainly when you **do not have a lot of target data** |
| **How to transfer?** | Covered later in this document — mainly through deep learning strategies (feature extraction / fine-tuning) |

### Negative Transfer

> [!WARNING]
> **Negative transfer** happens when knowledge from the source task actually **hurts** performance on the target task, instead of helping.
>
> **Analogy:** Someone who learned to drive on the **right-hand side** of the road (like in the US) may struggle — and even become *more* dangerous — when they first try driving in a country that drives on the **left** (like the UK or Malaysia/Australia). Their prior "knowledge" (habit) actively works against them in the new context. This is negative transfer: the source knowledge doesn't just fail to help, it actively interferes.

---

## 9. Key Takeaway: When Does Transfer Learning Work Well?

| Situation | Recommendation |
|---|---|
| Source and target domains are **similar** | Transfer learning works fine |
| Domains are **different** but share the **same input data structure** (e.g. image → image, or speech → speech), and **low-level features** could help | Transfer learning still makes sense |
| Domains are **very different** with **very different data structures** (e.g. ImageNet images → Natural Language text) | Better to train a deep network **from scratch** |

> **Analogy:** You can transfer knife skills from Italian cooking to French cooking (both involve chopping vegetables, cooking with heat, plating food) — but you probably *can't* transfer those same knife skills to help you become a better violinist. Skills transfer well when the *underlying structure* of the tasks overlaps; they transfer poorly (or not at all) when the structures are fundamentally unrelated.

---

## 10. Motivation for Using Transfer Learning in Deep Learning

### The Myth vs Reality

<div align="center">

```text
❌ Myth:     "You can't do deep learning unless you have a
              million labelled examples for your problem."

✅ Reality:  "You CAN transfer learned representations
              from a related task."
```

</div>

### Two Practical Motivations

1. **Reusing low-level features** — In computer vision, low-level features such as edges, curves, and contours learned from a huge dataset (e.g. ImageNet) tend to be broadly useful for *many other* vision problems (e.g. object recognition).
2. **The amount of available data matters** — If you have a huge amount of data for Problem A (e.g. ImageNet) but only a small dataset for Problem B, this strongly motivates using transfer learning techniques (e.g. fine-tuning) for Problem B.

> **Analogy:** Think of low-level features (edges, curves, textures) like the "alphabet" of vision. Once a model has learned this visual alphabet from millions of images, it doesn't need to relearn the alphabet every time it sees a new type of object — it just needs to learn how to combine that alphabet into new "words" (e.g. recognising a specific dog breed or a specific plant disease).

---

## 11. What Are Pre-trained Models?

**Pre-trained models** are the details of neural networks — usually the **millions of parameters/weights** — shared publicly by teams or individuals after training the network to a stable, high-performing state.

> **Analogy:** A pre-trained model is like buying a **cake base** that a professional baker has already perfected — properly risen, evenly baked, good texture. You don't need to bake a cake from raw flour and eggs; you can simply decorate (customise) that base for your own occasion. Similarly, we don't need to train a CNN from random weights — we can start from a pre-trained one and adapt it.

---

## 12. Two Deep Transfer Learning Strategies

There are two main strategies used to apply transfer learning with deep networks:

1. **Off-the-shelf pre-trained models as feature extractors**
2. **Fine-tuning off-the-shelf pre-trained models**

<div align="center">

```text
                         ┌── Strategy 1: Feature extraction (freeze everything, lr = 0)
Pre-trained model ──────▶│
                         └── Strategy 2: Fine-tuning (unfreeze some/all layers, lr > 0)
```

</div>

---

## 13. Strategy 1 — Off-the-Shelf Pre-trained Models as Feature Extractors

### The Key Idea

Simply **leverage** the weighted layers of the pre-trained model to **extract features**, but do **not** update those weights during training on the new data.

**Approach:** Use the output of one or more layers of a network trained for a *different* task as **generic feature extractors**. Then train a **new, shallow classifier** (e.g. SVM, random forest) on top of those extracted features.

<div align="center">

```text
Pre-training (on source data):                 Transfer (to target data):

source data (xs, ys)                           target data and labels (xt, yt)
      │                                                │
   conv1 ──▶ conv2 ──▶ conv3 ──▶ fc1 ──▶ fc2      conv1 ──▶ conv2 ──▶ conv3 ──▶ fc1
      │                                    │           (all FROZEN, lr = 0)      │
      ▼                                    ▼                                     ▼
    loss ◀───────── softmax ◀──────────────┘                          Shallow classifier
                                                                        (e.g. SVM, random forest)
```

</div>

> **Analogy:** This is like hiring a professional photographer (the frozen pre-trained network) to take high-quality photos of your product, and then having a separate, simpler person (the shallow classifier) just sort those photos into folders. The photographer's *skills* (feature extraction) don't change — only the sorting logic on top does.

### When Does This Work Well?

- Off-the-shelf pre-trained models are normally used when the **source and target datasets are from the same domain**, and the **target dataset is small**.
- Research has shown that CNN off-the-shelf features work well for **fine-grained classification**, but **fine-tuning** (Strategy 2) can lead to further improvement through better adaptation.

---

## 14. Strategy 2 — Fine-Tuning Off-the-Shelf Pre-trained Models

### The Key Idea

This is a **more complex** technique. We not only **replace** the last layer (for classification/regression), but we may also selectively **retrain** some of the previous layers. We use the *global architecture* of the pre-trained network and its learned weights as a **starting point** for further training, rather than starting from random weights.

**Approach:** **Freeze** (fix weights, learning rate = 0) some layers during retraining, or **fine-tune** (learning rate > 0) others, according to our needs.

<div align="center">

```text
source data (xs, ys)                    target data and labels (xt, yt)
      │                                          │
  conv1 → conv2 → conv3 → fc1 → fc2       conv1 → conv2 → conv3 → fc1 → fc2 (NEW) → softmax (NEW)
      │                    │                lrp = 0 (frozen)     lrn > lrp (new layers,
      ▼                    ▼                    OR                learn faster)
    loss ◀────── softmax ◀─┘                lrp > 0 (fine-tuned)
```

</div>

> **Analogy:** Fine-tuning is like buying a well-tailored suit off the rack (the pre-trained network) and then bringing it to a tailor to **adjust the fit** to your specific body (fine-tuning some layers), rather than sewing a brand-new suit from raw fabric (training from scratch). You keep the expensive, hard-to-replicate craftsmanship (the frozen layers) and only adjust what's specific to you (the new/fine-tuned layers).

### Question 1: Freeze or Fine-Tune?

It depends on the target task:

| Situation | Recommendation |
|---|---|
| **Target data is scarce** | **Freeze** layers — this avoids overfitting |
| **More target data is available** | **Fine-tune** layers — the model can safely adjust more without overfitting |

> [!NOTE]
> "Freeze" means the weights are **not updated** during training (i.e. they are skipped during backpropagation).

### Question 2: From Which Layer Should Feature Transfer Be Introduced?

This is a much trickier question — and it's exactly what the famous **Yosinski et al. (2014)** experiment investigated (see the worked example in Section 15).

---

## 15. Worked Example: Reading the Yosinski Transferability Plot

> **Source:** Yosinski, J., Clune, J., Bengio, Y. and Lipson, H., 2014. *How transferable are features in deep neural networks?* Advances in Neural Information Processing Systems, pp. 3320–3328.

This is one of the most important plots in transfer learning — let's break it down **step by step**, the same way we'd work through a convolution or pooling calculation.

### Setting Up the Experiment

- Two similar datasets, **A** and **B**, are each split in half: **baseA** (trained and tested only on A) and **baseB** (trained and tested only on B).
- A network is trained on dataset A (called `WA1, WA2, ... WA8`, 8 layers deep).
- The network is then **"chopped"** at layer $n$ (somewhere between layer 0 and layer 7), and everything **after** that cut point is thrown away and replaced with **new, randomly initialised layers**.
- The new network is then retrained on dataset **B**.

### Reading the Legend

| Legend label | What it means |
|---|---|
| **baseB** | Baseline: trained and tested entirely on B (no transfer at all) |
| **selffer BnB** | Layers were transferred from B → B (i.e. "transferred" from the *same* dataset), then **frozen** |
| **selffer BnB⁺** | Same as above, but the transferred layers are **fine-tuned** (the `+` always means "fine-tuned") |
| **transfer AnB** | Layers were transferred from A → B (a genuinely *different* source dataset), then **frozen** |
| **transfer AnB⁺** | Same as above, but the transferred layers are **fine-tuned** |

The **x-axis** is the layer $n$ at which the network was "chopped" (0 = nothing transferred, 7 = almost the whole network transferred). The **y-axis** is Top-1 accuracy — higher is better.

> **Analogy for the whole setup:** Imagine renovating a house (network B) using either your **own** old blueprints (`selffer`, i.e. from the same house/dataset) or a **neighbour's** blueprints (`transfer`, i.e. from a different house/dataset A). You can choose to either **keep the old rooms exactly as they were** (frozen, no `+`) or **remodel them to fit your needs** (fine-tuned, `+`).

### Interpreting the Plot, Section by Section

| Layer region | What the plot shows | Interpretation |
|---|---|---|
| **Layers 0–2 (early layers)** | All variants (selffer, transfer, frozen, fine-tuned) sit close to the baseline | Early layers learn very generic features (edges, colours, blobs) that transfer well **almost regardless** of the source — like a solid foundation that works for any house |
| **Middle layers (roughly 3–6), frozen** | Accuracy noticeably **drops**, especially for `transfer AnB` (no `+`) | **Fragile co-adaptation**: middle layers had learned to work *together* with the specific layers that came after them in the original network. Chopping them off and bolting on new random layers breaks that teamwork, and frozen middle layers can't repair it |
| **Middle layers, fine-tuned (`+`)** | Accuracy **recovers**, staying close to (or above) baseline | Fine-tuning **restores the broken co-adaptation** — the layers get a chance to re-adjust and work with their new neighbours |
| **Later layers (chopped near 6–7), frozen `transfer AnB`** | Accuracy drops the **most** here | **Representation specificity**: the last few layers of the source network became highly specialised for the *source* task specifically, so they're less useful, as-is, for a different target task |
| **`transfer AnB⁺` (transfer + fine-tune), across ALL layers** | Consistently at or **above** the baseline, and this gap **grows** at deeper layers | **Transfer + fine-tuning improves generalisation** — even features that started out too source-specific become genuinely useful once fine-tuned, and the model benefits from having seen *more total data* (source + target) overall |

### The Big Picture (5 Numbered Insights)

<div align="center">

```text
Top-1
accuracy
  ▲
  │  5: Transfer + fine-tuning improves generalisation ─────────────────
  │                                     ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  │  3: Fine-tuning recovers co-adapted interactions ── ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾
  │ ●━━━━━━━━━●
  │                  ╲                                     4: Performance
  │                    ╲   2: Performance drops due to        drops due to
  │                      ╲    fragile co-adaptation             representation
  │                        ╲______________________            specificity
  │                                                ╲________
  └──────────────────────────────────────────────────────────▶
    0        1        2        3        4        5        6        7
                  Layer n at which network is chopped and retrained
```

</div>

1. **Transfer + fine-tuning improves generalisation** — the best-performing curve overall.
2. **Performance drops due to fragile co-adaptation** — happens with *frozen* middle layers.
3. **Fine-tuning recovers co-adapted interactions** — the `+` variants bounce back up.
4. **Performance drops due to representation specificity** — happens with *frozen* late layers, and is worse than fragile co-adaptation.
5. **Transfer + fine-tuning improves generalisation**, restated — this combination is consistently the strongest across the whole plot.

### Practical Rules of Thumb From This Plot

| Rule | Why |
|---|---|
| **Avoid chopping/freezing *middle* layers** | Harder for the new model to relearn, due to fragile co-adaptation |
| **When source and target are very similar**, fine-tuning just the last FC layer is often enough | Saves time, same performance (see `selffer BnB` results at high layer numbers) |
| **When source and target are different**, keep the network structure and **fine-tune the whole thing**, using pre-trained weights only as *initialisation* | Boosts generalisation performance |
| **Only fine-tune the whole network if the target dataset is not too small** | A tiny target dataset + a fully fine-tuned (very flexible) network = high risk of **overfitting** |

---

## 16. Popular Pre-trained Models and Source Datasets

### Common Pre-trained Backbones for Computer Vision

| Model family | Examples |
|---|---|
| **VGG** | VGG-16 |
| **GoogleNet** | Inception V3 |
| **ResNet** | — |

### Common Source Datasets, by Application

| Application | Typical source datasets |
|---|---|
| **Image classification** | ImageNet, Places Dataset |
| **Video classification** | Sport-1M, Kinetics, ActivityNet |

> [!TIP]
> The choice of source dataset should match the *type* of target problem — a video-classification target task benefits more from a video-based source dataset (e.g. Kinetics) than from a purely image-based one (e.g. ImageNet), even though both are technically "visual" data.

---

## 17. Types of Deep Transfer Learning: An Overview

Transfer learning is a **general concept** — a principle of solving a target task using knowledge from a source task's domain. It shows up in several different flavours in deep learning:

<div align="center">

```text
                        ┌── Domain Adaptation
                        │
Transfer Learning ──────┼── Multi-task Learning
   (general concept)    │
                        ├── Zero-shot Learning
                        │
                        └── One-shot Learning
```

</div>

The remaining sections walk through each of these four in turn.

---

## 18. Domain Adaptation

### The Challenge

The distribution of data in the **target domain** is different from the **source domain** — this is called **domain shift**.

**The central question:** How do we overcome the differences between domains, so that a classifier trained on the source domain **generalises well** to the target domain?

<div align="center">

```text
        target domain  ~~~~~~~~~~~~~~~~~~~~~
                       ╱   ▽▽▽ (chair)   ●●● (mug)  ╲
                      ╱                              ╲
        source domain +++  (TV)   ▽▽▽ (chair)   ○○○ (mug)
```

</div>

The **same object category** (e.g. "chair" or "mug") can look statistically different depending on which domain it was captured in — different lighting, angle, background, or even a completely different type of image (drawing vs photo).

> **Analogy:** Imagine you trained a translator using only *formal written English* (the source domain). If you then ask that translator to interpret **casual spoken slang** (the target domain), even though it's still "English" (the same task — translation), the *style* of the input has shifted so much that performance drops. Domain adaptation is the process of helping the translator adjust to this shift.

### Three Approaches to Domain Adaptation

1. **Divergence-based Domain Adaptation**
2. **Adversarial-based Domain Adaptation**
3. **Reconstruction-based Domain Adaptation**

#### 18.1 Divergence-Based Domain Adaptation

Works on the principle of **minimising** some **divergence-based criterion** between the source and target feature **distributions**, so that the resulting features become **domain-invariant** (i.e. they no longer carry a "signature" of which domain they came from).

<div align="center">

```text
During Training:
Source domain ──▶ f() ──▶ features ──▶ Classification Loss (map inputs to correct class)
                              │
                    Divergence-Based Loss (keep source & target features similar)
                              │
Target domain ──▶ f() ──▶ features

During Inference:
Target domain ──▶ f() (Feature Extractor) ──▶ Flattened Feature ──▶ Classifier ──▶ Output
```

</div>

> **Analogy:** This is like two people from different countries agreeing to both learn a **shared simplified language** (e.g. basic English) so they can communicate, rather than one person having to fully learn the other's native language. The "divergence loss" is the pressure that keeps pushing both "languages" (feature representations) to stay close to each other.

> [!NOTE]
> All the divergence formulas used here are usually **non-parametric, hand-crafted mathematical formulas** — they aren't specifically tailored to your dataset or your exact problem (classification, detection, segmentation, etc.).

#### 18.2 Adversarial-Based Domain Adaptation

**Adversarial learning** is a technique used to train robust deep networks that can generate complex samples across diverse domains — it works by trying to **fool or misguide** a model with tricky input.

For domain adaptation specifically: the model learns a **discriminative mapping** of target images into the **source feature space** (via a "target encoder"), by trying to **fool a domain discriminator** that is meanwhile trying to tell apart encoded target images from real source examples.

<div align="center">

```text
Step 1: Pre-training                Step 2: Adversarial Adaptation         Step 3: Testing

source images+labels                source images ──▶ Source CNN ──┐      target image
      │                                                             ├─▶ Discriminator   │
   Source CNN ──▶ Classifier         target images ──▶ Target CNN ──┘   ──▶ domain label  Target CNN ──▶ Classifier ──▶ class label
      │                                     (trying to fool the discriminator
   class label                               into thinking target = source)
```

</div>

> **Analogy:** This is exactly like a game between a **forger** and an **art detective**. The forger (target encoder) tries to make fake paintings (target images) that look so convincing the detective (discriminator) can't tell them apart from real ones (source images). As the forger gets better at fooling the detective, its "fake" paintings become genuinely useful stand-ins for the real thing — meaning the target images can now be classified using the classifier that was trained only on source (real) images.

**Step by step:**

1. **Pre-train** a source encoder CNN using labelled source image examples.
2. The **discriminator** helps produce features that are **indistinguishable** between source and target domains.
3. At **test time**, target images are mapped through the target encoder into the shared feature space, and classified using the (already-trained) source classifier.

#### 18.3 Reconstruction-Based Domain Adaptation

This works on the idea of **image-to-image translation**. The simplest model is an **encoder-decoder** network, paired with a **discriminator** that pushes the encoder-decoder to produce images that look like they belong to the **source domain**.

<div align="center">

```text
Target domain image ──▶ Encoder-Decoder Network g() ──▶ "Generated Source domain" image
                                                                    │
Source domain image ─────────────────────────────────────────▶ Discriminator ──▶ Real or Fake?
```

</div>

**Then:** a separate **source domain classifier** is trained purely on source data.

<div align="center">

```text
Generating Pipeline:                     Classification Pipeline:
Target domain image ──▶ g() ──▶           Source domain image ──▶ f() ──▶ ... ──▶ Output
                     "translated" image
                     (looks like source)
```

</div>

**At inference time**, a target-domain image is first **translated** into a source-domain-*looking* image (using $g()$), and this translated image is then passed through the **source domain classifier** ($f()$, trained during Step 2) for prediction.

> **Analogy:** This is like translating a foreign recipe (target domain) into your **own language and measurement units** (source domain) *before* cooking it using your trusted, familiar cookbook (the source classifier) — rather than trying to build a brand-new cookbook that understands every possible language and unit system directly.

---

## 19. Multi-task Learning (MTL)

### A Different Flavour of Transfer Learning

**Multi-task learning** is a *slightly different* branch of the transfer learning world. Given a set of learning tasks $t_1, t_2, \dots, t_n$, the learner **co-learns all tasks simultaneously**, optimising performance across all $n$ tasks through **shared knowledge**.

$$
\boxed{\text{MTL: the learner receives information on multiple tasks at once, without distinguishing between "source" and "target"}}
$$

### Transfer Learning vs MTL

<div align="center">

```text
Transfer Learning                         Multitask Learning

Source dataset (big samples)              Task 1 Trained model ◀──▶ Task 2 Trained model
      │                                          ▲       ╲knowledge╱       ▲
  Learning system source task                    │        sharing         │
      │                                          ▼                        ▼
   knowledge                               Task 3 Trained model ◀──▶ Task 4 Trained model
      │
      ▼
Target dataset (small samples)
      │
  Learning system target task
```

</div>

| | Transfer Learning | Multi-task Learning |
|---|---|---|
| **Direction of knowledge flow** | One-directional: source → target | All tasks share knowledge with each other simultaneously |
| **Roles of the datasets** | Clearly split into "source" (big) and "target" (small) | No source/target distinction — all tasks are learned together |

> **Analogy:** Transfer learning is like a **senior mentor** passing knowledge down to a **junior mentee** — a one-way flow. Multi-task learning is more like a **study group**, where several students studying *related* subjects (say, biology and chemistry) all sit together and help each other, because the underlying concepts (e.g. molecules, cells) reinforce each subject at the same time.

### Generic Multi-task Learning Architecture

**Case study: Plant disease identification.** A single plant photo needs to predict **two related things at once**: the *host species* and the *disease*.

<div align="center">

```text
                     host species     disease
                          ▲              ▲
                    ┌──────────┐   ┌──────────┐
                    │Task-spec.│   │Task-spec.│   ← Task-specific layers
                    │  layers  │   │  layers  │
                    └────┬─────┘   └────┬─────┘
                         └───────┬───────┘
                          Shared bottom layers
                        (common representation)
                                 │
                          Plant disease image
```

</div>

Generic MTL consists of **two components**:

1. **Shared bottom layers** — a common representation learned from the input, useful for *all* the tasks.
2. **Task-specific layers** — separate "output branches," one per task, sitting on top of the shared layers.

### When Does MTL Make Sense?

| Condition | Explanation |
|---|---|
| **Tasks could benefit from sharing lower-level features** | E.g. the features of a plant's host species can help identify diseases — pathologists themselves use host-species features to infer likely diseases |
| **The amount of data per task is roughly similar** | Having *more total relevant data* (e.g. 10,000 samples for host species + 10,000 for disease = 20,000 combined) can help increase performance, compared to training on only one task's data |
| **You can afford to train a larger network** | The network size must scale with the number of tasks it's being asked to handle well |

<div align="center">

```text
host species (10,000 samples) ─┐
                                ├──▶ Generic MTL (20,000 combined, relevant samples)
disease (10,000 samples)      ─┘
```

</div>

**Real-world example — FashionNet:** A single large network simultaneously predicts *clothing styles*, *clothing attributes* (Clothes Recognition), and *landmark localisation* (Clothes Alignment) — all from one shared backbone, branching into task-specific heads.

> [!NOTE]
> The size of the neural network must be taken into consideration if you want it to adapt well to *many* tasks at once — a network too small for the number of tasks will struggle to do all of them well.

---

## 20. Zero-shot Learning

### Recognising What You've Never Seen

**Zero-shot Learning (ZSL)** enables identification of classes that are **not seen before**, by transferring knowledge from **seen classes** to **unseen classes**. This is usually done using **auxiliary information** — such as text descriptions, semantic attributes, or **word embeddings** (vector representations of a word).

> **Analogy:** Imagine describing a **zebra** to someone who has never seen one, by saying "it's basically a horse, but with black-and-white stripes." That person can now recognise a real zebra the very first time they see it — even though they've never seen one before — because they combined **prior knowledge** ("what a horse looks like") with a **description** ("striped, black and white"). That's exactly the idea behind zero-shot learning.

### Examples of Auxiliary Information

| Type | Example |
|---|---|
| **Semantic attributes** | Labelling a zebra image with attributes like "stripe," "head," "tail," "long leg" |
| **Word embeddings** | Representing words as points/vectors in space, where similar-meaning words end up close together |

### The Idea Behind Zero-Shot Learning, Step by Step

**Step 1 — Learn a projection function** from the **visual feature space** (i.e. image features) to the **semantic embedding space** (i.e. word vectors), using labelled data from **seen classes only**.

<div align="center">

```text
feature vector space              semantic embedding space

 seen classes:                     ●  (tiger)
 △ (tiger)  ◇ (?)  □ (?)   ──▶      ●  (chimp)
                                     ●  (lion)
       (train a projection: feature space → semantic space)
```

</div>

**Step 2 — Pass unseen class images** through this same trained projection, to get their corresponding point in **semantic embedding space**.

<div align="center">

```text
unseen class (e.g. dog) features ──▶ [trained projection] ──▶ some point in semantic space
```

</div>

**Step 3 — Nearest-neighbour search**: find which known semantic embedding is **closest** to this new point. The label of that closest match becomes the predicted label for the unseen image.

<div align="center">

```text
Test image (unseen: dog) ──▶ projected point ──▶ nearest neighbour search
                                                          │
                                                          ▼
                                            closest semantic embedding = predicted label
```

</div>

> [!IMPORTANT]
> Zero-shot learning doesn't require **any** training images of the target/unseen class — it only requires that the unseen class's **semantic description** (attributes/word embedding) be known in advance.

---

## 21. One-shot Learning

### Learning From Just One (or a Few) Examples

**One-shot learning** is a variant of transfer learning where we try to infer the required output based on **just one or a few** training examples. It emphasises **knowledge transfer** — using prior knowledge of previously learned categories to allow learning from minimal training examples.

### Case Study: Face Recognition

<div align="center">

```text
       A (Adele)              C (Celine Dion)
         ?  ╲              ╱  ?
             ╲            ╱
              Test (Johnny Depp)
             ╱            ╲
         ?  ╱              ╲  ?
       B (Dwayne Johnson)    D (Johnny Depp)
```

</div>

**The question:** Which of A, B, C, or D is the same person as the "Test" photo?

### The Problem With a Naïve Convolutional Network

- A **small training set is not enough** to train a robust neural network — the learned features won't contain the important information needed for future recognition.
- **Retraining the whole network** every single time a new person (class) is added is **far too time-consuming and resource-intensive**.

> **Analogy:** Imagine a security guard being asked to memorise the *entire face* of every single new employee by studying **thousands of photos of each one**, and having to be **retrained from scratch** every time a new employee joins. That's completely impractical! Instead, a smart guard just needs to learn one general skill: "given any two photos, tell me if they're the same person." That skill, once learned, works for *any* new employee — even ones the guard has only seen **once**.

### The Idea Behind One-Shot Learning

Rather than training a model to output a **class label** directly, we train it to learn a **similarity function**: the model takes **two images** and returns a value showing how *similar* they are.

<div align="center">

```text
Image A ──┐
          ├──▶ One-shot learning model ──▶ Distance-based loss / similarity score
Image B ──┘
```

</div>

**The decision rule:**

$$
\text{If } d(\text{img1}, \text{img2}) \leq \lambda \Rightarrow \text{same person/object}
$$
$$
\text{else } d(\text{img1}, \text{img2}) > \lambda \Rightarrow \text{different person/object}
$$

where $\lambda$ is a chosen **threshold** (e.g. zero, or some small value).

<div align="center">

```text
d(A, Test) > λ  →  not the same person       d(D, Test) ≤ λ  →  SAME person!
d(B, Test) > λ  →  not the same person       (Test = Johnny Depp, D = Johnny Depp)
d(C, Test) > λ  →  not the same person
```

</div>

> [!TIP]
> Because the model learns a **general notion of "same vs different"** (rather than memorising specific class labels), it generalises to **brand-new people/objects it has never seen during training** — this is exactly why it's called "one-shot": at deployment time, you might only need **one reference photo** per new person for the model to be able to recognise them.

---

## 22. Key Takeaways

> [!IMPORTANT]
> **The core things to remember from Week 6:**

1. **Transfer learning** reuses knowledge learned from a **source task/domain** to help solve a related **target task/domain**, instead of learning everything from scratch
2. Unlike traditional ML (isolated, single-task learning), transfer learning lets the target task's learning **rely on** knowledge gained from the source task — leading to **faster training, better accuracy, and less required data**
3. Formally: transfer learning helps learn $P(Y_T \vert X_T)$ in a target domain $D_T$, using knowledge from a source domain $D_S$ and task $T_S$, where $D_S \neq D_T$ or $T_S \neq T_T$
4. A **domain** differs by **feature space / marginal distribution** (what the data "looks like"); a **task** differs by **label space / conditional distribution** (what question is being asked)
5. Be aware of **negative transfer** — when source knowledge actively **hurts** target performance, like habits from driving on the wrong side of the road
6. Transfer learning works best when domains are **similar**, or at least share the **same input data structure**; very different data structures (e.g. images vs text) usually call for training from scratch instead
7. There are **two deep transfer learning strategies**: (1) using a pre-trained model purely as a **frozen feature extractor**, and (2) **fine-tuning** some or all of a pre-trained model's layers
8. Freeze layers when target data is **scarce** (to avoid overfitting); fine-tune when there's **more** target data available
9. The **Yosinski transferability plot** shows that middle layers suffer from **fragile co-adaptation** when frozen, late layers suffer from **representation specificity**, and **fine-tuning after transfer generally beats every other option**, especially as more layers are transferred
10. Popular pre-trained backbones include **VGG, GoogleNet (Inception), and ResNet**, typically pre-trained on **ImageNet, Places**, or (for video) **Sport-1M/Kinetics/ActivityNet**
11. Beyond the two basic strategies, transfer learning shows up as **Domain Adaptation, Multi-task Learning, Zero-shot Learning,** and **One-shot Learning**
12. **Domain adaptation** tackles *domain shift* using **divergence-based** (minimise distribution differences), **adversarial-based** (fool a discriminator), or **reconstruction-based** (translate target images to look like source images) approaches
13. **Multi-task learning** trains on several related tasks **at once**, sharing bottom layers while keeping task-specific top layers — useful when tasks share low-level features and have similar amounts of data
14. **Zero-shot learning** recognises entirely **unseen classes** by projecting both images and class descriptions into a shared **semantic embedding space**, then using nearest-neighbour search
15. **One-shot learning** trains a **similarity function** (rather than a fixed classifier) so that new classes can be recognised from **just one or a few** reference examples, using a distance threshold $\lambda$

---

## 23. Glossary

| Term | Definition |
|------|-----------|
| **Transfer Learning** | Reusing knowledge learned from a source task/domain to help solve a related target task/domain |
| **Source Domain / Source Task** | The domain/task that already has a trained model and abundant data |
| **Target Domain / Target Task** | The new domain/task we actually want to solve, usually with limited data |
| **Domain** | Defined by the feature space and marginal data distribution — "what the data looks like" |
| **Task** | Defined by the label space and conditional distribution — "what question is being asked" |
| **Domain Shift** | The difference in data distribution between a source domain and a target domain |
| **Negative Transfer** | When knowledge from the source task actively **hurts** performance on the target task |
| **Pre-trained Model** | A network's weights/parameters, shared publicly after being trained to a stable state |
| **Feature Extractor (frozen model)** | Using a pre-trained model's layers to generate features, without updating their weights |
| **Fine-tuning** | Retraining some or all of a pre-trained model's layers on new (target) data |
| **Freezing** | Fixing a layer's weights so they are not updated during backpropagation (learning rate = 0) |
| **Fragile Co-adaptation** | Performance drop caused by breaking the "teamwork" between layers that had learned to work together |
| **Representation Specificity** | Later layers becoming too specialised to the source task, making them less transferable |
| **Domain Adaptation** | Techniques to help a classifier trained on a source domain generalise well to a shifted target domain |
| **Divergence-based Domain Adaptation** | Minimising a distance/divergence metric between source and target feature distributions |
| **Adversarial-based Domain Adaptation** | Using a discriminator to force target features to become indistinguishable from source features |
| **Reconstruction-based Domain Adaptation** | Translating target-domain images into source-domain-looking images before classification |
| **Multi-task Learning (MTL)** | Training a single model on several related tasks simultaneously, sharing lower-level representations |
| **Shared Bottom Layers** | The common layers in an MTL architecture, learned jointly across all tasks |
| **Task-specific Layers** | The separate output branches in an MTL architecture, one per task |
| **Zero-shot Learning (ZSL)** | Recognising classes never seen during training, using auxiliary semantic information |
| **Semantic Attributes** | Human-interpretable descriptive features (e.g. "stripe," "long leg") used as auxiliary information in ZSL |
| **Word Embedding** | A vector representation of a word, capturing semantic meaning/relationships |
| **Seen / Unseen Classes** | "Seen" = classes with training examples; "Unseen" = classes with no training examples (only descriptions) |
| **One-shot Learning** | Learning to recognise a new class from just one (or very few) training examples |
| **Similarity Function** | A learned function that returns how "similar" two inputs are, instead of a direct class label |
| **Distance Threshold (λ)** | The cutoff value used to decide "same" vs "different" in one-shot learning |
| **Inductive Learning** | The general principle of inferring a mapping from a set of training examples |
| **Inductive Transfer** | Using the inductive biases of a source task to help narrow the hypothesis space for a target task |

---

> [!TIP]
> **Study tip for Week 6:** Make sure you can:
> 1. Explain, in your own words, the difference between a **domain** and a **task** in transfer learning
> 2. Describe the two main deep transfer learning strategies (feature extraction vs fine-tuning) and know **when** to use each one
> 3. Walk through the Yosinski transferability plot and explain **why** performance dips in the middle layers and at the late layers, and why fine-tuning + transfer wins overall
> 4. Compare **transfer learning** vs **multi-task learning** — who shares knowledge with whom, and in which direction?
> 5. Explain how **zero-shot learning** can recognise a class it has never seen, using only a description
> 6. Explain how **one-shot learning** uses a similarity function and a distance threshold instead of a fixed classifier
> 7. Give your own analogy for **negative transfer**, and explain why it can happen
> 8. Name the three domain adaptation approaches (divergence, adversarial, reconstruction) and briefly describe how each one works
