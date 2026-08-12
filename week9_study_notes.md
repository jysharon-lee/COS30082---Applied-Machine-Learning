# 📘 Week 9 Study Notes: Visualizing and Understanding Convolutional Networks

> **Course:** COS30082 — Applied Machine Learning  
> **Topic:** Visualizing and Understanding Convolutional Networks  
> **Prerequisite:** Weeks 1–6 (CNN fundamentals, convolution, pooling, backpropagation)

---

## Table of Contents

1. [What Has a CNN Actually Learned?](#1-what-has-a-cnn-actually-learned)
2. [The Black Box Problem](#2-the-black-box-problem)
3. [What Is Feature Visualization?](#3-what-is-feature-visualization)
4. [What Does "Unit" Mean?](#4-what-does-unit-mean)
5. [Three Ways to Visualize What a CNN Learns](#5-three-ways-to-visualize-what-a-cnn-learns)
6. [Method 1: Hidden Layer Output Visualization](#6-method-1-hidden-layer-output-visualization)
7. [Method 2: Gradient-based Visualization — Overview](#7-method-2-gradient-based-visualization--overview)
8. [DeConvNet (Deconvolutional Network)](#8-deconvnet-deconvolutional-network)
9. [Vanilla Gradient (Saliency Maps)](#9-vanilla-gradient-saliency-maps)
10. [Guided Backpropagation](#10-guided-backpropagation)
11. [Disadvantage Shared by Methods 1 & 2](#11-disadvantage-shared-by-methods-1--2)
12. [Method 3: Feature Visualization by Optimization](#12-method-3-feature-visualization-by-optimization)
13. [The Enemy of Feature Visualization: Fooling Images](#13-the-enemy-of-feature-visualization-fooling-images)
14. [Taming Fooling Images with Regularization](#14-taming-fooling-images-with-regularization)
15. [Key Takeaways](#15-key-takeaways)
16. [Glossary](#16-glossary)
17. [Study Tips for Week 9](#17-study-tips-for-week-9)

---

## 1. What Has a CNN Actually Learned?

A CNN like **AlexNet** can take a photo of a diseased leaf and confidently output **"Early blight disease"** — but *how* did it arrive at that answer? Internally, the network is doing two broad jobs:

<p align="center">
  <img src="https://github.com/user-attachments/assets/55b0cfc5-39f1-4777-bd42-115439576922">
</p>

$$
\boxed{\text{CNN pipeline} = \underbrace{\text{Feature Extraction}}_{\text{convolution + pooling}} \;+\; \underbrace{\text{Classification}}_{\text{fully connected layers}}}
$$

The convolution and pooling layers repeatedly transform the image into **convoluted maps** and **pooled maps**, and the fully connected layers turn those maps into a final decision (e.g. the "Species Class" neuron lighting up for "Early blight disease").

> **Analogy:** Think of a CNN as a long assembly line in a factory. Raw material (the image) goes in one end, passes through many stations that each reshape and refine it (convolution + pooling = feature extraction), and a final inspector at the end of the line stamps a label on the finished product (fully connected layers = classification). The question this whole topic asks is: **what is actually happening at each station along the way?**

---

## 2. The Black Box Problem

If we strip away all the internal detail, a trained CNN can be drawn as a single opaque rectangle: image goes in, "Early blight disease" comes out. Everything in between — the convolution filters, the pooled maps, the fully connected neurons — is invisible to us. This is the **black box** problem.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e0a008a3-61c4-42b9-926e-e2702bec252e">
</p>

> **Analogy:** It's like asking a friend to guess the ending of a movie just from its poster, and they get it right almost every time — but they can't explain *why*. Are they reading the actors' facial expressions? The colour scheme? The tagline? Without being able to see inside their reasoning, you only know that the guess works, not *how* it works. A CNN's black box is exactly this: correct answers with no visible reasoning.

---

## 3. What Is Feature Visualization?

**Feature Visualization** is the general name for techniques that make a CNN's learned features **explicit** — i.e., turn invisible internal numbers into something a human can actually look at.

$$
\boxed{\text{Feature Visualization for a unit} = \text{finding the input that maximizes that unit's activation}}
$$

In plain words: instead of asking "what did the network output for *this* image?", we ask the reverse question — **"what kind of image would make this specific part of the network fire as strongly as possible?"**

> **Analogy:** This is like being a detective who, instead of asking "what did the suspect do on this specific day?", asks "show me *every situation* that makes this suspect act a certain way, so I can figure out their pattern of behaviour." By hunting for the inputs that most excite a unit, we can infer what that unit is "looking for."

---

## 4. What Does "Unit" Mean?

Before diving into the techniques, we need to agree on what exactly we're trying to visualize. **"Unit"** is a general term that can refer to four different things inside a network, from the smallest possible piece up to the very final decision:

| Unit type | What it is | Rough size |
|---|---|---|
| **Individual neuron** | A single value at one specific (x, y) position in one feature map — the *atomic* unit of the network | Smallest |
| **Channel (feature map)** | One entire feature map — i.e., every neuron produced by one specific filter, across all spatial positions | Medium |
| **Layer** | An entire group of feature maps produced together at one depth of the network | Large |
| **Class probability neuron (softmax neuron)** | The final output neuron representing one specific class, after the softmax function | Final decision |

<p align="center">
  <img src="https://github.com/user-attachments/assets/d89b5a93-d3bc-4cd8-80d7-ecced406d964" width=600>
</p>

> **Analogy:** Imagine a big company. An **individual neuron** is like one single employee's opinion on one very specific micro-task. A **channel** is like an entire team that specializes in one skill (e.g. the "logo-detection team"). A **layer** is like an entire department made up of many teams (e.g. all of "Engineering," which includes several specialized teams). Finally, the **class probability neuron** is like the CEO's final, single sign-off decision — the one number that says "yes, this is a cat" or "no, it isn't."

---

## 5. Three Ways to Visualize What a CNN Learns

The slides group feature visualization techniques into **three broad families**:

| Family | Core idea |
|---|---|
| **1. Hidden Layer Output Visualization** | Feed real images through the trained network and simply **record** which images activate a given neuron the most |
| **2. Gradient-based Visualization** | Use **derivatives** (gradients) to trace, pixel by pixel, how much each part of the *input image* influenced a neuron's activation — includes **DeConvNet**, **Vanilla Gradient**, and **Guided Backpropagation** |
| **3. Feature Visualization by Optimization** | Start from **random noise** and use gradient descent to *synthesize a brand-new image from scratch* that maximally excites a chosen neuron |

<div align="left">

```text
Family 1: "Which of MY EXISTING images excites this neuron most?"
Family 2: "Which PIXELS of THIS image mattered for this neuron?"
Family 3: "What would a PERFECT, made-up image look like to this neuron?"
```

</div>

> **Analogy:** Imagine trying to figure out what makes your friend laugh. **Method 1** is like scrolling back through old text messages and picking out the ones where they laughed hardest — using existing evidence. **Method 2** is like taking one message they *did* laugh at and highlighting exactly which words made them laugh. **Method 3** is like sitting down and writing a brand-new joke from scratch, tweaking word by word, until you predict they'd laugh as hard as possible — even though that exact joke never existed before.

---

## 6. Method 1: Hidden Layer Output Visualization

This is the **simplest** visualization approach: directly visualize the hidden neurons using real images that already exist.

### 6.1 The Five Steps

1. Train a network
   
   <p align="center">
  <img src="https://github.com/user-attachments/assets/4c3cbb77-496a-44c1-b2c1-602b2ff78f79">
</p>

2. Feed images into the network
   
   <p align="center">
  <img src="https://github.com/user-attachments/assets/00df5bbe-d7a2-47c0-a878-0a7df8305f0c">
</p>

3. Observe the activation of some neuron
   <p align="center">
  <img src="https://github.com/user-attachments/assets/3d3db396-5d3b-4da8-9fa3-ef76ed338b0b">
</p>

4. Record images that maximally activate the neuron
   
<table align="center">
<tr>

<td align="center" valign="top">
  <div align="center">
    <img src="https://github.com/user-attachments/assets/e93b8f2c-fccb-4e04-b6d0-169e494d8a70">
  </div>
</td>

<td width="40"></td>

<td align="center" valign="top">
  <div align="center">
    <img src="https://github.com/user-attachments/assets/c9a63ea8-635e-4768-ba99-f5e487bc6f4e">
  </div>
</td>

</tr>
</table>

A grid of the top-activating image patches for neurons at **five different depths** of the network are shown. Reading this progression tells a very consistent story about how CNNs build understanding in stages:

| Layer | What the top-activating patches look like | How to interpret it |
|---|---|---|
| **Layer 1** | Small colored blocks, diagonal edges, colour gradients | The **earliest, simplest** features — just edges and colours, nothing object-like yet |
| **Layer 2** | Corners, circles, stripes, colour/edge combinations | Slightly more complex — combinations of Layer-1 features (an edge + a colour = a corner) |
| **Layer 3** | Repeating textures — mesh patterns, honeycomb grids, text-like patterns | **Textures and materials**, not yet whole objects |
| **Layer 4** | Dog faces, bird legs, wheels — clearly **class-specific** parts | Starting to represent **parts of objects**, tied to particular categories |
| **Layer 5** | Whole objects with varied poses — full dogs, keyboards, flowers | **Entire objects**, robust to changes in pose and viewpoint |

**How to read this kind of diagram in general:** always scan **left to right, shallow to deep**, and ask "did the patches get *more specific* or *more abstract*?" A healthy, well-trained CNN should show a clear progression from **generic** (edges/colours) → **structural** (textures/corners) → **semantic** (object parts) → **holistic** (whole objects). If a deep layer's patches still look like random edges, that's a sign something in training went wrong.

> **Analogy:** This progression is exactly like how a child learns to read. First they learn individual **letters** (Layer 1: edges, colours). Then they learn how letters combine into **syllables** (Layer 2: corners, simple shapes). Then they recognise **whole words** (Layer 3: textures, repeated patterns). Then they start recognising **specific meaningful phrases** (Layer 4: object parts like "dog face"). And finally they can read and understand **entire sentences** (Layer 5: whole objects) without having to sound out every single letter.

5. Analyse the similar pattern in those images

Concretely, for a convolution layer with **512 filters**, each filter produces its own **feature map** (e.g. 512 filters of size 14×14×3 each produce a 7×7×1 output map). We pick one specific neuron inside one specific feature map, run many images through the network, and simply **keep a record** of which images light that neuron up the most.

> **Analogy:** This is like a talent scout who doesn't ask "what specific quality am I looking for?" up front. Instead, they watch **hundreds of auditions** (feed in many images), note down the handful of performers who scored highest on one particular judge's scorecard (record images with the highest activation for one neuron), and only *afterwards* try to spot what those top performers have in common (analyse the pattern).

### 6.2 Disadvantage: Lack of Interpretability

The core weakness of this method: even after collecting the top-activating images, we **still can't be 100% sure** exactly *what* about those images the neuron cares about.

> **Example:** if a neuron fires strongly on nine different photos of dog faces, is it detecting the **dog's face** as a whole? Just the **eyes**? Just the **nose**? We only see the *whole* image that caused the activation — we don't automatically know *which part* of that image mattered.

> **Analogy:** It's like noticing that your smoke alarm keeps going off every time you fry bacon. Is it detecting the **smoke**? The **smell**? The **heat**? The **sizzling sound** (if it had a microphone)? Just knowing "bacon triggers it" doesn't tell you *which specific ingredient* of "frying bacon" the alarm is actually reacting to.

---

## 7. Method 2: Gradient-based Visualization — Overview

Instead of just collecting whole images that fire a neuron (Method 1), **gradient-based** methods dig **inside** one specific image and ask: *which exact pixels* pushed the activation up? This family includes three related techniques, covered in order from oldest to most refined:

| Technique | Core idea |
|---|---|
| **DeConvNet** | Build a **mirror-image network** that runs the CNN's operations in reverse, projecting an activation back down to pixel space |
| **Vanilla Gradient** | Directly compute the **mathematical derivative** of a neuron's score with respect to every input pixel |
| **Guided Backpropagation** | Same idea as Vanilla Gradient, but **suppresses negative gradients** while backpropagating through ReLU layers, for a cleaner result |

> **Analogy:** All three are trying to answer the same question — "which pixels of this photo mattered?" — but with progressively sharper tools. It's like trying to figure out which ingredient made a dish taste amazing: DeConvNet is like **literally reverse-cooking the dish** step by step to see what went in. Vanilla Gradient is like using a **very sensitive taste-meter** on every single ingredient. Guided Backpropagation is the same taste-meter, but it's been calibrated to **only report ingredients that made the dish taste better**, ignoring ones that made it taste worse.

---

## 8. DeConvNet (Deconvolutional Network)

### 8.1 What DeConvNet Does

**DeConvNet** provides a way to map an activation at a higher (deeper) or intermediate layer **back down to input-pixel space**, so we can see which pixels a given feature map "corresponds to." It performs the **same operations as a normal CNN, but in reverse**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/d7122351-ef16-4199-a805-74389110f65c" width=600>
</p>

| Forward CNN operation | DeConvNet's reverse operation |
|---|---|
| Convolution | Convolve with the **transposed filter** (filters copied straight from the trained CNN) |
| ReLU | Apply the **same** ReLU (copied from the CNN) |
| Max Pooling | **Unpool** using recorded "switches" (see below) |

Crucially, DeConvNet requires **no inference and no training** of its own — it simply reuses the already-trained CNN's filters, run backwards.

> **Analogy:** DeConvNet is like **playing a recorded video in reverse**. You're not creating new footage — you're taking the *exact same footage* (the trained filters) that was recorded going forward, and simply rewinding it, frame by frame, back to the very beginning (the input pixels).

### 8.2 Recap: How Convolution Works Forward

At layer $l$, an input image (or feature map) $y_1, \dots, y_{k_{l-1}}$ is convolved with a set of filters $f_{c,1}, \dots, f_{c,k_{l-1}}$ to produce feature maps $z_1, \dots, z_{k_l}$:

$$
\sum_{k=1}^{k_{l-1}} y_k * f_{c,k} = z_c \qquad c = 1, \dots, k_l
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/7c623116-783d-467a-970d-fb60ee54841c" width=1000>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/1832259b-4ec6-4220-8982-76076a5c5f4c" width=1000>
</p>

Each output feature map $z_c$ is the **sum** of every input channel convolved with its own filter.

### 8.3 Recap: How Pooling Works Forward

After convolution, **max pooling** compresses each feature map $z_c$ into a smaller **pooled map**, keeping only the strongest (maximum) activation within each pooling window and discarding the rest.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e2b2e5d1-8271-409b-99e5-8cb117afe88a">
</p>

> **Analogy for pooling:** Imagine summarising a long meeting transcript by keeping only the **single loudest statement** made in each 5-minute block, and throwing away everything quieter. You lose detail, but you keep the most important highlights — and the summary is much shorter and easier to work with, which is exactly why pooling shrinks feature maps.

### 8.4 Reversing Pooling: Switches and Unpooling

Regular max pooling **throws away** the exact position of the maximum value — after pooling, you only know "the loudest statement was somewhere in this block," not exactly where. To make pooling **reversible**, DeConvNet records **"switches"**: the exact (x, y) location of the maximum value in every pooling window, *at the time of the forward pass*.

During unpooling, the recorded activation value is placed back **only** at its original switch location; every other position in the unpooled map is set to **0**:

$$
\text{Unpooled map} = \begin{cases} z'_1 & \text{at the recorded switch location} \\ 0 & \text{everywhere else} \end{cases}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/264a66a7-9234-47a3-a73c-faea78ff6a74">
</p>

> **Analogy:** A switch is like a **bookmark**. When you highlight the single most important sentence on a page (max pooling), you also slip in a bookmark marking *exactly* which line it was (the switch). Later, when you want to put that sentence back into its original page (unpooling), the bookmark tells you precisely where it goes — everything else on the page stays blank because you never recorded what was there.

### 8.5 Reversing Convolution: The Deconvolution Step

Finally, the unpooled map is **"deconvolved"** — convolved with the **transposed** version of the original filters — to project the selected activations all the way back down into a reconstructed image:

$$
\sum_{k=1}^{k_l} z'_k * f_{k,c} = y'_c \qquad c = 1, \dots, k_{l-1}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e59c3e3-e502-4be8-a01b-eff23bbc4a49">
</p>

Repeating unpooling → ReLU → deconvolution layer by layer eventually produces a **reconstructed image**, showing exactly which pixel pattern in the original input caused that particular activation.

> **Analogy:** If convolution is like **blending** ingredients together into a smoothie (multiple pixels combined into one feature value), deconvolution is the (impossible in real life, but mathematically doable here) reverse process of **un-blending** — taking that one smoothie value and pouring it back out into an approximation of which original ingredients contributed to it.

### 8.6 Worked Example: What Each Layer's DeConvNet Visualization Reveals

A side-by-side pairs at each layer are shown as follow: the **DeConvNet reconstruction** (grey, abstract patches) next to the **actual image patches** that caused them. Here's how to read the progression:

| Layer | DeConvNet reconstruction shows | Real-world meaning |
|---|---|---|
| **Layer 1** | Simple diagonal/coloured strokes | Edges and colours |
| **Layer 2** | Circles, stripes, colour/edge conjunctions | Corners and simple textures |
| **Layer 3** | Repeating mesh and grid patterns | More complex textures/materials |
| **Layer 4** | Dog-face-like blob shapes, leg-like curved shapes | Class-specific object parts, with real variation between examples |
| **Layer 5** | Full object silhouettes with pose variation | Whole objects, robust to pose changes |

**How to read this kind of diagram in general:** compare the **abstract grey DeConvNet pattern** on the left to the **real image crop** on the right — the grey pattern is DeConvNet's best guess at "what pixel arrangement in the real crop caused this neuron to fire." The closer the grey pattern visually resembles the real crop (e.g. the same curved dog-face shape appearing in both), the more confidently we can say the neuron is really responding to that specific shape, rather than to something coincidental in the background.

> **Analogy:** It's like a sketch artist trying to draw a suspect's face purely from a witness's description of "what made them stand out" (the activation). Comparing the sketch (DeConvNet output) to an actual photo (the real image crop) tells you how accurately the "important features" were captured — a good sketch that closely matches the photo means the witness (the neuron) was really focused on the right details.

---

## 9. Vanilla Gradient (Saliency Maps)

### 9.1 Purpose

**Vanilla gradient** is the **original saliency map algorithm** for supervised deep learning. Its purpose: given an already-trained classification CNN and one specific image, figure out **which pixels were most important** for that image's high score on a particular class.

<p align="center">
  <img src="https://github.com/user-attachments/assets/90d346bf-5e0a-485b-913c-5868c9afedb1" width=1000>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/78d9d9b3-9fc0-4a71-986a-78a603541afd">
</p>

> **Example:** if we feed in an image of a cat and the network scores it highly for class "cat," we want to know **which pixels in that image** were responsible for pushing that "cat" score so high.

### 9.2 Steps for Computing a Saliency Map

<div align="left">
  
```text
      Step 1                 Step 2                     Step 3                   Step 4
Find the derivative    The derivative is an      For RGB images, take       Plot the resulting
of the class score     m × n matrix — take       the MAX of the 3 colour    matrix as an image
w.r.t. the image        the absolute value        channel derivatives        → that's your
                        of every element           at each pixel              saliency map
```

</div>

### 9.3 Worked Example: Reading a Saliency Map (the Cat Image)

A photo of a ginger kitten, alongside its saliency map rendered at increasing contrast are shown. Reading it step by step:

<p align="center">
  <img src="https://github.com/user-attachments/assets/dab543d6-2cc0-478c-a0ee-95708d9954bb" width=900>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b7718959-691b-4604-bcff-71de3618b5f0" width=600>
</p>
<p align="center">
  <em>Saliency map of a ginger kitten sitting on the grass</em>
</p>
- The **original photo** shows a whole kitten sitting on grass.
- The **saliency map** highlights certain pixels in **bright blue/white**, while the rest fades to near-black.
- The brightest regions cluster tightly around the kitten's **eyes**.

**How to read a saliency map in general:** brightness = importance. A pixel glowing brightly means "changing this pixel would have changed the class score a lot" — i.e., the network is paying close attention to it. A dark/black pixel means "this pixel barely matters" — the network would give roughly the same score even if that pixel changed. So when the eyes glow far brighter than the rest of the kitten's body, that tells us the network's "cat" decision leans heavily on the **eye region**, not the fur, background, or paws.

> **Analogy:** A saliency map is like shining a **UV blacklight** over a piece of paper to reveal hidden ink. Most of the page stays dark, but wherever there's hidden writing (the pixels the network actually used), it **glows**. The brighter the glow, the more that specific spot mattered to the final "message" (classification decision).

### 9.4 The Math Behind It (Chain Rule Walk-through)

The slides trace this with a tiny toy network: inputs $x_1, x_2, x_3$ feed into two intermediate neurons $s_1$ and $s_2$ (each passed through a ReLU), and the larger of $h_1, h_2$ becomes the final score $h_3$ (assume $h_2 > h_3$'s sibling, i.e. $h_1 < h_2$, so the `max` picks $h_2$).

Applying the chain rule step by step:

<p align="center">
  <img src="https://github.com/user-attachments/assets/e84059d3-2f1f-4ebd-8ecc-80ca0ab5648d" width=900>
</p>
<p align="center">
  <em>Detailed step-by-step derivation of partial derivatives using the chain rule across a neural network branch with shared weights and a max-pooling output.</em>
</p>

**How to read this:** the gradient of the final score with respect to any input pixel is just the **product of every "local slope"** along the path connecting that pixel to the output — exactly the ordinary backpropagation you already know from training a network, except here we stop at the **input image** instead of continuing on to update the weights.

> **Analogy:** This chain rule walk is like tracing **how much a single domino falling at the very start of the line affects the very last domino**. Each domino in between either passes the "push" along at full strength or (if it's already lying flat, i.e. the ReLU is off) **stops the chain entirely**. Multiplying all these individual pass-through effects together tells you exactly how much influence that one starting domino really had.

### 9.5 Disadvantage

Vanilla gradient saliency maps tend to be:
- **mostly zero away from the object**, with results often **not very visually satisfying** — the maps look noisy and speckled rather than cleanly outlining the object.
- **every pixel influencing the neuron through multiple different hidden neurons and paths**, and gradients along "negative" paths can cancel out or add noisy interference to gradients along "positive" paths.

> **Analogy:** It's like trying to hear one specific person's voice in a room where **everyone is talking at once, and some voices are actively arguing against the point you're trying to hear**. The signal you want is technically in there somewhere, but it comes out garbled and noisy rather than crisp and clean.

---

## 10. Guided Backpropagation

### 10.1 What It Does Differently

**Guided Backpropagation** visualizes gradients with respect to the image, exactly like Vanilla Gradient — **except negative gradients are suppressed (zeroed out) every time we backpropagate through a ReLU layer.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/4f017b8e-7cc7-477a-97c9-394f19332db5" width=600>
</p>

### 10.2 Motivation: Why Suppress Negative Gradients

- Neurons act as **detectors** of particular features in the image.
- We're only interested in what a neuron **does** detect — not in the things it actively *suppresses* or reacts negatively to.
- So, while propagating the gradient backward, we set **all negative gradients to 0**.
- We don't care if a pixel "suppresses" a neuron somewhere along a side-path — we only care about pixels that **positively** contribute.

Revisiting the toy network from Section 9.4, guided backprop **only follows the path** from $x$ to $h_3$ where **both** the weights **and** the neuron activations are positive (greater than 0) — any negative contribution along the way gets zeroed out.

<p align="center">
  <img src="https://github.com/user-attachments/assets/37114da1-ae13-406a-825a-af2eea5175f0" width=1000>
</p>

> **Analogy:** Guided backpropagation is like reading **only the five-star reviews** of a restaurant to figure out what people love about it, and completely ignoring the one-star reviews. You're not trying to understand *everything* that happened at the restaurant — you specifically want to know **what worked**, so you filter out anything negative and focus purely on the positive signal.

### 10.3 Worked Example: Vanilla Gradient vs Guided Backprop Comparison

Three images are placed side by side for comparison: the original kitten photo, its Vanilla Gradient map, and its Guided Backpropagation map.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1468d66e-2aa5-4c59-b071-bc08128900a1" width=600>
</p>

| Version | What it looks like | How to interpret it |
|---|---|---|
| **Vanilla gradient** | A fuzzy, noisy blob roughly in the shape of the kitten, with scattered speckles everywhere | Shows *some* signal on the cat, but it's cluttered — hard to say exactly which features mattered |
| **Guided backpropagation** | A much cleaner image where the kitten's **eyes, whiskers, and outline** are crisply visible, and the noisy background is almost gone | A far sharper, more human-readable picture of *exactly* which pixels drove the "cat" decision |

**How to read this kind of comparison in general:** when two saliency-style maps are shown side by side for the *same* image, judge them by **how cleanly the highlighted region matches a part of the object a human would agree is meaningful** (like eyes or edges), versus how much of the highlight is just scattered noise across the whole frame. The cleaner, more object-shaped result is doing a better job of genuinely isolating "what mattered."

> **Analogy:** If Vanilla Gradient is like listening to a noisy, crowded room and trying to make out one voice, Guided Backpropagation is like putting **noise-cancelling headphones** on that same room — background chatter (negative gradients) gets filtered out, and the one voice you actually care about (the positive-contributing pixels) suddenly comes through clearly.

---

## 11. Disadvantage Shared by Methods 1 & 2

Both **searching through training images** (Method 1) and **gradient-based visualization** (Method 2) share two important limitations:

1. **Correlation, not causation.** If the images that most activate a channel all happen to show a **dog next to a tennis ball**, we genuinely don't know whether the network is responding to the *dog*, the *tennis ball*, or *both together* — the elements in real photos are naturally correlated, and these methods can't separate them.
2. **You always need a real input image first.** There's no way to visualize what a feature "means" in the abstract — you can only observe it by feeding the network a specific, already-existing photo.

> **Analogy:** Imagine you notice that every time your friend seems happiest, they're at the beach **and** eating ice cream at the same time — because that's simply the only situation you've ever observed them in. Is it the beach that makes them happy? The ice cream? You can't tell, because in every real example you have, the two are tangled together. To truly test it, you'd need to somehow show them *just the ice cream* without the beach, or *just the beach* without the ice cream — which is exactly the idea behind Method 3, below.

---

## 12. Method 3: Feature Visualization by Optimization

### 12.1 The Core Idea

Feature visualization by optimization uses **gradient descent on the input pixels themselves** to *generate a brand-new image from scratch* that maximally excites a chosen neuron — rather than searching through existing photos (Method 1) or explaining an existing photo (Method 2).

$$
\boxed{\text{Optimization visualization} = \text{find pixels that} \; \textit{cause} \; \text{high activation, not pixels that merely} \; \textit{correlate} \; \text{with it}}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/61e5d3dc-d1db-408a-af43-d740150c0d5a" width=900>
</p>

> **Analogy:** This is like a sculptor who starts with a shapeless block of clay (random noise) and, guided by continuous feedback ("warmer / colder" — the gradient), slowly **reshapes the clay itself** step by step until it perfectly represents whatever concept they're chasing — rather than digging through a warehouse of finished statues hoping to find one that already resembles it (Method 1).

### 12.2 Steps of Optimization

Step 1: Start with a random noise image

<p align="center">
  <img src="https://github.com/user-attachments/assets/f67453bd-282d-428e-a403-7ee69e721831" width=600>
</p>

Step 2: Forward pass: Compute the activation a(x) at the chosen neuron

<p align="center">
  <img src="https://github.com/user-attachments/assets/c94cd74d-fee7-4a85-808e-362a7a392829" width=600>
</p>

Step 3: Backward pass: Backprop to find the gradient of the activation w.r.t x

<p align="center">
  <img src="https://github.com/user-attachments/assets/32226afa-e5a9-4315-a56c-f289f128ac2b" width=600>
</p>

Step 4: Nudge the image:  x ← x + α · (gradient) then repeat steps 2-4 until the image causes high activation

The key update rule is:

$$
x \leftarrow x + \alpha \cdot \frac{\delta a_i(x)}{\delta x}
$$

where $\alpha$ is a small step size and $\frac{\delta a_i(x)}{\delta x}$ tells us **how to change the colour of each pixel** to increase the activation of neuron $i$.

<p align="center">
  <img src="https://github.com/user-attachments/assets/37e048e2-beb5-45e6-b60f-179fc5df2432" width=600>
</p>

> **Analogy:** This is exactly like the children's game "hot and cold," played on a blank canvas instead of a room. You start by painting completely random colours everywhere (Step 1). Someone forward-passes your painting through the network and tells you a single number: how "hot" (activated) the target neuron currently is (Step 2). Backpropagation works out, for every single pixel, whether nudging it brighter or darker would make you "hotter" (Step 3). You make a **tiny** nudge to every pixel in the "hotter" direction (Step 4), then repeat the whole game thousands of times — slowly refining random static into a meaningful image.

### 12.3 Advantages of Optimization

**Advantage 1 — separates causation from correlation.** Because we're *generating* an image purely to maximize one neuron's activation (not borrowing it from a dataset), we can isolate what a neuron truly reacts to, distinct from things that just happened to co-occur in real photos.

<p align="center">
  <img src="https://github.com/user-attachments/assets/28788b8a-ca53-43f5-b702-fcd8744dcdea" width=600>
</p>

| Real dataset examples (top-activating photos) | What optimization reveals is *actually* driving activation |
|---|---|
| Baseballs, striped fabric | Not the baseball itself — the **stripes** |
| Dog and animal faces | Not the whole face — the **snout** shape |
| Clouds | Not "cloud" as a concept — the **fluffy texture** |
| Buildings | Not the building — the **sky** behind it |

> **Analogy:** This is like finally being able to run the controlled experiment from Section 11 — showing your friend a picture of **just ice cream, no beach**, and a picture of **just a beach, no ice cream**, to isolate exactly which one actually makes them happy, instead of only ever seeing them together.

**Advantage 2 — flexibility.** Optimization lets us ask flexible "what if" questions — e.g., "how would this exact photo need to change for one *additional* neuron to also activate?" — and it can be used to **watch how a feature evolves** over the course of training, something that's very hard to explore with only a fixed dataset of examples.

> **Analogy:** A fixed dataset is like a photo album — useful, but you can only look at moments that were already captured. Optimization is like having a **video game character creator** instead: you can freely tweak any "slider" you like and immediately see the result, without being limited to only the faces someone already photographed.

### 12.4 Optimization Variants

Different optimization *targets* reveal what different **parts** of the network are looking for:

<p align="center">
  <img src="https://github.com/user-attachments/assets/fe012303-0dd1-4412-873b-17a4bf39c23b" width=600>
</p>

| Target | What's being maximized | Typical visual result |
|---|---|---|
| **Neuron** | One value at one (x, y, channel) position | Small, localized pattern |
| **Channel** | An entire feature map (`layer[:,:,z]`) | A repeating texture pattern |
| **Layer / DeepDream** | An entire layer's activations, squared (`layer[:,:,:]²`) | Dreamlike, densely repeating imagery |
| **Class Logits** | The pre-softmax score for one class | Fragmented but recognisable class-related shapes |
| **Class Probability** | The post-softmax probability for one class | Sparser, more subtle patterns (since softmax competes classes against each other) |

> **Analogy:** Think of these five targets like **five different camera zoom levels** used to photograph the same office building. Zooming in on one **neuron** is like photographing a single doorknob. A **channel** is like photographing one whole room's décor style. A **layer** is like photographing an entire floor. **Class logits** is like photographing the building's overall silhouette from outside. **Class probability** is like photographing the building *while it's being compared against every neighbouring building on the block* — the result reflects not just "what does a bank look like" but "what makes this building look like a bank *and not* like the office tower next door."

---

## 13. The Enemy of Feature Visualization: Fooling Images

Merely optimizing an image to make a neuron fire, with **no other constraints**, doesn't reliably work. Instead, you can end up with high-confidence predictions for images that look like **pure noise** to a human — so-called **fooling images**.

### 13.1 Worked Example: Reading the Fooling-Image Grid

<p align="center">
  <img src="https://github.com/user-attachments/assets/2a6f5191-ce73-439d-91d4-015ffd8e628f" width=900>
</p>

A grid of images are shown the network labels "brambling," "redshank," "king penguin," "starfish," and so on, each with **over 99.6% confidence** — yet visually they range from static-like noise to abstract geometric wave patterns, with **nothing a human would recognise** as the labelled animal or object.

**How to read this kind of grid in general:** never trust a confidence score in isolation. A network reporting "99.6% cheetah" tells you the network's **internal math** strongly favours that class — it says nothing about whether the image would make sense to a human. When a high-confidence prediction is paired with an image that looks unrecognisable, that's a strong sign the optimization process found a **narrow mathematical shortcut** (some specific pixel pattern the network is oddly sensitive to) rather than anything resembling the real-world object.

> **Analogy:** This is exactly like a security guard whose face-recognition badge scanner is fooled by an oddly-shaped smear of face paint that isn't a face at all, but happens to trigger the same sensor pattern as an authorised employee's face. The scanner reports "99.6% match" — but any human glancing at the smear can immediately tell something is wrong. The system found a mathematical shortcut, not real understanding.

**Why this happens:** these images were optimized purely to maximally activate the target neuron, **without any natural image prior** (i.e., without any rule enforcing "this should look like a real photo"). Dealing with this high-frequency noise problem has been one of the central, ongoing challenges in feature visualization research.

---

## 14. Taming Fooling Images with Regularization

If you want *useful*, human-readable visualizations instead of noisy fooling images, you need to impose some kind of **natural structure** on the optimization using a **prior, regularizer, or constraint**.

| Regularization strength | Trade-off |
|---|---|
| **Weak regularization** | Avoids misleading correlations, but results are **less connected to realistic use** (closer to noisy/abstract patterns) |
| **Strong regularization** | Gives **more realistic-looking** examples, but risks re-introducing **misleading correlations** (since you're nudging the image toward looking "natural," you may accidentally bias it toward whatever the regularizer considers natural, not purely what the neuron wants) |

### 14.1 Worked Example: Reading the Regularization Trade-off Table

<p align="center">
  <img src="https://github.com/user-attachments/assets/2433f4a4-1c98-4449-8985-703f6a5c10f4" width=900>
</p>

A table of research approaches are presented (Erhan et al. 2009, Szegedy et al. 2013, Mahendran & Vedaldi 2015, and others), each checking off which regularization ingredients they used: **Frequency Penalization**, **Transformation Robustness**, **Learned Prior**, and **Dataset Examples**.

**How to read a table like this in general:** as you scan down the rows (roughly chronological order of research), notice how **later approaches increasingly combine multiple regularizers together** rather than relying on just one. This tells a story: no single regularization trick fully solves the fooling-image problem on its own — the field progressively found that **layering several complementary constraints** (e.g. penalizing high-frequency noise *and* requiring robustness to small transformations *and* blending in some dataset knowledge) produces visualizations that are both **interpretable** and **trustworthy**.

> **Analogy:** Think of regularization like the difference between asking someone to draw "anything that makes a judge say 'yes, that's a dog'" (weak/no regularization — they might draw a bizarre abstract pattern that technically fools the judge) versus asking them to draw "a *realistic-looking* dog that also makes the judge say yes" (strong regularization — closer to a real dog, but now influenced by what "realistic" means to *them*, which might not perfectly match what the judge originally cared about). Good regularization tries to strike a careful balance between the two extremes.

---

## 15. Key Takeaways

> [!IMPORTANT]
> **The core things to remember from Week 9:**

1. A trained CNN behaves like a **black box** — it gives correct answers, but the reasoning inside (feature extraction + classification) is invisible by default.
2. **Feature Visualization** makes learned features explicit by finding the input(s) that maximize a chosen **unit's** activation — where "unit" can mean a neuron, a channel, a layer, or a class probability neuron.
3. **Method 1 (Hidden Layer Output Visualization):** train a network, feed in many real images, and record which images maximally activate a neuron. Simple, but suffers from **lack of interpretability** — you can't tell exactly *which part* of the image mattered.
4. **Method 2 (Gradient-based Visualization)** digs into a single image and traces which **pixels** mattered, via three related techniques:
   - **DeConvNet** reverses convolution and pooling (using recorded "switches") to project an activation back to pixel space — no training needed, just the trained filters run backward.
   - **Vanilla Gradient** computes the mathematical derivative of a class score with respect to every pixel, producing a **saliency map** — but the result is often noisy.
   - **Guided Backpropagation** improves on Vanilla Gradient by **zeroing out negative gradients** during backprop through ReLU layers, producing a much cleaner, more human-readable map.
5. Both Method 1 and Method 2 suffer from the same fundamental limitation: they only reveal **correlation**, not **causation** — e.g. a "dog and tennis ball" example can't tell you which element the network actually cares about.
6. **Method 3 (Feature Visualization by Optimization)** starts from random noise and uses **gradient descent on the input pixels themselves** to synthesize a brand-new image that maximizes a neuron's activation, using the update rule $x \leftarrow x + \alpha \cdot \frac{\delta a_i(x)}{\delta x}$. This isolates true causes from mere correlations and offers great flexibility (e.g. watching features evolve during training).
7. Different **optimization targets** (neuron, channel, layer/DeepDream, class logits, class probability) reveal what different parts of the network are looking for.
8. Optimizing without any constraints produces **fooling images** — unrecognisable, noise-like images that the network labels with **over 99.6% confidence**. High confidence does **not** guarantee human-recognisable content.
9. **Regularization** (frequency penalization, transformation robustness, learned priors, dataset examples) is needed to keep optimized visualizations natural-looking, but there's an inherent trade-off: **weak regularization** avoids misleading correlations but looks less realistic; **strong regularization** looks more realistic but risks reintroducing misleading correlations.
10. Feature visualization will likely **never give a fully complete understanding** of a CNN, but combined with other tools, it's an important building block for interpretability. Open challenges include understanding how neurons **interact**, finding the **most significant units**, and getting a **holistic view** of all the facets of a feature.

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Black Box** | A trained model whose internal reasoning is hidden — only the input and final output are visible |
| **Feature Visualization** | Techniques for making a network's learned features explicit, typically by finding inputs that maximize a unit's activation |
| **Unit** | A general term for any of: an individual neuron, a channel (feature map), an entire layer, or a class probability (softmax) neuron |
| **Individual Neuron** | A single value at one specific position within one feature map — the atomic unit of a network |
| **Channel / Feature Map** | The complete output of one convolution filter across all spatial positions |
| **Hidden Layer Output Visualization** | Recording which real training images maximally activate a given neuron |
| **Lack of Interpretability** | The core weakness of Method 1: not knowing exactly *which part* of a top-activating image caused the activation |
| **DeConvNet (Deconvolutional Network)** | A network that reverses a CNN's convolution and pooling operations to project an activation back to pixel space |
| **Switches** | Recorded (x, y) positions of the maximum value in each max-pooling window, used to reverse pooling |
| **Unpooling** | The reverse of max pooling: placing a value back at its recorded switch location, with zeros everywhere else |
| **Deconvolution** | Convolving with a transposed filter to project a feature-map activation back toward the input |
| **Vanilla Gradient / Saliency Map** | A map of the derivative of a class score with respect to every input pixel, showing which pixels most influenced the score |
| **Guided Backpropagation** | Vanilla gradient computation where negative gradients are zeroed out when backpropagating through ReLU layers |
| **Correlation vs Causation (in feature visualization)** | The problem where top-activating images might contain multiple co-occurring elements, making it unclear which one truly caused the activation |
| **Feature Visualization by Optimization** | Using gradient descent directly on the input pixels, starting from random noise, to synthesize an image that maximizes a chosen unit's activation |
| **Optimization Variants** | Different optimization targets — neuron, channel, layer/DeepDream, class logits, class probability — each revealing a different part of the network |
| **DeepDream** | An optimization variant that maximizes an entire layer's (squared) activations, producing dreamlike imagery |
| **Class Logits** | The raw, pre-softmax score for a class |
| **Class Probability** | The post-softmax probability for a class, which competes against all other classes |
| **Fooling Image** | An unrecognisable, noise-like image that a network nonetheless classifies with very high confidence |
| **Natural Image Prior** | A constraint or assumption pushing an optimized image to look like a realistic photo, rather than pure noise |
| **Regularization (in feature visualization)** | Techniques (frequency penalization, transformation robustness, learned priors, dataset examples) used to keep optimized visualizations natural-looking and interpretable |

---

> [!TIP]
> **Study tip for Week 9:** Make sure you can:
> 1. Explain, in your own words, why a CNN is called a "black box," and what "feature visualization" is trying to solve.
> 2. Describe the difference between a **neuron**, a **channel**, a **layer**, and a **class probability neuron** as visualization targets.
> 3. Walk through the **five steps** of Hidden Layer Output Visualization, and explain its "lack of interpretability" weakness with your own example (not just dog face/eye/nose).
> 4. Explain how **DeConvNet** reverses convolution and pooling — in particular, what a "switch" is and why it's needed to reverse max pooling.
> 5. Trace through the **saliency map worked example** and explain, in your own words, how to read brightness in a saliency map.
> 6. Explain the key difference between **Vanilla Gradient** and **Guided Backpropagation**, and why suppressing negative gradients produces a cleaner result.
> 7. Explain why methods based on real images (Methods 1 & 2) can only show **correlation**, while optimization (Method 3) can isolate **causation** — use the "dog and tennis ball" example.
> 8. Walk through the **four steps of optimization** (Section 12.2) and be able to explain the update rule $x \leftarrow x + \alpha \cdot \frac{\delta a_i(x)}{\delta x}$ in plain English.
> 9. Explain what a **fooling image** is, why high confidence scores don't guarantee a recognisable image, and how regularization tries to fix this — including the weak-vs-strong regularization trade-off.
