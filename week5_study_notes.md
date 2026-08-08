# 📘 Week 5 Study Notes: Convolutional Neural Network (CNN)

> **Course:** COS30082 — Applied Machine Learning  
> **Topic:** Convolutional Neural Network (CNN)  
> **Prerequisite:** Week 4 Part 1 (Artificial Neural Networks), Week 4 Part 2 (Support Vector Machine)

---

## Table of Contents

1. [What Is a Convolutional Neural Network?](#1-what-is-a-convolutional-neural-network)
2. [ANN vs CNN: Why Not Just Use a Regular Neural Network?](#2-ann-vs-cnn-why-not-just-use-a-regular-neural-network)
3. [The CNN Architecture at a Glance](#3-the-cnn-architecture-at-a-glance)
4. [The Convolutional Layer](#4-the-convolutional-layer)
5. [Worked Example: The Convolution Operation](#5-worked-example-the-convolution-operation)
6. [Stride, Padding, and Output Size](#6-stride-padding-and-output-size)
7. [Multiple Filters → Multiple Feature Maps](#7-multiple-filters--multiple-feature-maps)
8. [Weight Sharing and Translation Invariance](#8-weight-sharing-and-translation-invariance)
9. [Sparsity of Connection](#9-sparsity-of-connection)
10. [Convolution vs Cross-Correlation](#10-convolution-vs-cross-correlation)
11. [Activation Functions](#11-activation-functions)
12. [The Pooling Layer](#12-the-pooling-layer)
13. [Worked Example: Max Pooling](#13-worked-example-max-pooling)
14. [The Fully Connected (FC) Layer](#14-the-fully-connected-fc-layer)
15. [Replacing FC Layers with Convolutional Layers](#15-replacing-fc-layers-with-convolutional-layers)
16. [Recap: The Softmax Layer and Softmax Regression](#16-recap-the-softmax-layer-and-softmax-regression)
17. [Forward and Backward Propagation in a Conv Layer](#17-forward-and-backward-propagation-in-a-conv-layer)
18. [Recap: Gradient Descent Variants](#18-recap-gradient-descent-variants)
19. [Advanced Optimizers: From Momentum to Adam](#19-advanced-optimizers-from-momentum-to-adam)
20. [Classic CNN Architectures](#20-classic-cnn-architectures)
21. [Batch Normalization](#21-batch-normalization)
22. [The Practical Deep Learning Design Process](#22-the-practical-deep-learning-design-process)
23. [Common Training Problems and How to Fix Them](#23-common-training-problems-and-how-to-fix-them)
24. [Debugging Strategies](#24-debugging-strategies)
25. [Key Takeaways](#25-key-takeaways)
26. [Glossary](#26-glossary)

---

## 1. What Is a Convolutional Neural Network?

### A Deep Learning Model Built for Grids of Data

A **Convolutional Neural Network (CNN or ConvNet)** is a deep learning model that was originally designed to work with **2‑dimensional image data**, although it can also be applied to 1‑dimensional data (e.g. audio, text) and 3‑dimensional data (e.g. video, volumetric scans).

$$
\boxed{\text{CNNs are mainly used for: image classification, segmentation, object detection, and other spatially/temporally correlated data}}
$$

> **Analogy:** Think of a CNN as a team of specialists scanning a photograph with a small magnifying glass. Each specialist only looks at a small patch at a time (not the whole photo at once), slides that magnifying glass across every part of the image, and reports back what pattern they found (an edge, a curve, a texture). Put enough of these specialists together in layers, and the network builds up from simple patterns (edges) to complex ones (a cat's face).

---

## 2. ANN vs CNN: Why Not Just Use a Regular Neural Network?

### Structural Difference

| | ANN (regular neural network) | CNN |
|---|---|---|
| **Neuron arrangement** | Neurons arranged in flat layers | Neurons arranged in **3 dimensions**: width, height, and depth |
| **Connections** | Every neuron connects to every neuron in the previous layer | Each neuron only connects to a **small local region** of the previous layer |
| **Parameters** | Each connection has its own independent weight | The **same filter (set of weights)** is reused across the whole image |

<div align="center">

```text
ANN with 2 hidden layers                CNN with 2 convolutional layers

  ○ ─┬─○ ─┬─○                              ▮ ──▶ [width x height x depth] ──▶ [width x height x depth] ──▶ ▭
  ○ ─┼─○ ─┼─○                                    (a 3-D block of neurons, not a flat layer)
  ○ ─┼─○ ─┼─○
  ○ ─┴─○ ─┴─○
```

</div>

> [!NOTE]
> An ANN "flattens" everything, so it loses the spatial relationship between neighbouring pixels. A CNN keeps that spatial structure alive by organising its layers as 3‑D volumes (width × height × depth/number of channels).

---

## 3. The CNN Architecture at a Glance

A typical CNN passes an image through two broad stages:

<div align="center">

```text
                 ┌────────────── Feature Learning ──────────────┐   ┌────── Classification ──────┐

Input Image ──▶ Convolution + ReLU ──▶ Pooling ──▶ Convolution + ReLU ──▶ Pooling ──▶ Fully Connected ──▶ Softmax ──▶ dog / cat / bird / ...
```

</div>

| Stage | Layers involved | Purpose |
|---|---|---|
| **Feature learning** | Convolution + ReLU, Pooling (repeated several times) | Detect increasingly complex visual features (edges → textures → shapes → objects) |
| **Classification** | Fully connected (FC) layers, Softmax | Combine the learned features and output a probability for each class |

> [!IMPORTANT]
> Each input image passes through a series of **convolution layers** (with filters/kernels), **pooling layers**, **fully connected layers**, and finally a **softmax** function that turns the output into probabilistic values between 0 and 1.

---

## 4. The Convolutional Layer

### Feature Detectors

The convolutional layer acts as a **feature detector**. It generates **feature maps** by convolving (sliding) filters across the input.

To compute pixel $i$ of the $r$-th feature map at layer $l$:

$$
x_i^{(l)} = g\left(\sum_{j=1}^{k} x_j^{(l-1)} * w_j^{(l-1)} + b_i^{(l-1)}\right)
$$

| Symbol | Meaning |
|--------|---------|
| $g(\cdot)$ | Activation function |
| $x^{(l-1)}$ | One input region (patch) from the previous layer |
| $w^{(l-1)}$ | The filter, of size $N \times N$ |
| $b_i$ | Bias term |
| $k$ | Total number of pixels in the filter |
| $*$ | Element‑wise multiplication (then summed) |

### How the Filter Slides

<div align="center">

```text
Input layer                     Filter w                Feature map
┌───┬───┬───┬───┬───┐                                    ┌───┬───┬───┐
│▓▓▓│   │   │   │   │      f( Σ (patch · filter) )        │▓▓▓│   │   │
├───┼───┼───┼───┼───┤     ─────────────────────▶          ├───┼───┼───┤
│   │   │   │   │   │                                     │   │   │   │
├───┼───┼───┼───┼───┤                                     ├───┼───┼───┤
│   │   │   │   │   │                                     │   │   │   │
└───┴───┴───┴───┴───┘                                     └───┴───┴───┘
```

</div>

The filter starts at the top‑left corner of the image, computes a weighted sum with the pixels it currently covers, applies the activation function, and writes the result into the corresponding cell of the feature map. It then **slides** across the image (left to right, top to bottom) and repeats the process until the whole image has been covered.

---

## 5. Worked Example: The Convolution Operation

### Setup

- Input feature map $x^{(l-1)}$: size $(d,d) = (5,5)$
- Filter $w^{(l-1)}$: size $(N,N) = (3,3)$
- Stride $s = 1$ (the filter moves 1 pixel at a time)
- Padding = none

### Output Size

$$
x^{(l)}\text{'s output size} = \frac{d-N}{s}+1 = \frac{5-3}{1}+1 = 3
$$

### The Numbers

$$
x^{(l-1)} =
\begin{bmatrix}
1 & 1 & 0 & 0 & 1\\
0 & 1 & 0 & 1 & 0\\
0 & 0 & 1 & 1 & 0\\
1 & 1 & 0 & 1 & 1\\
1 & 0 & 1 & 0 & 0
\end{bmatrix}
\qquad
w =
\begin{bmatrix}
0 & 1 & 1\\
0 & 1 & 1\\
0 & 0 & 1
\end{bmatrix}
\qquad\Longrightarrow\qquad
x^{(l)} =
\begin{bmatrix}
3 & 2 & 2\\
2 & 4 & 3\\
3 & 3 & 3
\end{bmatrix}
$$

To get the very first value (top-left, $=3$), the filter is placed over the top-left $3\times3$ patch of the input:

$$
\begin{bmatrix}1&1&0\\0&1&0\\0&0&1\end{bmatrix} * \begin{bmatrix}0&1&1\\0&1&1\\0&0&1\end{bmatrix}
= (1{\times}0)+(1{\times}1)+(0{\times}1)+(0{\times}0)+(1{\times}1)+(0{\times}1)+(0{\times}0)+(0{\times}0)+(1{\times}1) = 3
$$

The filter then slides one column to the right (stride = 1) and repeats, filling out the rest of the $3\times3$ feature map, one cell at a time, until it has swept the entire input.

> [!TIP]
> Don't try to memorise the numbers — focus on the *process*: multiply overlapping cells, sum them up, write one output number, then slide the window and repeat.

---

## 6. Stride, Padding, and Output Size

### The General Formula

$$
\boxed{\text{output size} = \frac{d - N + 2p}{s} + 1}
$$

where $d$ = input size, $N$ = filter size, $p$ = padding, $s$ = stride.

### Stride

**Stride** is the number of pixels the filter shifts over the input matrix at each step.

| Stride $s$ | Effect |
|---|---|
| $s=1$ | Filter moves 1 pixel at a time → larger, more detailed output |
| $s=2$ | Filter moves 2 pixels at a time → output is roughly half the size, computation is faster |

Example: with a $(5,5)$ input, a $(3,3)$ filter, and $s=2$:

$$
\text{output size} = \frac{5-3}{2}+1 = 2
$$

### Padding

**Padding** adds extra rows/columns of (usually zero-valued) pixels around the border of the input before convolving. This is useful because, without it, the feature map shrinks every time a convolution is applied, and pixels at the edges are used far less often than pixels in the centre.

Example: with padding $p=(1,1)$, a $(5,5)$ input effectively becomes $(7,7)$:

$$
\text{output size} = \frac{7-3}{1}+1 = 5
$$

<div align="center">

```text
No padding:            With padding = 1:
┌───┬───┬───┬───┬───┐   ┌───┬───┬───┬───┬───┬───┬───┐
│ x │ x │ x │ x │ x │   │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┤   ├───┼───┼───┼───┼───┼───┼───┤
│ x │ x │ x │ x │ x │   │ 0 │ x │ x │ x │ x │ x │ 0 │
├───┼───┼───┼───┼───┤   ├───┼───┼───┼───┼───┼───┼───┤
│ x │ x │ x │ x │ x │   │ 0 │ x │ x │ x │ x │ x │ 0 │
└───┴───┴───┴───┴───┘   └───┴───┴───┴───┴───┴───┴───┘
```

</div>

> [!NOTE]
> Bigger stride → smaller, faster output. Padding → keeps (or slows down the shrinking of) the output size and lets the filter "see" the edge pixels properly.

---

## 7. Multiple Filters → Multiple Feature Maps

A convolutional layer usually applies **many filters** to the same input, not just one. Using $N$ different filters produces $N$ different feature maps, which are then stacked together (concatenated) to form the output volume.

<div align="center">

```text
Input image (15,15,3)
        │
        │  convolve with 512 filters, each (3,3,3), stride 2
        ▼
512 separate feature maps, each (7,7,1)
        │
        │  concatenate along depth
        ▼
Output volume (7,7,512)
```

</div>

| Quantity | Value in the example |
|---|---|
| Number of filters | 512 |
| Size of each filter | $3\times3\times3$ |
| Size of each resulting feature map | $7\times7\times1$ |
| Depth of the concatenated output | 512 (one "slice" per filter) |

> [!IMPORTANT]
> The **depth** of a CNN layer's output is simply the number of filters used in that layer — each filter contributes exactly one feature map (channel) to the output volume.

---

## 8. Weight Sharing and Translation Invariance

### What Is Weight Sharing?

**Weight sharing** happens across the **receptive field** of the filters in a particular layer. The "weights" here are simply the numbers stored inside each filter.

- We are essentially trying to **learn one filter** that acts on a small receptive field / patch of the image.
- As the filter slides across the image, **the filter itself does not change**.
- The reasoning: if detecting an edge is useful in one part of the image, that same edge-detecting pattern is likely useful in other parts of the image too — so we might as well reuse the same filter everywhere.

$$
\boxed{\text{Weight sharing} \;\Longrightarrow\; \text{Translation invariance}}
$$

### Translation Invariance

**Translation invariance** means the network can recognise the same object regardless of *where* in the image it appears.

<div align="center">

```text
image size = 50 × 50 pixels

┌───────────┐  ┌───────────┐  ┌───────────┐
│           │  │           │  │        🐱  │
│        🐱  │  │      🐱   │  │            │
└───────────┘  └───────────┘  └───────────┘
    (cat detected regardless of its position in the frame)
```

</div>

> **Analogy:** Imagine a cookie cutter shaped like a star. It cuts the exact same star shape no matter where on the dough you press it. A CNN filter behaves the same way — it detects the exact same pattern (e.g. an edge) no matter where in the image it is applied.

---

## 9. Sparsity of Connection

**Sparsity of connection** means each element of the output depends only on a **small section** of the input — not the entire image.

| Layer depth | What one output element depends on |
|---|---|
| First conv layer, $3\times3$ filter | Only 9 numbers from the input |
| Deeper layers | Progressively **more** of the original image (indirectly), but still a limited, local region at each individual layer |

<div align="center">

```text
x⁽ˡ⁻¹⁾ (input)              w⁽ˡ⁻¹⁾ (filter)        x⁽ˡ⁾ (output)
┌───┬───┐                                            ┌─┐
│▓▓▓│   │  ──▶  *  ──▶ ┌───┬───┐ ──▶ = ──▶            │▓│
├───┼───┤              │   │   │                      └─┘
│   │   │              └───┴───┘
└───┴───┘        (only this small patch feeds into one output cell)
```

</div>

> [!NOTE]
> Because each output unit only "looks at" a small patch, sparsity of connection lets us train with **fewer parameters** and **less data**, and helps prevent overfitting compared to a fully connected layer covering the whole image.

---

## 10. Convolution vs Cross-Correlation

From a strict mathematical standpoint, "**convolution**" requires the filter to be **flipped 180°** in both directions before being applied. The operation described in Sections 4–5 (sliding the filter directly, without flipping) is technically called **cross-correlation**.

$$
w_j =
\begin{bmatrix}5&2&7\\9&4&1\\6&0&2\end{bmatrix}
\qquad\xrightarrow{\text{flip 180°}}\qquad
w_j =
\begin{bmatrix}2&1&7\\0&4&2\\6&9&5\end{bmatrix}
$$

### Why Does Math Require Flipping?

The flip exists so that convolution satisfies the **associative law**: $(AB)C = A(BC)$.

### Does It Matter in Deep Learning?

$$
\boxed{\text{No — in deep learning, whether you flip the kernel or not doesn't matter, as long as you're consistent}}
$$

The network is going to **learn** the filter's weights from data anyway, so it doesn't matter whether those learned weights represent a "flipped" or "unflipped" pattern. It only matters if you're trying to *interpret* the filter's weights directly (e.g. comparing them to a hand-designed filter from classic image processing).

> [!NOTE]
> In practice, deep learning frameworks (and this course) use the simpler cross-correlation operation and just call it "convolution" — this is standard practice across the field.

---

## 11. Activation Functions

### What They Do

An activation function $g(\cdot)$ maps the input signals of a neuron into output signals. Modern neural networks favour **non-linear** activation functions, because non-linearity is what allows a network to learn complex functions instead of just a straight-line relationship.

<div align="center">

```text
input ──▶ [ neuron: Σ(weights × inputs) ] ──▶ [ activation function g(z) ] ──▶ output
```

</div>

### Sigmoid and Tanh

$$
\text{Sigmoid: } g(z) = \frac{1}{1+e^{-z}} \qquad\qquad \text{Tanh: } g(z) = \tanh(z)
$$

| Function | Output range | Shape |
|---|---|---|
| Sigmoid | $(0,1)$ | S-shaped curve |
| Tanh | $(-1,1)$ | S-shaped curve, centred at 0 |

### Limitations of Sigmoid and Tanh

| Problem | Explanation |
|---|---|
| **Saturation / limited sensitivity** | Large inputs "snap" to 1.0 (or -1); small inputs snap to 0 (or -1 for tanh). The functions are only really sensitive around their mid-point (0.5 for sigmoid, 0.0 for tanh) |
| **Vanishing gradient** | In a feedforward network, the backpropagated error tends to shrink exponentially the further it travels from the output layer, which can make the network effectively stop learning in its earlier layers |

> [!WARNING]
> A saturated activation function has a derivative close to zero. Since backpropagation multiplies gradients together layer by layer, many near-zero derivatives multiplied together quickly shrink toward zero — this is the root cause of the vanishing gradient problem (see Section 23).

### ReLU (Rectified Linear Unit)

$$
f(x) = \max(0, x)
$$

> *"Because rectified linear units are nearly linear, they preserve many of the properties that make linear models easy to optimize with gradient-based methods. They also preserve many of the properties that make linear models generalize well."* — **Deep Learning**, 2016, p.175

ReLU became the default choice in most modern CNNs because it is cheap to compute and does **not saturate** for positive inputs, which greatly reduces the vanishing gradient problem.

### Other Common Activation Functions

| Name | Formula | Notes |
|---|---|---|
| Identity | $f(x)=x$ | No non-linearity at all |
| Binary step | $f(x)=0$ if $x<0$, else $1$ | Not differentiable at $x=0$ |
| Logistic (Sigmoid) | $f(x)=\dfrac{1}{1+e^{-x}}$ | Output $(0,1)$ |
| Tanh | $f(x)=\tanh(x)$ | Output $(-1,1)$ |
| ArcTan | $f(x)=\tan^{-1}(x)$ | Similar shape to tanh, wider range |
| ReLU | $f(x)=\max(0,x)$ | Most widely used in CNNs |
| Parametric ReLU (PReLU) | $\alpha x$ if $x<0$, else $x$ | $\alpha$ is a *learnable* parameter |
| Exponential Linear Unit (ELU) | $\alpha(e^x-1)$ if $x<0$, else $x$ | Smooths the negative side |
| SoftPlus | $f(x)=\log_e(1+e^x)$ | A smooth approximation of ReLU |

---

## 12. The Pooling Layer

### Purpose

- **Feature selection** — keeps the strongest/most representative signal
- **Improves generalisation**
- Helps make the representation **approximately invariant** to small translations of the input

### Pooling Methods

| Method | What it keeps |
|---|---|
| **Max pooling** | The largest value in each window |
| **Min pooling** | The smallest value in each window |
| **Average pooling** | The average value in each window |

$$
x_i^{(l)} = \max\left(x^{(l-1)}\right)
$$

where $x_i^{(l)}$ is one output pixel, and $x^{(l-1)}$ is one input window region of the $r$-th feature map.

<div align="center">

```text
Feature map                     Pooled feature map

┌───┬───┬───┬───┐               ┌───┬───┐
│▓▓▓│▓▓▓│   │   │   max( )      │▓▓▓│   │
├───┼───┼───┼───┤  ────────▶    ├───┼───┤
│▓▓▓│▓▓▓│   │   │               │   │   │
├───┼───┼───┼───┤               └───┴───┘
│   │   │   │   │
└───┴───┴───┴───┘
```

</div>

---

## 13. Worked Example: Max Pooling

### Setup

- Input feature map $x^{(l-1)}$: size $(5,5)$
- Pool size $(N,N) = (3,3)$
- Stride $s=1$, no padding

### Output Size

$$
x^{(l)}\text{'s output size} = \frac{d-N}{s}+1 = \frac{5-3}{1}+1 = 3
$$

### The Numbers

$$
x^{(l-1)}=
\begin{bmatrix}
6&8&0&0&1\\
4&7&5&9&8\\
0&0&6&4&0\\
1&2&1&1&1\\
1&0&1&5&0
\end{bmatrix}
\qquad\xrightarrow{\max(\;)}\qquad
x^{(l)}=
\begin{bmatrix}
8&9&9\\
7&9&9\\
6&6&6
\end{bmatrix}
$$

To find the top-left output value ($=8$), take the maximum of the top-left $3\times3$ patch:

$$
\max\begin{pmatrix}6&8&0\\4&7&5\\0&0&6\end{pmatrix} = 8
$$

The window then slides one step to the right (and eventually down), taking the max of each new patch, until the full $3\times3$ output has been produced.

### Effect of Padding and Stride (same rules as convolution)

| Setting | Effect on a $(5,5)$ input with a $(3,3)$ pool |
|---|---|
| Padding = (1,1) → input effectively $(7,7)$ | output size $= \frac{7-3}{1}+1 = 5$ |
| Stride = 2 | output size $= \frac{5-3}{2}+1 = 2$ |

---

## 14. The Fully Connected (FC) Layer

### FC Layer vs Convolutional Layer

| | Fully Connected (FC) Layer | Convolutional Layer |
|---|---|---|
| **Connections** | Every neuron connects to **every** neuron in the previous layer | Every neuron connects to only **a few nearby (local)** neurons |
| **Weights** | Each connection has its **own independent** weight | The **same set of weights** (filter) is reused for every neuron (weight sharing) |

<div align="center">

```text
Feature map from final layer
      ┌───┬───┐
      │   │   │   Flatten        Fully       Fully       Fully
      ├───┼───┤  ─────────▶   Connected ─▶ Connected ─▶ Connected ─▶ softmax ─▶ class probabilities
      │   │   │
      └───┴───┘
```

</div>

The feature map produced by the final convolution/pooling layer is **flattened** into a 1-D vector, which then becomes the input to one or more fully connected layers — essentially a standard Multi-Layer Perceptron (MLP) — that combines all the learned features together before making the final prediction.

---

## 15. Replacing FC Layers with Convolutional Layers

### Why Bother?

$$
\boxed{\text{Replacing an FC layer with a convolutional layer reduces the number of parameters, because convolutional weights are shared}}
$$

This produces **faster and more robust learning**.

### A Concrete Comparison

<div align="center">

```text
Traditional approach (Flatten + FC):
(14,14,386) ─conv─▶ (7,7,386) ─pool─▶ flatten (1,18914) ─FC(18914,4096)─▶ (1,4096) ─FC(4096,1000)─▶ (1,1000) ─▶ softmax

                                        ≈ 81.5 million parameters in the FC layers
```

```text
Convolution-only approach (no flatten needed):
(14,14,386) ─conv─▶ (7,7,386) ─pool─▶ (7,7,386) ─conv(7×7)─▶ (1,1,386) ─conv(1×1)─▶ (1,1,4096) ─conv(1×1)─▶ (1,1,1000) ─▶ softmax

                                        ≈ 13 million parameters
```

</div>

By replacing the flatten + FC combo with convolutions that use filter sizes matching the remaining spatial dimensions (e.g. a $7\times7$ filter, then $1\times1$ filters), the same transformation can be achieved with roughly **6× fewer parameters** (13M vs 81.5M in this example) — because the weights are shared instead of each connection getting its own independent weight.

---

## 16. Recap: The Softmax Layer and Softmax Regression

### The Softmax Function

The **softmax layer** is typically the final output layer in a network performing **multi-class classification** (e.g. object recognition). It takes a vector of $k$ raw output scores and converts them into a probability distribution:

$$
h_\theta^{(c)}(x) = \text{softmax}(z_c) = \frac{e^{z_c}}{\sum_{i=1}^{k}e^{z_i}} \qquad \text{where } 1\leq i\leq k
$$

Every resulting probability lies in the range $(0,1)$, and all of them **sum to 1**.

### Softmax Regression = Softmax Activation + Cross-Entropy Loss

The cost of one training sample, summed across all $k$ output classes:

$$
cost\big(h_\theta(x),y\big) = -\sum_{c=1}^{k}\mathbb{1}\{y=c\}\log h_\theta^{(c)}(x) = -\sum_{c=1}^{k}\mathbb{1}\{y=c\}\log\frac{e^{z_c}}{\sum_{i=1}^{k}e^{z_i}}
$$

| Component | Role |
|---|---|
| **Softmax (activation)** | Outputs a probability for each class; all probabilities sum to 1 |
| **Cross-entropy (loss)** | The sum of the negative logarithm of the *correct* class's predicted probability |

> [!NOTE]
> "Softmax regression" and "multinomial logistic regression" are two names for the exact same thing: a softmax activation paired with a cross-entropy loss.

---

## 17. Forward and Backward Propagation in a Conv Layer

### The General Idea (Chain Rule)

<div align="center">

```text
Forward pass:                          Backward pass (chain rule):

  x ──┐                                        δJ/δx = (δJ/δz)(δz/δx)
      ├──▶ [ f(w,x) ] ──▶ z                       ▲
  w ──┘                                     [ df ] │ ◀── δJ/δz
                                                    ▼
                                            δJ/δw = (δJ/δz)(δz/δw)
```

</div>

### Forward Pass Example

For a $3\times3$ input convolved with a $2\times2$ filter (stride 1, no padding), the output is $2\times2$:

$$
z_{11}=w_{11}x_{11}+w_{12}x_{12}+w_{21}x_{21}+w_{22}x_{22}
$$
$$
z_{12}=w_{11}x_{12}+w_{12}x_{13}+w_{21}x_{22}+w_{22}x_{23}
$$
$$
z_{21}=w_{11}x_{21}+w_{12}x_{22}+w_{21}x_{31}+w_{22}x_{32}
$$
$$
z_{22}=w_{11}x_{22}+w_{12}x_{23}+w_{21}x_{32}+w_{22}x_{33}
$$

Notice how the **same four weights** ($w_{11},w_{12},w_{21},w_{22}$) show up again and again — that's weight sharing in action.

### Backward Pass Example

Because each weight in the filter contributes to *every* output pixel, a change in one weight affects **all** output pixels — so the gradient for each weight must add up the contributions from every pixel it touched:

$$
\delta w_{11} = x_{11}\delta z_{11} + x_{12}\delta z_{12} + x_{21}\delta z_{21} + x_{22}\delta z_{22}
$$
$$
\delta w_{12} = x_{12}\delta z_{11} + x_{13}\delta z_{12} + x_{22}\delta z_{21} + x_{23}\delta z_{22}
$$
$$
\delta w_{21} = x_{21}\delta z_{11} + x_{22}\delta z_{12} + x_{31}\delta z_{21} + x_{32}\delta z_{22}
$$
$$
\delta w_{22} = x_{22}\delta z_{11} + x_{23}\delta z_{12} + x_{32}\delta z_{21} + x_{33}\delta z_{22}
$$

where $\delta w_{ij}$ represents $\dfrac{\delta J}{\delta w_{ij}}$ and $\delta z_{ij}$ represents $\dfrac{\delta J}{\delta z_{ij}}$.

> [!IMPORTANT]
> During backpropagation, **all** error terms contribute to each weight's gradient update, as a weighted sum with the input pixels that were affected by that weight. This is the direct mathematical consequence of weight sharing.

---

## 18. Recap: Gradient Descent Variants

### The Standard Update Rule

$$
w_t \leftarrow w_{t-1} - \propto\left[\frac{\delta J}{\delta w}\right]_{w_{t-1}} \qquad\text{where } t=\text{new},\ t-1=\text{old}
$$

| Variant | How much data is used per update |
|---|---|
| **Batch Gradient Descent** | The **entire** training set |
| **Stochastic Gradient Descent (SGD)** | **One** training sample at a time |
| **Mini-batch Gradient Descent** | A **small batch** of samples at a time |

<div align="center">

```text
Batch GD:        long, smooth, but slow steps toward the minimum
Stochastic GD:    noisy, erratic steps, but fast per-step updates
Mini-batch GD:    a practical middle ground between the two
```

</div>

### Why Traditional Gradient Descent Struggles in Deep Networks

- The loss functions of logistic regression, linear regression, and SVM are **convex** — they contain only a minimum (or maximum), no saddle points.
- The loss function of a neural network is **complex and non-convex**, so it can contain many **saddle points** and **local minima**.
- With plain SGD, if training gets stuck at a saddle point or a local minimum, it is very hard to escape, because **the gradient term becomes (close to) zero** there.

> [!WARNING]
> This is exactly why more advanced optimizers (Section 19) were developed — they add extra mechanisms to keep the parameters moving even when the raw gradient is tiny.

---

## 19. Advanced Optimizers: From Momentum to Adam

### Momentum

Momentum accelerates gradient descent on surfaces that curve more steeply in one direction than another, and it **dampens oscillation**.

$$
v_t = \gamma v_{t-1} + \propto\left[\frac{\delta J}{\delta w}\right]_{w_{t-1}} \qquad\qquad w_t = w_{t-1}-v_t
$$

It combines the **current** gradient with an exponentially weighted average of **past** gradients ("momentum" — literally like a ball rolling downhill and building up speed).

<div align="center">

```text
Batch GD without momentum:        Batch GD with momentum:
   zig-zag path to the minimum       smoother, more direct path
```

</div>

### Nesterov Accelerated Gradient (NAG)

NAG is very similar to momentum, but it evaluates the gradient **after** the current velocity (momentum step) has already been applied — i.e. it "looks ahead" first.

$$
v_t = \gamma v_{t-1} + \propto\left[\frac{\delta J}{\delta w}\right]_{w_{t-1}-\gamma v_{t-1}} \qquad\qquad w_t=w_{t-1}-v_t
$$

$w_{t-1}-\gamma v_{t-1}$ is the "looked-ahead" position. NAG evaluates the gradient there, and updates the weights based on how significant that looked-ahead gradient is.

### Adagrad

Adagrad is an **adaptive learning rate** method: it makes **larger** updates for infrequent parameters (small past gradients) and **smaller** updates for frequent parameters (large past gradients).

$$
w_t = w_{t-1} - \frac{\propto}{\sqrt{G_t+\varepsilon}}\left[\frac{\delta J}{\delta w}\right]_{w_{t-1}}
$$

where $G_t$ is the sum of squares of past gradients for all parameters $w$, and $\varepsilon$ is a tiny positive number that prevents division by zero.

- **Benefit:** rare but highly predictive features stand out and get bigger updates.
- **Drawback:** $G_t$ only ever grows, so the effective learning rate keeps **shrinking** and can eventually become too small.

### Adadelta

Adadelta extends Adagrad to fix its aggressive, ever-shrinking learning rate. Instead of storing the full sum $G_t$, it keeps a **decaying average** of past squared gradients:

$$
eda_t = \gamma\, eda_{t-1} + (1-\gamma)\left[\frac{\delta J}{\delta w}\right]^2_{w_{t-1}}
\qquad\qquad
w_t = w_{t-1} - \frac{\propto}{\sqrt{eda_t+\varepsilon}}\left[\frac{\delta J}{\delta w}\right]_{w_{t-1}}
$$

### RMSProp

RMSProp solves the **same** problem as Adadelta (Adagrad's shrinking learning rate), developed independently around the same time.

$$
\boxed{\text{RMSProp} = \text{Adadelta with } \gamma \text{ fixed at } 0.95}
$$

### Adam (Adaptive Moment Estimation)

Adam is one of the most **popular** optimizers used today. It keeps track of exponential moving averages of both the gradient ($m_t$, the "first moment") and the squared gradient ($v_t$, the "second moment").

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)\left[\frac{\delta J}{\delta w}\right]_{w_{t-1}}
\qquad\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)\left[\frac{\delta J}{\delta w}\right]^2_{w_{t-1}}
$$

Because $m_t$ and $v_t$ start at 0, they are initially **biased toward 0** — this is corrected with:

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t} \qquad\qquad \hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

Finally:

$$
w_t = w_{t-1} - \frac{\propto}{\sqrt{\hat{v}_t}+\varepsilon}\;\hat{m}_t
$$

| Optimizer | Key idea |
|---|---|
| Momentum | Add a fraction of the previous update to smooth the path |
| NAG | Like momentum, but the gradient is measured after a "look ahead" step |
| Adagrad | Adapts the learning rate per-parameter, but shrinks it forever |
| Adadelta / RMSProp | Adapts the learning rate per-parameter using a *decaying* average (fixes Adagrad's shrinkage) |
| Adam | Combines momentum-style averaging **and** adaptive per-parameter learning rates, with bias correction |

---

## 20. Classic CNN Architectures

### LeNet-5 (1990s)

Proposed by **Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner**. Designed for **handwritten and machine-printed character recognition**.

| Layer | Feature maps | Size | Kernel | Stride | Activation |
|---|---|---|---|---|---|
| Input (image) | 1 | 32×32 | – | – | – |
| Conv 1 | 6 | 28×28 | 5×5 | 1 | tanh |
| Avg Pool 2 | 6 | 14×14 | 2×2 | 2 | tanh |
| Conv 3 | 16 | 10×10 | 5×5 | 1 | tanh |
| Avg Pool 4 | 16 | 5×5 | 2×2 | 2 | tanh |
| Conv 5 | 120 | 1×1 | 5×5 | 1 | tanh |
| FC 6 | – | 84 | – | – | tanh |
| Output FC | – | 10 | – | – | softmax |

### AlexNet

Very similar architecture to LeNet, but **deeper and bigger**, and it stacked multiple convolutional layers on top of each other (previously it was common to only have a single conv layer immediately followed by a pooling layer).

**Special features and practices:**

| Feature | Benefit |
|---|---|
| **ReLU** instead of tanh/sigmoid | Faster training |
| **Multiple GPUs** | Allows a bigger model, faster training |
| **Overlapping pooling** | Improves accuracy |
| **Data augmentation** | Reduces overfitting |
| **Dropout** | A regularization technique that reduces overfitting |

**Dropout**, in more detail:

Individual nodes are either **dropped out** of the network with probability $1-p$, or **retained** with probability $p$. All of that node's inbound and outbound connections are removed for that training step.

<div align="center">

```text
Standard network:                Network with dropout:

  ○───○───○                        ○   ⊗───○
  ○───○───○                        ⊗   ○───○
  ○───○───○                        ○───○   ⊗
```

</div>

### VGG

VGG improves on AlexNet, and its main contribution was showing that **network depth is essential for good performance**.

**Special features:**

- Replaces AlexNet's large kernels (11×11 and 5×5 in the first two conv layers) with **multiple stacked 3×3 kernels** applied one after another.
- Introduced the idea of **blocks/modules** — applying the same filter size repeatedly to extract progressively more complex features. This concept became a common theme in the networks that followed.
- **Limitation:** VGG has a huge number of parameters (~140 million), mostly located in its **first fully connected layer**.

| VGG configuration | Weight layers | Structure |
|---|---|---|
| A | 11 | 8 conv + 3 FC layers |
| E | 19 | 16 conv + 3 FC layers |
| **D (VGG-16, best-known)** | **16** | **13 conv + 3 FC layers** |

### GoogLeNet (Inception)

GoogLeNet's main contribution was the **Inception layer**, which dramatically reduced the number of parameters compared to AlexNet (≈4M vs ≈60M).

**The Inception layer** combines several operations in parallel and concatenates their outputs into one volume:

<div align="center">

```text
                     ┌────────────── Filter concatenation ──────────────┐
                     │            │             │              │
                1×1 conv     3×3 conv       5×5 conv        1×1 conv
                     │            │             │              │
                     │        1×1 conv      1×1 conv       3×3 max pool
                     │            │             │              │
                     └────────────┴── Previous layer ───────────┘
```

</div>

| Component | Purpose |
|---|---|
| 1×1, 3×3, 5×5 convolutions (in parallel) | Let the network capture patterns at multiple scales simultaneously |
| 1×1 convolution *before* the 3×3/5×5 branches | **Dimensionality reduction** — cuts down the number of channels cheaply before the more expensive convolutions |
| Parallel 3×3 max pooling branch | Gives the layer another option besides convolution |

**Intuition:** the Inception layer lets the network **pick and choose** which filter size is most relevant for the information it needs to learn — a small object might need a small filter, while a large or blurry object might need a bigger one, and the network doesn't have to commit to just one choice.

### ResNet (Residual Network)

**Problems with very deep networks:**

| Problem | Description |
|---|---|
| **Vanishing gradient** | Earlier layers get neglected because gradients shrink as they backpropagate through many layers |
| **Curse of dimensionality / degradation problem** | Shallower networks sometimes learn *better* than their deeper counterparts, simply because depth alone can hurt optimisation |

**Solution — skip connections (identity shortcut connections):** developed by **Kaiming He et al.**

$$
R(x) = f(x) - x
$$

<div align="center">

```text
A regular block:                    A single residual block:

     Activation                          Activation
         ▲                                ▲      ▲
       f(x)                            f(x)      │
         ▲                                ▲       + ◀── x (skip connection)
    Weight layer                     Weight layer  │
         ▲                                ▲        │
    Activation                       Activation     │
         ▲                                ▲          │
    Weight layer                     Weight layer     │
         ▲                                ▲            │
         x                                x ───────────┘
```

</div>

If the ideal mapping for a block is simply the **identity** ($f(x)=x$), it turns out to be much easier for the network to learn the **residual** $R(x)=f(x)-x$ (i.e. push the block's weights toward zero) than to directly learn the identity function through ordinary stacked layers. The skip connection carries $x$ straight through, so the network only has to learn the "correction" on top of it.

---

## 21. Batch Normalization

### The Problem: Internal Covariate Shift

As training progresses, the **distribution of each layer's inputs keeps changing**, because the parameters of every earlier layer are also being updated. This is called **internal covariate shift**, and it can slow down training considerably.

<div align="center">

```text
Changes in earlier weights ──▶ hidden unit values keep shifting ──▶ this imbalance cascades through the whole network
```

</div>

### The Fix: Batch Normalization (BN)

**Batch normalization** is applied right after each convolution and **before** the activation function.

<div align="center">

```text
... ──▶ 3×3 conv ──▶ Batch norm ──▶ ReLU ──▶ 3×3 conv ──▶ Batch norm ──▶ (+ skip) ──▶ ReLU ──▶ ...
```

</div>

**The BN process, step by step:**

1. Normalize the hidden unit values using the **mean** and **variance** of the current mini-batch:
$$
z_{norm} = \frac{z-\boldsymbol{\mu}}{\boldsymbol{\sigma}}
$$
2. Multiply the normalized output by a learnable scale parameter $g$:
$$
z_{norm} * g
$$
3. Add a learnable shift parameter $b$:
$$
(z_{norm} * g) + b
$$

$\boldsymbol{\mu}$, $\boldsymbol{\sigma}$, $g$, and $b$ are all **trainable** — they are optimised throughout training just like any other weight.

### Benefits of Batch Normalization

| Benefit | Explanation |
|---|---|
| Prevents imbalance | Stops the weights in the network from becoming too high or too low |
| Speeds up training | Each layer's inputs stay in a more stable, predictable distribution |
| Slight regularization effect | Because normalization is computed per mini-batch, it adds a small amount of noise, which slightly **reduces overfitting** |

---

## 22. The Practical Deep Learning Design Process

### The 4-Step Recommended Process

1. **Determine your goals** — Is this a classification problem or a regression problem? This decides the loss function and the target label.
2. **Establish a working end-to-end pipeline** — Pick appropriate performance metrics (precision, recall, cosine similarity, etc.) and deploy a baseline model for the first experiment.
3. **Determine the bottlenecks in performance** — Is the model overfitting? Underfitting? Is there a defect in the data or the software?
4. **Repeatedly make changes to improve the system** — Increase dataset size, tune hyperparameters, try different algorithms.

### Case Study: Automatic Car Park Payment System

**Scenario:** An automatic parking system needs to authenticate cars via their **license plate numbers**, so an automated registration system is needed.

**Step 1 — Determine your goals:**

This is actually **two problems in one**: a **regression** problem (where on the image is the license plate located?) and a **classification** problem (what are the characters on the plate?).

**Step 2 — Establishing the pipeline: performance metrics**

| Task | Suitable metrics |
|---|---|
| Multi-class character classification (A–Z, 0–9) | Top-1 / Top-5 accuracy; Mean Average Precision (mAP) — average precision per class, then averaged across all classes |
| Object detection (locating the license plate) | Intersection over Union (IoU); Precision and Recall; mAP |

**Step 2 — Establishing the pipeline: baseline models**

| Question | Guidance |
|---|---|
| How complex is the problem? | This is an "AI-complete" style problem → a deep learning model is recommended over classical approaches |
| What is the structure of the input? | Images → a **CNN** is recommended (as opposed to a plain FC network for fixed-size vectors, or an RNN for sequential data) |
| Which CNN family for **object detection**? | R-CNN, Fast R-CNN, Faster R-CNN, YOLO (covered in a future lecture) |
| Which CNN family for **character classification**? | AlexNet or VGG — pick based on how complex your data is |

---

## 23. Common Training Problems and How to Fix Them

### Overfitting

| | Description |
|---|---|
| **Definition** | The model fits the training data well, but does not generalise to new, unseen test data |
| **Symptom** | Low bias, high variance |
| **Common causes** | Model is too complex; training data set is too small |

**Solutions:**

- Gather **more training data**
- **Dataset augmentation**: translation, flip, mirror, noise, colour augmentation, etc.
- **Reduce model complexity**: fewer neurons, fewer layers, fewer weight parameters
- **Regularization**: L1/L2 weight decay, dropout, early stopping
- **Early stopping**: train for an arbitrarily large number of epochs, and stop as soon as performance on a held-out validation set stops improving

### Underfitting

| | Description |
|---|---|
| **Definition** | The model performs poorly on **both** the training and testing data |
| **Symptom** | High bias, low variance |
| **Common cause** | Model is too simple |

**Solutions:**

- **Increase model complexity**: add layers, add neurons
- **Increase training time**
- **Reduce dropout**

### Gradient Exploding

**Cause (intuition):** if a layer's weight $w^{(l)} > 1$, then as the signal passes through many layers, $z^{(l)}$ (and therefore the output $\hat y$) grows **exponentially**. During backpropagation, the gradient $\dfrac{\delta J(w)}{\delta w^{(l)}}$ then also grows **exponentially**, because each layer's error term is a weighted sum that keeps multiplying by these large weights:

$$
\delta^{(l)} = (w^{(l)})^{T}\,\delta^{(l+1)} \odot g'(z^{(l)})
$$

**Solutions:**

- **Gradient clipping**: force gradient values (element-wise) to a fixed minimum/maximum if they exceed an expected range
- Use a **smaller learning rate**
- **Weight regularization**
- **Data normalization** (see below)
- **Re-design the network** to have fewer layers

### Gradient Vanishing

**Cause (intuition):** the mirror-image problem — if $w^{(l)} < 1$, then $z^{(l)}$ (and $\hat y$) **shrinks** exponentially as the signal passes through many layers, and so does the backpropagated gradient $\dfrac{\delta J(w)}{\delta w^{(l)}}$.

**Solutions:**

- Use **ReLU** instead of sigmoid: sigmoid squishes a large input space into a small output range $(0,1)$, so a large change in input causes only a small change in output — meaning the derivative becomes tiny
- Use **residual (skip) connections**: they provide a direct path back to earlier layers
- **Batch normalization**: forces each layer's activations to follow a consistent distribution, independent of upstream parameter changes
- Use **careful weight initialization**

### Data Normalization

Two common approaches to keep input values on a consistent, well-behaved scale:

**Min-max scaling** — rescale data into a fixed range, usually $[0,1]$:

$$
x_{norm} = \frac{x-x_{min}}{x_{max}-x_{min}}
$$

**Standardization** — rescale data to have mean $\boldsymbol{\mu}=0$ and standard deviation $\boldsymbol{\sigma}=1$:

$$
x_{norm} = \frac{x-\boldsymbol{\mu}}{\boldsymbol{\sigma}}
$$

> [!TIP]
> Raw pixel values typically range from 0–255. Without normalization, this large, skewed range of numbers can itself contribute to unstable, exploding gradients — normalizing the input data is one of the simplest and most effective first steps when training is unstable.

---

## 24. Debugging Strategies

| Strategy | What to do |
|---|---|
| **Qualitative analysis** | Visually inspect what the trained model detects — e.g. is it actually capturing the car license plates? |
| **Quantitative analysis** | Examine specific misclassifications using the returned softmax probabilities — e.g. the model might confuse the character `Q` with `0`, which could point to incorrectly labelled data |
| **Fit a tiny dataset** | Bugs in an implementation are often hard to spot. Try to overfit the model on a tiny dataset (e.g. 5–10 samples) first — if it *can't* even memorise 5 samples, something in the code is broken |
| **Monitor the learning curves** | Plot training vs validation loss/accuracy over time to catch overfitting, underfitting, or instability early |

---

## 25. Key Takeaways

> [!IMPORTANT]
> **The core things to remember from Week 5:**

1. A **CNN** organises its neurons in 3 dimensions (width, height, depth) and is built from **convolution**, **pooling**, and **fully connected** layers, ending in a **softmax** classifier
2. The **convolutional layer** slides a filter across the input, computing a weighted sum (plus bias, then activation) at every position to build a **feature map**
3. **Output size** after a conv/pool layer $= \dfrac{d-N+2p}{s}+1$, where $d$=input size, $N$=filter size, $p$=padding, $s$=stride
4. Using $N$ filters in one layer produces $N$ feature maps, stacked together to form the layer's output depth
5. **Weight sharing** (the same filter reused across the image) gives CNNs **translation invariance** and drastically fewer parameters than a fully connected layer
6. **Sparsity of connection** means each output value only depends on a small local patch of the input, which helps prevent overfitting and reduces the data needed to train
7. True mathematical "convolution" flips the filter 180°; deep learning frameworks skip this and use **cross-correlation** instead — it doesn't matter, since the filter weights are learned either way
8. **ReLU** ($\max(0,x)$) is generally preferred over sigmoid/tanh because it avoids saturation and reduces the vanishing gradient problem
9. **Pooling** (commonly **max pooling**) down-samples feature maps, improving generalisation and giving some robustness to small translations
10. A **fully connected layer** connects every neuron to every neuron in the previous layer; replacing FC layers with equivalent convolutional layers can massively cut parameter counts (e.g. 81.5M → 13M in one example)
11. **Softmax regression** = softmax activation + cross-entropy loss, producing a valid probability distribution across classes
12. Backpropagation through a conv layer sums each weight's contribution across **every** output pixel it influenced, because of weight sharing
13. Beyond plain SGD, optimizers like **Momentum, NAG, Adagrad, Adadelta, RMSProp,** and **Adam** help navigate the non-convex, saddle-point-riddled loss surfaces of deep networks
14. Key architectures: **LeNet-5** (the original), **AlexNet** (ReLU + dropout + GPUs), **VGG** (depth matters, stacked 3×3 filters), **GoogLeNet/Inception** (parallel multi-scale filters, fewer parameters), **ResNet** (skip connections solve vanishing gradients and degradation)
15. **Batch normalization** normalises each layer's inputs using the mini-batch's mean/variance (plus learnable scale/shift), which speeds up and stabilises training
16. A practical CNN project follows: **define the goal → build an end-to-end baseline pipeline with metrics → diagnose bottlenecks → iterate**
17. **Overfitting** (low bias, high variance) and **underfitting** (high bias, low variance) require opposite fixes — more data/regularization vs. more capacity/training time
18. **Gradient exploding/vanishing** stem from weights repeatedly multiplying above/below 1 across many layers; fixes include clipping, ReLU, batch norm, residual connections, and data normalization

---

## 26. Glossary

| Term | Definition |
|------|-----------|
| **Convolutional Neural Network (CNN)** | A deep learning model designed to process grid-like data (images, audio, video) using convolutional and pooling layers |
| **Feature Map** | The output produced by convolving a filter across an input |
| **Filter / Kernel** | A small matrix of learnable weights that slides across the input to detect a specific pattern |
| **Stride** | The number of pixels a filter shifts on each step |
| **Padding** | Extra (usually zero-valued) pixels added around the input's border before convolving |
| **Weight Sharing** | Reusing the same filter (set of weights) across every position in the input |
| **Translation Invariance** | The ability to recognise a pattern regardless of where it appears in the image |
| **Sparsity of Connection** | Each output element depends only on a small local region of the input, not the whole input |
| **Cross-Correlation** | The sliding-and-multiplying operation commonly (if loosely) called "convolution" in deep learning; unlike true mathematical convolution, it does not flip the filter |
| **Activation Function** | A (usually non-linear) function applied to a neuron's weighted sum, e.g. Sigmoid, Tanh, ReLU |
| **ReLU (Rectified Linear Unit)** | The activation function $f(x)=\max(0,x)$; avoids saturation for positive inputs |
| **Vanishing Gradient** | Gradients shrink exponentially as they backpropagate through many layers, stalling learning in earlier layers |
| **Exploding Gradient** | Gradients grow exponentially through many layers, causing unstable, huge weight updates |
| **Pooling Layer** | A layer that down-samples a feature map (e.g. via max pooling) to select strong features and add translation robustness |
| **Max Pooling** | A pooling method that keeps the maximum value within each window |
| **Fully Connected (FC) Layer** | A layer where every neuron connects to every neuron in the previous layer, each with its own independent weight |
| **Flatten** | Reshaping a multi-dimensional feature map into a 1-D vector before feeding it into FC layers |
| **Softmax** | An activation function that converts a vector of scores into a probability distribution that sums to 1 |
| **Cross-Entropy Loss** | The loss function that measures the negative log-probability assigned to the correct class |
| **Gradient Descent (Batch / Stochastic / Mini-batch)** | Optimization algorithms that update weights using the gradient of the loss, computed over the whole dataset / one sample / a small batch respectively |
| **Momentum** | An optimizer that accelerates and smooths gradient descent using an exponentially weighted average of past gradients |
| **Adagrad / Adadelta / RMSProp** | Adaptive-learning-rate optimizers that scale each parameter's update based on its historical gradient magnitude |
| **Adam** | A popular optimizer combining momentum-style averaging with adaptive, bias-corrected per-parameter learning rates |
| **LeNet-5** | An early CNN (1990s) designed for handwritten character recognition |
| **AlexNet** | A deeper, GPU-trained CNN that popularised ReLU and dropout |
| **VGG** | A CNN architecture showing that depth (via stacked 3×3 filters) improves performance, at the cost of many parameters |
| **Inception Layer (GoogLeNet)** | A layer that applies multiple filter sizes in parallel and concatenates the results, reducing overall parameter count |
| **Residual Block / Skip Connection (ResNet)** | A shortcut that adds a block's input directly to its output, easing the learning of very deep networks |
| **Internal Covariate Shift** | The phenomenon where each layer's input distribution keeps changing during training as earlier layers' weights update |
| **Batch Normalization** | Normalising each layer's inputs using the current mini-batch's mean and variance (plus learnable scale/shift), to stabilise and speed up training |
| **Overfitting** | A model that fits training data well but fails to generalise (low bias, high variance) |
| **Underfitting** | A model that performs poorly on both training and test data (high bias, low variance) |
| **Dropout** | A regularization technique that randomly disables neurons (and their connections) during training |
| **Data Augmentation** | Artificially expanding a training set via transformations like flipping, rotating, or adding noise |
| **Early Stopping** | Halting training once performance on a validation set stops improving, to prevent overfitting |
| **Min-Max Scaling** | Rescaling data into a fixed range, typically $[0,1]$ |
| **Standardization** | Rescaling data to have zero mean and unit standard deviation |
| **Intersection over Union (IoU)** | A metric for object detection measuring the overlap between a predicted and a ground-truth bounding box |
| **Mean Average Precision (mAP)** | The average of the per-class average precision, used for both classification and detection tasks |

---

> [!TIP]
> **Study tip for Week 5:** Make sure you can:
> 1. Explain, in your own words, why a CNN organises neurons in 3-D volumes instead of flat layers
> 2. Compute the output size of a conv/pool layer given input size, filter size, stride, and padding
> 3. Work through a small convolution or max-pooling example by hand
> 4. Explain weight sharing, translation invariance, and sparsity of connection — and why they matter
> 5. Explain why ReLU is generally preferred over sigmoid/tanh in deep networks
> 6. Describe what a residual/skip connection does and why it helps very deep networks train
> 7. Explain what batch normalization does and why it's applied before the activation function
> 8. Compare Momentum, Adagrad, RMSProp, and Adam — what problem does each one solve?
> 9. Recognise the symptoms of overfitting, underfitting, exploding gradients, and vanishing gradients, and name at least one fix for each
