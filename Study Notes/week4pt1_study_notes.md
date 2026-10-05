# 📘 Week 4 Study Notes: Artificial Neural Networks (ANN)

> **Course:** COS30082 — Applied Machine Learning  
> **Topic:** Artificial Neural Networks (ANN)  
> **Prerequisite:** Week 3 (Logistic Regression)

---

## Table of Contents

1. [Why Do We Need a Non-Linear Hypothesis?](#1-why-do-we-need-a-non-linear-hypothesis)
2. [Understanding Artificial Neural Networks](#2-understanding-artificial-neural-networks)
3. [From Logistic Regression to a Neuron](#3-from-logistic-regression-to-a-neuron)
4. [Neural Network Architecture](#4-neural-network-architecture)
5. [ANN Notation and Weight Dimensions](#5-ann-notation-and-weight-dimensions)
6. [The ANN Cost Function](#6-the-ann-cost-function)
7. [Feature Learning and Deep Networks](#7-feature-learning-and-deep-networks)
8. [Multiclass Classification](#8-multiclass-classification)
9. [Forward Propagation](#9-forward-propagation)
10. [Why Backpropagation Is Needed](#10-why-backpropagation-is-needed)
11. [Backward Propagation](#11-backward-propagation)
12. [The Backpropagation Algorithm](#12-the-backpropagation-algorithm)
13. [Backpropagation Intuition](#13-backpropagation-intuition)
14. [Random Weight Initialization](#14-random-weight-initialization)
15. [Logistic Regression vs ANN](#15-logistic-regression-vs-ann)
16. [Key Takeaways](#16-key-takeaways)
17. [Glossary](#17-glossary)

---

## 1. Why Do We Need a Non-Linear Hypothesis?

### The Problem: Real Data Is Often Complicated

<p align="center">
  <img src="https://github.com/user-attachments/assets/eb699e48-a56d-4c60-ab7a-95a6e335e84f" width=1000>
</p>

Suppose we want to classify an image as either a **dog** or a **cat**. Each pixel can be treated as an input feature:

$$
x = \begin{bmatrix}x_1 & x_2 & \cdots & x_n\end{bmatrix}^T
$$

For a grayscale image of $60 \times 60$ pixels:

$$
n = 60 \times 60 = 3600 \text{ features}
$$

For an RGB image, every pixel has red, green, and blue channels:

$$
n = 60 \times 60 \times 3 = 10{,}800 \text{ features}
$$

The classification process can be summarized as:

<p align="center">
  <img src="https://github.com/user-attachments/assets/0e260c37-f470-4114-9732-e7010f59399c" />
</p>

### From Image Pixels to a Non-Linear Decision Boundary

<table align="center">
  <tr>
    <th align="center" width="33%">
      1️⃣ Extract Pixel Features
    </th>
    <th align="center" width="33%">
      2️⃣ Plot Training Examples
    </th>
    <th align="center" width="33%">
      3️⃣ Learn a Non-Linear Boundary
    </th>
  </tr>

  <tr>
    <td align="center" valign="top">
      <a href="https://github.com/user-attachments/assets/bfc159ba-595b-40c8-b46d-f8958ac14c21">
        <img
          src="https://github.com/user-attachments/assets/bfc159ba-595b-40c8-b46d-f8958ac14c21"
          width="100%"
          alt="Dog and cat images represented using two pixel features">
      </a>
    </td>
    <td align="center" valign="top">
      <a href="https://github.com/user-attachments/assets/d4df5120-5c6c-4d96-bcbe-0ef87d85aeac">
        <img
          src="https://github.com/user-attachments/assets/d4df5120-5c6c-4d96-bcbe-0ef87d85aeac"
          width="100%"
          height="750"
          alt="Dog and cat training examples plotted using pixel features">
      </a>
    </td>
    <td align="center" valign="top">
      <a href="https://github.com/user-attachments/assets/62d846af-8c92-40a1-96e4-d303ea221713">
        <img
          src="https://github.com/user-attachments/assets/62d846af-8c92-40a1-96e4-d303ea221713"
          width="100%"
          height="760"
          alt="Non-linear decision boundary separating dogs and cats">
      </a>
    </td>
  </tr>

  <tr>
    <td align="center" valign="top">
      Selected pixel intensities become the input features
      <b>x<sub>1</sub></b> and <b>x<sub>2</sub></b>.
    </td>
    <td align="center" valign="top">
      Each image becomes a training point. 
      <br>🔵 = Dogs
      🔴 = Cats
    </td>
    <td align="center" valign="top">
      A non-linear decision boundary is needed to separate the two classes.
    </td>
  </tr>
</table>

### Why Not Add Polynomial Features?

From logistic regression, we know that polynomial terms can create a non-linear decision boundary. For example:

$$
h_\theta(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_1^2 + \theta_4x_1x_2 + \theta_5x_2^2)
$$

However, manually adding quadratic features to an image causes the number of inputs to grow enormously.

The approximate number of second-order feature combinations is:

$$
\frac{n(n+1)}{2}
$$

For $n = 10{,}800$ RGB features:

$$
\frac{10{,}800(10{,}801)}{2} \approx 58.3 \text{ million features}
$$

<div align="center">

| Problem | Effect |
|---------|--------|
| Too many polynomial features | Very high memory usage |
| Manual feature construction | Difficult to decide which combinations matter |
| High-dimensional input | Expensive model training and prediction |
| Complex patterns | A simple linear boundary may underfit |

</div>

> [!IMPORTANT]
> **Neural networks provide a better alternative:** instead of manually creating millions of polynomial features, the hidden layers learn useful non-linear features automatically.

> **Analogy:** Polynomial feature engineering is like manually listing every possible combination of ingredients before cooking. A neural network learns which ingredient combinations are useful while it trains.

---

## 2. Understanding Artificial Neural Networks

### Inspiration from the Human Brain

The human brain contains interconnected biological neurons. A neuron receives signals, processes them, and sends an output to other neurons.

An **Artificial Neural Network (ANN)** follows a simplified version of the same idea:

<p align="center">
  <img src="https://github.com/user-attachments/assets/1217fa08-7e92-48fd-9ae2-523c57f766a1" width=600>
</p>

1. Receive input values
2. Multiply each input by a weight
3. Add the weighted inputs and a bias
4. Apply an activation function
5. Produce an output

### The Artificial Neuron

For inputs $x_1, x_2, \ldots, x_n$, weights $w_1, w_2, \ldots, w_n$, and bias $b$:

$$
z = b + \sum_{i=1}^{n} w_i x_i
$$

The neuron then applies an activation function $g$:

$$
a = g(z)
$$

<div align="center">

```text
Inputs x → Weighted Sum z → Activation g(z) → Output a
```

</div>

| Component | Purpose |
|-----------|---------|
| **Inputs** $x$ | Values supplied to the neuron |
| **Weights** $w$ or $\theta$ | Control the importance of each input |
| **Bias** $b$ | Shifts the activation threshold |
| **Weighted sum** $z$ | Combines the inputs before activation |
| **Activation** $g(z)$ | Introduces non-linearity |
| **Output** $a$ | Value passed to the next layer |

> **Analogy:** A neuron is like a panel of judges. Every judge gives a score ($x_i$), each judge has a different influence ($w_i$), and the bias adjusts the starting point. The activation function converts the total score into a decision signal.

---

## 3. From Logistic Regression to a Neuron

### Logistic Regression as a Single Neuron

Recall the logistic regression hypothesis:

$$
h_\theta(x) = g(\theta^T x)
$$

where the sigmoid function is:

$$
g(z) = \frac{1}{1+e^{-z}}
$$

and:

$$
z = \theta^T x
$$

Using a bias input $x_0=1$:

$$
z = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n
$$

### Neuron Terminology for Logistic Regression

| Logistic Regression Term | ANN Term |
|--------------------------|----------|
| Features $x_1,\ldots,x_n$ | Input neurons |
| Intercept $\theta_0$ | Bias weight |
| Parameters $\theta$ | Weights |
| $\theta^Tx$ | Weighted sum $z$ |
| Sigmoid $g(z)$ | Activation function |
| $h_\theta(x)$ | Predicted output / activation |

For binary classification:

$$
0 < h_\theta(x) < 1
$$

The prediction can therefore be interpreted as:

$$
h_\theta(x)=P(y=1\mid x;\theta)
$$

> [!NOTE]
> Logistic regression can be viewed as a neural network with an input layer and an output neuron, but **no hidden layer**.

### Why a Hidden Layer Changes Everything

A logistic regression neuron directly maps the original inputs to the output. An ANN inserts one or more hidden layers. The hidden neurons create new learned features, allowing the model to represent much more complex patterns.

<table align="center">
<tr>

<td align="center" width="50%" valign="top">

<img src="https://github.com/user-attachments/assets/87b0d2eb-3fc0-49bc-8240-71316b3679b7" width="100%">

<br>

Logistic Regression

</td>

<td align="center" width="50%" valign="top">

<img src="https://github.com/user-attachments/assets/dfc288e3-0375-48fe-a70e-2069659bba77" width="100%">

<br>

Neural Networks
</td></tr></table>

---

## 4. Neural Network Architecture

### The Three Main Layer Types

<div align="center">

| Layer | Role |
|-------|------|
| **Input layer** | Holds the original features $x$ |
| **Hidden layer(s)** | Learns intermediate feature representations |
| **Output layer** | Produces the final prediction $h_\theta(x)$ |

</div>

### A Single-Hidden-Layer Network

Consider a network with:

- 3 input features
- 3 hidden neurons
- 1 output neuron

<p align="center">
  <img src="https://github.com/user-attachments/assets/cfc0e736-d888-4c89-9d0d-3f09ae22473c" width=600>
</p>

Every neuron in one layer is normally connected to every non-bias neuron in the next layer. This is called a **fully connected** or **dense** layer.

### Bias Units (encircled with blue colour)

A bias unit has a fixed activation of +1, where $l$ is the current layer:

$$
a_0^{(l)} = 1
$$

It is added to each non-output layer and has its own outgoing weights.

> [!WARNING]
> Bias units are **not counted** when stating the number of ordinary neurons $s_l$ in a layer, but the bias column **is included** in the weight matrix.

---

## 5. ANN Notation and Weight Dimensions

### Core Notation

| Symbol | Meaning |
|--------|---------|
| $L$ | Total number of layers |
| $s_l$ | Number of non-bias neurons in layer $l$ |
| $a_i^{(l)}$ | Activation of neuron $i$ in layer $l$ |
| $z_i^{(l)}$ | Weighted input to neuron $i$ in layer $l$ |
| $\Theta^{(l)}$ | Weight matrix mapping layer $l$ to layer $l+1$ |
| $\theta_{jk}^{(l)}$ | Weight from neuron $k$ in layer $l$ to neuron $j$ in layer $l+1$ |
| $h_\Theta(x)$ | Network prediction |

```mermaid
flowchart LR
    subgraph layer1["Layer 1: Input Layer"]
        direction TB
        B1(("Bias<br/>a₀⁽¹⁾ = 1"))
        X1(("a₁⁽¹⁾ = x₁"))
        X2(("a₂⁽¹⁾ = x₂"))
        X3(("a₃⁽¹⁾ = x₃"))
    end

    subgraph layer2["Layer 2: Hidden Layer"]
        direction TB
        B2(("Bias<br/>a₀⁽²⁾ = 1"))
        A1(("z₁⁽²⁾ → a₁⁽²⁾"))
        A2(("z₂⁽²⁾ → a₂⁽²⁾"))
        A3(("z₃⁽²⁾ → a₃⁽²⁾"))
    end

    subgraph layer3["Layer 3: Output Layer"]
        direction TB
        Y(("z₁⁽³⁾ → a₁⁽³⁾"))
    end

    B1 --> A1
    B1 --> A2
    B1 --> A3

    X1 -->|"θ₁₁⁽¹⁾"| A1
    X1 --> A2
    X1 --> A3

    X2 --> A1
    X2 --> A2
    X2 --> A3

    X3 --> A1
    X3 --> A2
    X3 --> A3

    B2 --> Y
    A1 -->|"θ₁₁⁽²⁾"| Y
    A2 --> Y
    A3 --> Y

    classDef input fill:#2563eb,stroke:#93c5fd,color:#ffffff
    classDef hidden fill:#7c3aed,stroke:#c4b5fd,color:#ffffff
    classDef output fill:#059669,stroke:#6ee7b7,color:#ffffff
    classDef bias fill:#dc2626,stroke:#fca5a5,color:#ffffff

    class X1,X2,X3 input
    class A1,A2,A3 hidden
    class Y output
    class B1,B2 bias
```

In this network, the total number of layers is $L=3$. The values $s_1=3$, $s_2=3$, and $s_3=1$ describe the number of non-bias neurons in each layer.

Each $z_i^{(l)}$ is the weighted input entering neuron $i$, while $a_i^{(l)}=g(z_i^{(l)})$ is its activation. The matrix $\Theta^{(l)}$ contains all weights connecting layer $l$ to layer $l+1$, whereas $\theta_{jk}^{(l)}$ represents one individual connection from neuron $k$ to neuron $j$.

> [!NOTE]
> Capital $\Theta$ is commonly used for a layer's complete weight matrix, while lowercase $\theta_{jk}$ represents one individual weight.

### Weight Matrix Dimension Rule

If layer $l$ has $s_l$ neurons and layer $l+1$ has $s_{l+1}$ neurons, then:

$$
\boxed{\Theta^{(l)} \in \mathbb{R}^{s_{l+1}\times(s_l+1)}}
$$

Why?

- **Rows:** one row for each destination neuron in layer $l+1$
- **Columns:** one column for each source neuron in layer $l$, plus one bias column

### Example

For 3 input neurons, 3 hidden neurons, and 1 output neuron:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6ce1e26b-5945-4dca-a5e5-da938a3d5c33" width=600>
</p>

<div align="center">
  
| Mapping | Destination Neurons | Source Neurons + Bias | Matrix Size |
|---------|---------------------|-----------------------|-------------|
| Input (Layer 1) → Hidden (Layer 2) | 3 | $3+1=4$ | $3\times4$ |
| Hidden (Layer 2) → Output (Layer 3) | 1 | $3+1=4$ | $1\times4$ |

</div>

### Parameter Count

The total number of weights in this network is:

$$
(3\times4)+(1\times4)=16
$$

> **Memory trick:** **Next × (Current + 1)** gives the weight-matrix dimensions.

---
## 6. The ANN Cost Function

### Important Terminology

<div align="center">
  
| Symbol | Meaning |
|--------|---------|
| $K$ | Number of output neurons/classes |
| $L$ | Total number of network layers |
| $s_l$ | Number of non-bias neurons in layer $l$ |
| $m$ or $N$ | Number of training examples |

</div>

### Binary Classification Cost

For one sigmoid output, binary cross-entropy is:

$$
J(\Theta)=-\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log h_\Theta(x^{(i)})+(1-y^{(i)})\log\left(1-h_\Theta(x^{(i)})\right)\right]
$$

### Multiclass Classification Cost

For $K$ output neurons:

$$
J(\Theta)=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}\left[y_k^{(i)}\log h_\Theta(x^{(i)})_k+(1-y_k^{(i)})\log\left(1-h_\Theta(x^{(i)})_k\right)\right]
$$

The purpose of training is to find weights that minimize $J(\Theta)$.

### What the Cost Measures

| Prediction | Actual Class | Cost Behaviour |
|------------|--------------|----------------|
| Confident and correct | Match | Very small cost |
| Uncertain | Either | Moderate cost |
| Confident and wrong | Mismatch | Very large cost |

> **Analogy:** The cost function is the network's report card. Forward propagation answers the questions; the cost tells the network how badly its answers differ from the targets.

---

## 7. Feature Learning and Deep Networks

### Hidden Neurons Learn New Features

Each hidden activation is a learned function of the previous layer:

$$
a^{(2)}=g\left(\Theta^{(1)}x\right)
$$

Instead of manually selecting polynomial terms, the network learns which combinations of inputs help reduce the prediction error.

<div align="center">
  
| Model | Feature Representation |
|-------|------------------------|
| Logistic Regression | Uses the supplied features directly |
| ANN with hidden layers | Learns new features from the supplied features |

</div>

### Deeper Networks

A deeper network contains more than one hidden layer:

<p align="center">
  <img src="https://github.com/user-attachments/assets/8acd6117-ee5e-40eb-a7e6-a04ffa58829a" width=1000>
</p>

As information travels deeper through the network, representations can become increasingly abstract.

For image classification, a simplified interpretation is:

<div align="center">
  
| Layer | Possible Learned Feature |
|-------|--------------------------|
| Early hidden layer | Edges and colour changes |
| Middle hidden layer | Curves, textures, eyes, ears |
| Deeper hidden layer | Complete object parts or shapes |
| Output layer | Dog, cat, or another class |

</div>

> [!IMPORTANT]
> The network is not explicitly told to detect an edge or an ear. These useful intermediate features are learned through training.

---

## 8. Multiclass Classification

### More Than Two Classes

<p align="center">
  <img src="https://github.com/user-attachments/assets/941a2fcb-f9f9-4694-ae5e-fae10bfa462f" width=1000>
</p>

Suppose an image must be classified into three classes:

1. Dog
2. Penguin
3. Tiger

The output layer should contain one neuron per class:

$$
h_\Theta(x)\in\mathbb{R}^3
$$

### One-Hot Encoding

Instead of storing the label as 1, 2, or 3, represent it as a vector:

<div align="center">
  
| Class | One-Hot Target $y$ |
|-------|--------------------|
| Dog | $\left[1,0,0\right]^T$ |
| Penguin | $\left[0,1,0\right]^T$ |
| Tiger | $\left[0,0,1\right]^T$ |
  
</div>

Only the position belonging to the correct class is 1.

### Interpreting the Output

If the network produces:

$$ h_\Theta(x)=\begin{bmatrix}0.08 \\\\0.87 \\\\0.05\end{bmatrix} $$

then the second output is largest, so the model predicts **penguin**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6d636018-0413-4f36-920f-d1b19f81c0df" width=600>
</p>

### Binary vs Multiclass Output

| Problem | Target | Network Output |
|---------|--------|----------------|
| Binary classification | $y\in\{0,1\}$ | $h_\Theta(x)\in\mathbb{R}$ |
| $K$-class classification | $y\in\mathbb{R}^K$ | $h_\Theta(x)\in\mathbb{R}^K$ |

> [!NOTE]
> For mutually exclusive classes, modern neural networks commonly use **softmax** in the output layer so that all $K$ output probabilities sum to 1.

---

## 9. Forward Propagation

### What Is Forward Propagation?

**Forward propagation** computes the network's prediction / calculates the weighted sum of $z_j^{(l)}$ by moving from the input layer to the output layer.

For each layer:

$$z^{(l+1)}=\Theta^{(l)}a^{(l)}$$

$$a^{(l+1)}=g\left(z^{(l+1)}\right)$$

The activation function is applied element by element.

### Step-by-Step Forward Propagation for One Hidden Layer

Consider a neural network containing:

- Three input features
- Three hidden neurons
- One output neuron
- A bias unit in Layers 1 and 2

> [!IMPORTANT]
> The superscript identifies the **layer**, not a power.  
> For example, $\(a_2^{(1)}\)$ means the activation of neuron 2 in Layer 1.

---

### Step 1: Pass the Inputs to the Hidden Layer

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/eb99379d-2f01-4919-b42c-5ed96d88328e"
    width="80%"
    alt="Input features propagated to the hidden layer">
</p>

<p align="center">
  <em>
    Step 1 — The input features and input bias are passed from Layer 1 towards the three neurons in Layer 2.
  </em>
</p>

The input-layer activation is written as:

$$
a^{(1)}=x
$$

The three ordinary input activations are:

$$
a_1^{(1)},\qquad
a_2^{(1)},\qquad
a_3^{(1)}
$$

The input layer also contains the fixed bias unit:

$$
a_0^{(1)}=x_0= +1
$$

Therefore, the complete input vector used in the calculation is:

$$a^{(1)}=\begin{bmatrix}a_0 \cr a_1 \cr a_2 \cr a_3\end{bmatrix}=\begin{bmatrix}1 \cr a_1 \cr a_2 \cr a_3\end{bmatrix}$$

---

### Step 2: Calculate the Hidden Layer Weighted Inputs

The first weight matrix maps Layer 1 to Layer 2:

$$\Theta^{(1)}\in\mathbb{R}^{3\times4}$$

It has three rows because Layer 2 contains three neurons and four columns because Layer 1 supplies three inputs plus one bias.

The hidden-layer weighted-input vector is:

$$
z^{(2)}=\Theta^{(1)}a^{(1)}
$$

Its dimensions are:

$$\underbrace{z^{(2)}}_{3\times1}=\underbrace{\Theta^{(1)}}_{3\times4}\underbrace{a^{(1)}}_{4\times1}$$

Therefore:

$$z^{(2)}=\begin{bmatrix}z_1^{(2)} \cr z_2^{(2)} \cr z_3^{(2)}\end{bmatrix}$$

For the three hidden neurons:

$$z_1^{(2)}=\theta_{10}^{(1)}x_0
+\theta_{11}^{(1)}x_1
+\theta_{12}^{(1)}x_2
+\theta_{13}^{(1)}x_3
$$

$$z_2^{(2)}=
\theta_{20}^{(1)}x_0
+\theta_{21}^{(1)}x_1
+\theta_{22}^{(1)}x_2
+\theta_{23}^{(1)}x_3
$$

$$z_3^{(2)}=
\theta_{30}^{(1)}x_0
+\theta_{31}^{(1)}x_1
+\theta_{32}^{(1)}x_2
+\theta_{33}^{(1)}x_3
$$

Because $x_0=1$, the terms $\(\theta_{10}^{(1)}x_0\)$, $\(\theta_{20}^{(1)}x_0\)$, and $\(\theta_{30}^{(1)}x_0\)$ are the bias contributions.

---

### Step 3: Activate the Hidden Neurons (neurons in Layer 2)

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/06252155-1229-47c6-ae64-4c8316518c8c"
    width="80%"
    alt="Hidden-layer activations calculated during forward propagation">
</p>

<p align="center">
  <em>
    Step 2 to Step 3 — Layer 2 first calculates
    <i>z</i><sup>(2)</sup> =
    &Theta;<sup>(1)</sup><i>a&#771;</i><sup>(1)</sup>,
    then applies <i>g</i> to produce
    <i>a</i><sup>(2)</sup>.
  </em>
</p>

The activation function produces only the three ordinary hidden-neuron activations. Before propagating them to Layer 3 (output layer), another bias unit is added:

$$a_0^{(2)}=1$$

The bias-augmented hidden-layer vector is therefore:

$$\{a}^{(2)}=\begin{bmatrix}1 \cr a_1^{(2)} \cr a_2^{(2)} \cr a_3^{(2)}\end{bmatrix}$$

> [!NOTE]
> $\(\{a}^{(l)}\)$ indicates that the activation vector includes the bias unit $\(a_0^{(l)}=1\)$.

---

### Step 4: Add the Hidden-Layer Bias

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/89770faf-6d11-40d7-bbe0-03f54985fea6"
    width="80%"
    alt="Hidden-layer bias added before the output calculation">
</p>

<p align="center">
  <em>
    Step 4 — After calculating the three hidden activations, the bias
    <i>a</i><sub>0</sub><sup>(2)</sup> = 1 is added before propagation
    to Layer 3.
  </em>
</p>

The activation function produces only the three ordinary hidden-neuron activations. A new fixed bias unit is then added:

$$
a_0^{(2)}=1
$$

The resulting bias-augmented activation vector is:

$$\ a^{(2)}=\begin{bmatrix}a_0^{(2)} \cr a_1^{(2)} \cr a_2^{(2)} \cr a_3^{(2)}\end{bmatrix}=\begin{bmatrix}1 \cr a_1^{(2)} \cr a_2^{(2)} \cr a_3^{(2)}\end{bmatrix}$$

> [!NOTE]
> The symbol $a^{(2)}$ represents the ordinary hidden-neuron
> activations, while $\ a^{(2)}$ represents the same activation
> vector with the bias unit included.

---

#### Step 5: Calculate the Output-Layer Weighted Input

The second weight matrix maps Layer 2 to Layer 3:

$$\Theta^{(2)}\in\mathbb{R}^{1\times4}$$

It has one row because Layer 3 contains one output neuron and four columns because Layer 2 supplies three hidden activations plus one bias.

The output weighted input is:

$$z^{(3)}=\Theta^{(2)}\ a^{(2)}$$

Its dimensions are:

$$\underbrace{z^{(3)}}_{1\times1}=\underbrace{\Theta^{(2)}}_{1\times4}\underbrace{\ a^{(2)}}_{4\times1}$$

Expanded:

$$z^{(3)}=\theta_{10}^{(2)}a_0^{(2)}+\theta_{11}^{(2)}a_1^{(2)}+\theta_{12}^{(2)}a_2^{(2)}+\theta_{13}^{(2)}a_3^{(2)}$$

Since $\(a_0^{(2)}=1\)$:

$$z^{(3)}=\theta_{10}^{(2)}+\theta_{11}^{(2)}a_1^{(2)}+\theta_{12}^{(2)}a_2^{(2)}+\theta_{13}^{(2)}a_3^{(2)}$$

---

#### Step 6: Calculate the Final Output

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/ed4fd850-5f38-4a99-9e32-e418f49686da"
    width="80%"
    alt="Final prediction produced by forward propagation">
</p>

<p align="center">
  <em>
    Final step — The output neuron applies <i>g</i> to
    <i>z</i><sup>(3)</sup>, producing the final prediction
    <i>h</i><sub>&Theta;</sub>(<i>x</i>).
  </em>
</p>

The output neuron applies the activation function:

$$
a^{(3)}=g\left(z^{(3)}\right)
$$

The result is the neural-network hypothesis:

$$\boxed{h_\Theta(x)=a^{(3)}=g\left(z^{(3)}\right)}$$

> **Analogy:** Forward propagation is like doing a presentation from beginning to end. The input layer represents the collected research, facts, and findings. The hidden layers filter the important information, connect related ideas, and organise them into clear slides. The output layer represents the final conclusion or message presented to the audience.

---

## 10. Why Backpropagation Is Needed

### The Training Problem

Forward propagation tells us the prediction, and the cost function tells us how wrong it is. We still need to determine:

> Which weights caused the error, and how should each one change?

The gradient answers this question:

$$
\frac{\partial J(\Theta)}{\partial \theta_{jk}^{(l)}}
$$

It measures how sensitive the cost is to a small change in one weight.

### Gradient Descent Update

Once the gradient is known:

$$
\theta_{jk}^{(l)}:=\theta_{jk}^{(l)}-\alpha\frac{\partial J(\Theta)}{\partial\theta_{jk}^{(l)}}
$$

where $\alpha$ is the learning rate.

### Why Calculate Backwards?

The prediction depends on the last hidden layer, which depends on the previous hidden layer, and so on. The **chain rule** lets the output error be traced backwards through these dependencies.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0945a47-4629-42ee-af87-c661fab5facf" width=600>
</p>

<p align="center">
  <em>Forward propagation:   Input  → Hidden → Output → Cost</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/85fd0335-3e8c-499c-bde8-404e0dbb4ec2" width=600>
</p>

<p align="center">
  <em>Backward propagation:  Input  ← Hidden ← Error  ← Cost</em>
</p>

> **Analogy:** If the final product from a production line is faulty, backpropagation inspects the last process first, then traces responsibility backwards to earlier processes.

---
## 11. Backward Propagation

Backward propagation, commonly called **backpropagation**, is an efficient method for calculating how much each neural-network parameter contributes to the prediction error.

Forward propagation moves from the input layer to the output layer:

$$x\longrightarrow a^{(2)}\longrightarrow h_\Theta(x)$$

Backward propagation works in the opposite direction:

$$\text{output error}\longrightarrow\text{hidden-layer error}\longrightarrow\text{parameter gradients}$$

It repeatedly applies the chain rule to calculate the derivatives of the cost function \(J\) with respect to every weight in the network:

$$
\frac{\partial J}{\partial\Theta^{(l)}}
$$

These gradients indicate:

- Which weights contributed to the error
- The direction in which each weight should move
- The relative amount by which each weight should change

> [!IMPORTANT]
> Backpropagation calculates the gradients, but it does not update the
> weights itself. An optimizer such as gradient descent uses the
> calculated gradients to update the parameters.

---

### Error-Term Notation

The error associated with neuron $\(j\)$ in layer $\(l\)$ is represented by:

$$\boxed{\delta_j^{(l)}=\frac{\partial J}{\partial z_j^{(l)}}}$$

The vector containing the error terms of all ordinary neurons in layer $\(l\)$ is:

$$\delta^{(l)}=\begin{bmatrix}\delta_1^{(l)} \cr \delta_2^{(l)} \cr \vdots \cr \delta_{s_l}^{(l)}\end{bmatrix}$$

The error term measures how sensitive the cost is to a change in the neuron's weighted input $\(z_j^{(l)}\)$.

A large magnitude of $\(\delta_j^{(l)}\)$ means that the neuron made a relatively large contribution to the final error.

> [!NOTE]
> The error term belongs to the weighted input $\(z_j^{(l)}\)$, not
> directly to the activation $\(a_j^{(l)}\)$.

---

### Step 1: Calculate the Output-Layer Error

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/f946bc1c-b0cd-4757-a4b5-c41673f96c63"
    width="80%"
    alt="Calculation of the output-layer error">
</p>

<p align="center">
  <em>
    The backward pass begins by calculating how sensitive the cost is to the weighted input of the output neuron.
  </em>
</p>

Backward propagation begins at the output layer because this is where the prediction can be compared with the expected target.

For the one-output network:

$$
a^{(3)}=g\left(z^{(3)}\right)=h_\Theta(x)
$$

The output error is defined as:

$$\delta^{(3)}=\frac{\partial J}{\partial z^{(3)}}
$$

Using the chain rule:

$$\delta^{(3)}=\frac{\partial J}{\partial a^{(3)}}\frac{\partial a^{(3)}}{\partial z^{(3)}}$$

Since:

$$\frac{\partial a^{(3)}}{\partial z^{(3)}}=g'\left(z^{(3)}\right)$$

the general output-error equation is:

$$\boxed{\delta^{(3)}=\frac{\partial J}{\partial a^{(3)}}g'\left(z^{(3)}\right)}$$

For a sigmoid output combined with binary cross-entropy loss, this expression simplifies to:

$$
\boxed{
\delta^{(3)}=a^{(3)}-y
}
$$

Because $\(a^{(3)}=h_\Theta(x)\)$:

$$
\boxed{
\delta^{(3)}=h_\Theta(x)-y
}
$$

The sign of the error provides useful information:

- $\(\delta^{(3)}>0\)$: the prediction is higher than the target
- $\(\delta^{(3)}<0\)$: the prediction is lower than the target
- $\(\delta^{(3)}=0\)$: the prediction matches the target

> [!NOTE]
> The simplification $\(\delta^{(3)}=a^{(3)}-y\)$ depends on the chosen
> output activation and loss function. For other combinations, use the
> complete chain-rule expression.

---

### Step 2: Calculate the Gradient of the Output Weights

The output-layer weighted input is:

$$
z^{(3)}=\Theta^{(2)}\ a^{(2)}
$$

An individual output weight $\(\theta_{1k}^{(2)}\)$ connects activation $\(a_k^{(2)}\)$ to the output neuron.

Using the chain rule:

$$\frac{\partial J}{\partial\theta_{1k}^{(2)}}=\frac{\partial J}{\partial z^{(3)}}\frac{\partial z^{(3)}}{\partial\theta_{1k}^{(2)}}$$

Since:

$$
\frac{\partial J}{\partial z^{(3)}}=\delta^{(3)}
$$

and:

$$\frac{\partial z^{(3)}}{\partial\theta_{1k}^{(2)}}=a_k^{(2)}$$

the gradient is:

$$\boxed{\frac{\partial J}{\partial\theta_{1k}^{(2)}}=\delta^{(3)}a_k^{(2)}}$$

For all output-layer weights simultaneously:

$$\boxed{\frac{\partial J}{\partial\Theta^{(2)}}=\delta^{(3)}\left(\ a^{(2)}\right)^T}$$

The matrix dimensions are:

$$\underbrace{\frac{\partial J}{\partial\Theta^{(2)}}}_{1\times4}=\underbrace{\delta^{(3)}}_{1\times1}\underbrace{\left(\ a^{(2)}\right)^T}_{1\times4}$$

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/f5fe9836-7cf8-4547-8f84-4ae5ab1f27b7"
    width="80%"
    alt="Calculation of the output-layer weight gradients">
</p>

<p align="center">
  <em>
    The output error is multiplied by each hidden-layer activation to
    calculate the gradient of every weight in
    &Theta;<sup>(2)</sup>.
  </em>
</p>

The bias weight follows the same rule. Because $\(a_0^{(2)}=1\)$:

$$\frac{\partial J}{\partial\theta_{10}^{(2)}}=\delta^{(3)}a_0^{(2)}=\delta^{(3)}$$

---

### Step 3: Propagate the Error to the Hidden Layer

After calculating the output error, the error is propagated backwards into the three hidden neurons.

Only the non-bias columns of $\(\Theta^{(2)}\)$ are used:

$$\Theta_{\mathrm{nb}}^{(2)}=\begin{bmatrix}\theta_{11}^{(2)}&\theta_{12}^{(2)}&\theta_{13}^{(2)}\end{bmatrix}$$

The hidden-layer error is:

$$\boxed{\delta^{(2)}=\left(\Theta_{\mathrm{nb}}^{(2)}\right)^T\delta^{(3)}\odot g'\left(z^{(2)}\right)}$$

where $\(\odot\)$ represents element-wise multiplication.

The dimensions are:

$$\underbrace{\delta^{(2)}}_{3\times1}=\left(\underbrace{\left(\Theta_{\mathrm{nb}}^{(2)}\right)^T}_{3\times1}\underbrace{\delta^{(3)}}_{1\times1}\right)\odot\underbrace{g'\left(z^{(2)}\right)}_{3\times1}$$

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/1e865bea-501f-48a4-be0a-8e397f366a9d"
    width="80%"
    alt="Propagation of the output error into the hidden layer">
</p>

<p align="center">
  <em>
    The output error is distributed backwards through the associated
    weights and adjusted by the local activation derivatives of the
    hidden neurons.
  </em>
</p>

For each hidden neuron $j$:

$$\delta_j^{(2)}=\theta_{1j}^{(2)}\delta^{(3)}g'\left(z_j^{(2)}\right)$$

For sigmoid activation:

$$g'(z)=g(z)\left(1-g(z)\right)$$

Since \(a=g(z)\), this can also be written as:

$$g'\left(z^{(2)}\right)=a^{(2)}\odot\left(1-a^{(2)}\right)$$

> [!IMPORTANT]
> There is no error term for the bias unit $(a_0^{(2)})$. The bias is a
> fixed value of 1 rather than an ordinary neuron activation.

---

### Step 4: Calculate the Gradient of the Input-to-Hidden Weights

The hidden-layer weighted input is:

$$
z^{(2)}=\Theta^{(1)}a^{(1)\}
$$

For an individual weight $(\theta_{jk}^{(1)})$:

$$\frac{\partial J}{\partial\theta_{jk}^{(1)}}=\frac{\partial J}{\partial z_j^{(2)}}\frac{\partial z_j^{(2)}}{\partial\theta_{jk}^{(1)}}$$

Since:

$$\frac{\partial J}{\partial z_j^{(2)}}=\delta_j^{(2)}$$

and:

$$\frac{\partial z_j^{(2)}}{\partial\theta_{jk}^{(1)}}=a_k^{(1)}$$

the gradient is:

$$\boxed{\frac{\partial J}{\partial\theta_{jk}^{(1)}}=\delta_j^{(2)}a_k^{(1)}}$$

For the complete matrix:

$$\boxed{\frac{\partial J}{\partial\Theta^{(1)}}=\delta^{(2)}\left(\ a^{(1)}\right)^T}$$

The matrix dimensions are:

$$\underbrace{\frac{\partial J}{\partial\Theta^{(1)}}}_{3\times4}=\underbrace{\delta^{(2)}}_{3\times1}\underbrace{\left(\ a^{(1)}\right)^T}_{1\times4}$$

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/bcd73ea9-5ddd-4a1b-95f1-25403438d192"
    width="80%"
    alt="Calculation of the input-to-hidden weight gradients">
</p>

<p align="center">
  <em>
    Each hidden-neuron error is multiplied by the corresponding
    input-layer activation to calculate the gradients of
    &Theta;<sup>(1)</sup>.
  </em>
</p>

For each bias weight in $(\Theta^{(1)})$:

$$\frac{\partial J}{\partial\theta_{j0}^{(1)}}=\delta_j^{(2)}a_0^{(1)}=\delta_j^{(2)}$$

because $\(a_0^{(1)}=1\)$.

---

### Step 5: Collect the Calculated Gradients

After the backward pass, the required gradients are:

$$\boxed{\frac{\partial J}{\partial\Theta^{(2)}}=\delta^{(3)}\left(\ a^{(2)}\right)^T}$$

and:

$$\boxed{\frac{\partial J}{\partial\Theta^{(1)}}=\delta^{(2)}\left(\ a^{(1)}\right)^T}$$

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/d4c83fbb-374e-4a6f-9b58-9132afc0f295"
    width="80%"
    alt="Summary of backward propagation and gradient calculation">
</p>

<p align="center">
  <em>
    Backward propagation applies the chain rule from right to left,
    producing one gradient matrix for each weight matrix in the network.
  </em>
</p>

The complete calculation for one training example can be summarised as:

$$\delta^{(3)}=\frac{\partial J}{\partial a^{(3)}}\odot g'\left(z^{(3)}\right)$$

$$\delta^{(2)}=\left(\Theta_{\mathrm{nb}}^{(2)}\right)^T\delta^{(3)}\odot g'\left(z^{(2)}\right)$$

$$\frac{\partial J}{\partial\Theta^{(2)}}=\delta^{(3)}\left(\ a^{(2)}\right)^T$$

$$\frac{\partial J}{\partial\Theta^{(1)}}=\delta^{(2)}\left(\ a^{(1)}\right)^T$$

### Why There Is No $\(\delta^{(1)}\)$

An error term is not normally calculated for the input layer because its values are supplied features rather than trainable neuron activations.

Backward propagation stops after calculating the gradients of $\(\Theta^{(1)}\)$.

## 12. The Backpropagation Algorithm

Given a training set containing $m$ examples:

$$\mathcal{D}=
\lbrace
(x^{(i)},y^{(i)})
\mid
i=1,2,\ldots,m
\rbrace
$$

the network processes each input, compares its prediction with the corresponding target, calculates the required gradients, and adjusts its parameters to reduce the prediction error.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/24aa2c6d-f30f-4175-8ef2-64d691481576"
    alt="Complete neural-network training process using backpropagation">
</p>

<p align="center">
  <em>
    Overview of neural-network training: perform forward propagation,
    measure the output error, propagate the error backwards, calculate
    the gradients, and use an optimizer to improve the parameters.
  </em>
</p>

The complete training process is described below.

---

### Step 1: Initialize the Parameters

Before training begins, initialize each weight matrix $\(\Theta^{(l)}\)$ using small random values.

Random initialization prevents neurons in the same layer from learning identical features.

$$
\Theta^{(l)}
\leftarrow
\text{small random values}
$$

The gradient accumulators are also initialized to zero:

$$
\Delta^{(l)}=0
$$

> [!IMPORTANT]
> Initializing every weight to zero would cause neurons in the same
> layer to remain identical during training. This is known as the
> symmetry problem.

---

### Step 2: Supply a Training Example

For each training example $\(i\)$, provide the input features $\(x^{(i)}\)$ and their corresponding target $\(y^{(i)}\)$:

$$
\left(x^{(i)},y^{(i)}\right),
\qquad i=1,2,\ldots,m
$$

The input $\(x^{(i)}\)$ is passed through the network, while $\(y^{(i)}\)$ is retained for comparison with the final prediction.

---

### Step 3: Perform Forward Propagation

Starting from the input layer, calculate the weighted inputs and activations of each following layer:

$$z^{(l+1)}=\Theta^{(l)}\tilde a^{(l)}$$

$$a^{(l+1)}=g\left(z^{(l+1)}\right)$$

The tilde indicates that the activation vector includes its bias unit.

This process continues from left to right until the final prediction is obtained:

$$a^{(L)}=h_\Theta\left(x^{(i)}\right)$$

Forward propagation answers the question:

> Given the current weights, what does the network predict?

---

### Step 4: Calculate the Output Error

Compare the predicted output $\(a^{(L)}\)$ with the expected target $\(y^{(i)}\)$.

For sigmoid activation combined with cross-entropy loss:

$$\delta^{(L)}=a^{(L)}-y^{(i)}$$

The output error indicates how far the prediction is from the expected result and provides the starting point for the backward pass.

The cost function $\(J(\Theta)\)$ provides an overall numerical measure of the prediction error.

---

### Step 5: Backpropagate the Error

Propagate the output error backwards through the hidden layers.

For a hidden layer $\(l\)$:

$$\delta^{(l)}=\left(\Theta_{\mathrm{nb}}^{(l)}\right)^T\delta^{(l+1)}\odot g'\left(z^{(l)}\right)$$

Here, $\(\Theta_{\mathrm{nb}}^{(l)}\)$ represents the weight matrix with its bias column excluded.

This calculation determines how much each hidden neuron contributed to the final prediction error.

Backpropagation answers the question:

> Which neurons and connections were responsible for the error?

---

### Step 6: Calculate and Accumulate the Gradients

For each weight matrix, calculate:

$$\frac{\partial J^{(i)}}{\partial\Theta^{(l)}}=\delta^{(l+1)}\left(\tilde a^{(l)}\right)^T$$

Accumulate the gradients across the training examples:

$$\Delta^{(l)}:=\Delta^{(l)}+\delta^{(l+1)}\left(\tilde a^{(l)}\right)^T$$

After all \(m\) training examples have been processed, calculate the average gradient:

$$\frac{\partial J}{\partial\Theta^{(l)}}=\frac{1}{m}\Delta^{(l)}$$

The gradient describes the direction and sensitivity of the cost with respect to each weight.

---

### Step 7: Update the Weights

An optimization algorithm uses the calculated gradients to adjust the weights.

Using gradient descent:

$$\Theta^{(l)}:=\Theta^{(l)}-\alpha\frac{\partial J}{\partial\Theta^{(l)}}$$

where $\(\alpha\)$ is the learning rate.

The update moves the weights in the direction that reduces the cost:

- A positive gradient causes the weight to decrease.
- A negative gradient causes the weight to increase.
- A larger gradient produces a larger adjustment.
- The learning rate controls the overall update size.

> [!NOTE]
> Backpropagation calculates the gradients. Gradient descent uses those
> gradients to update the weights.

---

### Step 8: Repeat Until Convergence

Repeat the forward pass, backward pass, and parameter update over multiple epochs:

```text
Random Initialization
         ↓
Supply Training Examples
         ↓
Forward Propagation
         ↓
Compute Cost and Output Error
         ↓
Backward Propagation
         ↓
Calculate Gradients
         ↓
Update Parameters
         ↓
Repeat Until Convergence
```

One epoch is completed when the network has processed the entire training set once.

Training continues until a suitable stopping condition is reached, such as:

- The cost stops decreasing significantly
- The validation performance stops improving
- A specified number of epochs is completed
- The gradients become sufficiently small

With repeated training, the network gradually learns parameter values that produce predictions closer to the expected targets.

## 13. Backpropagation Intuition

### Forward: Combine Information

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0945a47-4629-42ee-af87-c661fab5facf" width=600>
</p>

During forward propagation, a neuron computes a weighted sum of activations from the left:

$$
z_j^{(l+1)}=\theta_{j0}^{(l)}a_0^{(l)}+\theta_{j1}^{(l)}a_1^{(l)}+\cdots+\theta_{js_l}^{(l)}a_{s_l}^{(l)}
$$

It asks:

> Given the current weights, what should this neuron output?

### Backward: Assign Responsibility

<p align="center">
  <img src="https://github.com/user-attachments/assets/85fd0335-3e8c-499c-bde8-404e0dbb4ec2" width=600>
</p>

Backward propagation calculates the error terms from the output layer towards the input layer.

The error term of neuron $(j)$ in layer $(l)$ is defined as:

$$\boxed{\delta_j^{(l)}=\frac{\partial J}{\partial z_j^{(l)}}}$$

This measures how sensitive the final cost $J$ is to a change in the neuron's weighted input $(z_j^{(l)})$.

In simple terms, it asks:

> How much did this neuron contribute to the final prediction error?

A large value of $(\left|\delta_j^{(l)}\right|)$ means that changing the neuron would have a relatively large effect on the final cost.

---

#### How a Hidden Neuron Receives Its Error

A hidden neuron does not have a target value that can be directly compared with its activation. Its error must therefore be calculated from the neurons in the following layer.

For neuron $j$ in layer $l$, first combine the error terms of all neurons in layer $l+1$ that receive its activation:

$$
\sum_{k=1}^{s_{l+1}}
\theta_{kj}^{(l)}
\delta_k^{(l+1)}
$$

The hidden neuron's error is then adjusted by its local activation derivative:

$$
\boxed{
\delta_j^{(l)}=
\left(
\sum_{k=1}^{s_{l+1}}
\theta_{kj}^{(l)}
\delta_k^{(l+1)}
\right)
g'\left(z_j^{(l)}\right)
}
$$

This calculation contains two parts:

| Component | Meaning |
|:---:|---|
| $\displaystyle \sum_{k=1}^{s_{l+1}}\theta_{kj}^{(l)}\delta_k^{(l+1)}$ | Error received from the following layer |
| $g'\left(z_j^{(l)}\right)$ | Sensitivity of the current neuron |
| $\delta_j^{(l)}$ | Responsibility assigned to the current neuron |

A hidden neuron receives greater responsibility when:

1. It is connected to a downstream neuron with a large error.
2. The connecting weight is large.
3. Its activation function is sensitive at the current value of $(z_j^{(l)})$.

---

#### Equivalent Matrix Form

To write the same three calculations compactly, first remove the bias weight from $(\Theta^{(2)})$.

The complete output weight matrix is:

$$
\Theta^{(2)}=
\begin{bmatrix}
\theta_{10}^{(2)}
&
\theta_{11}^{(2)}
&
\theta_{12}^{(2)}
&
\theta_{13}^{(2)}
\end{bmatrix}
$$

After excluding the bias column:

$$
\overline{\Theta}^{(2)}=
\begin{bmatrix}
\theta_{11}^{(2)}
&
\theta_{12}^{(2)}
&
\theta_{13}^{(2)}
\end{bmatrix}
$$

The hidden-layer error vector is then:

$$
\boxed{
\delta^{(2)}=
\left(\overline{\Theta}^{(2)}\right)^T
\delta^{(3)}
\odot
g'\left(z^{(2)}\right)
}
$$

where $(\odot\)$ represents element-wise multiplication.

The dimensions are:

$$
\underbrace{\delta^{(2)}}_{3\times1}=
\left(
\underbrace{
\left(\overline{\Theta}^{(2)}\right)^T
}_{3\times1}
\underbrace{\delta^{(3)}}_{1\times1}
\right)
\odot
\underbrace{
g'\left(z^{(2)}\right)
}_{3\times1}
$$

> [!NOTE]
> The scalar and matrix equations describe the same operation. The
> scalar form explains how one hidden neuron receives its error, while
> the matrix form calculates all hidden-neuron errors simultaneously.

### Meaning of a Weight Gradient

$$
\frac{\partial J}{\partial\theta_{jk}^{(l)}}
$$

| Gradient Value | Meaning |
|----------------|---------|
| Large positive | Increasing the weight increases cost strongly; reduce it |
| Large negative | Increasing the weight reduces cost strongly; increase it |
| Near zero | Small local effect on cost |

### Credit Assignment

Backpropagation solves the **credit-assignment problem**: it determines how much credit or blame each weight should receive for the final prediction error.

> **Analogy:** Imagine a group assignment receives a low mark. Backpropagation traces each part of the final submission back to the team member and earlier decision that influenced it, then estimates what should change next time.

---

## 14. Random Weight Initialization

### Why Not Initialize Every Weight to Zero?

If all hidden neurons start with identical weights, they receive the same inputs, compute the same activations, receive the same gradients, and remain identical after every update.

This is called the **symmetry problem**.

### What Goes Wrong

<p align="center">
  <img src="https://github.com/user-attachments/assets/072b6a66-1c5e-40b5-a6f5-0f233b2dec17" width=600>
</p>

Suppose all three hidden neurons have the same weights:

$$
\theta_1=\theta_2=\theta_3
$$

Then:

$$
a_1^{(2)}=a_2^{(2)}=a_3^{(2)}
$$

All three neurons learn the same feature, so the network wastes its hidden units.

### The Solution

Initialize weights to small random values close to zero:

$$
\theta_{jk}^{(l)}\sim\text{small random distribution}
$$

This breaks symmetry, allowing different neurons to learn different features.

| Initialization | Result |
|----------------|--------|
| All zeros | Hidden neurons stay identical ❌ |
| Same constant | Hidden neurons stay identical ❌ |
| Small random values | Hidden neurons can specialize ✅ |

> [!WARNING]
> The goal is not simply “randomness.” The scale also matters: weights that are too large or too small can make training unstable or slow.

> **Analogy:** If every team member receives exactly the same starting instructions and makes every decision identically, the team behaves like one person. Slightly different starting points allow members to specialize.

---

## 15. Logistic Regression vs ANN

| Feature | Logistic Regression | Artificial Neural Network |
|---------|---------------------|---------------------------|
| Architecture | No hidden layer | One or more hidden layers |
| Decision boundary | Best for approximately linear separation | Can learn complex non-linear boundaries |
| Feature learning | Relies mainly on supplied features | Learns intermediate features |
| Data requirement | Often effective on small datasets | Often benefits from more data |
| Computation | Relatively fast and inexpensive | More computationally expensive |
| Interpretability | Easier to interpret | More difficult to interpret |
| Cost surface | Convex for standard logistic regression | Generally non-convex |
| Minimum | Global minimum is reachable for convex cost | May contain multiple local minima or saddle points |
| Best use | Simple, robust classification baseline | Complex patterns such as image data |

### Convex vs Non-Convex Cost

For standard logistic regression, the cross-entropy cost is convex. This means there is one global minimum.

For a neural network, interactions among many layers and weights create a non-convex cost surface. Different initializations may lead to different trained solutions.

<table align="center">
<tr>

<td align="center" width="50%" valign="top">

<img src="https://github.com/user-attachments/assets/62e89cf9-8e0f-40d1-8cac-745c1137ccd5" width="100%" height="230">

<br>
Logistic Regression (Convex cost surface)

</td>

<td align="center" width="50%" valign="top">

<img src="https://github.com/user-attachments/assets/dc2141a3-3112-4f58-b740-be438bbaeb7d" width="100%" height="245">

<br>
Neural Networks (Non-convex cost surface)
</td></tr></table>

### Which One Should You Choose?

| Scenario | Recommended Starting Point |
|----------|----------------------------|
| Small, structured, approximately linearly separable dataset | Logistic Regression |
| Need a simple and interpretable baseline | Logistic Regression |
| Image or other high-dimensional pattern data | ANN |
| Strong non-linear relationships | ANN |
| Large dataset and sufficient computational resources | ANN |

> [!TIP]
> Start with logistic regression as a baseline when it is reasonable. Use an ANN when the simpler model cannot capture the required non-linear structure.

> [!IMPORTANT]
> Always check matrix dimensions before doing the arithmetic. Most hand-calculation errors in forward propagation come from forgetting the bias term or reversing the weight-matrix dimensions.

---

## 16. Key Takeaways

> [!IMPORTANT]
> **The 15 things to remember from Week 4:**

1. Real-world classification problems often require a **non-linear hypothesis**
2. Polynomial expansion becomes impractical for high-dimensional data such as images
3. An artificial neuron computes a weighted sum and applies an **activation function**
4. Logistic regression is equivalent to a neural network with **no hidden layer**
5. Hidden layers automatically learn new representations of the original features
6. The three main layer types are **input, hidden, and output**
7. A bias unit has a fixed activation of 1 and shifts a neuron's threshold
8. The weight matrix dimension is $s_{l+1}\times(s_l+1)$
9. **Forward propagation** moves left to right to compute $h_\Theta(x)$
10. Multiclass ANN targets use one-hot vectors and normally one output neuron per class
11. **Backpropagation** moves right to left and applies the chain rule to compute gradients
12. For sigmoid plus cross-entropy, the output error is $\delta^{(L)}=a^{(L)}-y$
13. Backpropagation computes gradients; gradient descent uses them to update weights
14. Weights must be randomly initialized to break symmetry between hidden neurons
15. ANN models learn complex non-linear patterns, but their optimization problem is generally non-convex

---

## 17. Glossary

| Term | Definition |
|------|-----------|
| **Activation** | The output of a neuron after its activation function is applied |
| **Activation Function** | A function such as sigmoid that transforms a neuron's weighted input and introduces non-linearity |
| **Artificial Neural Network (ANN)** | A model made of connected layers of artificial neurons |
| **Backpropagation** | An efficient chain-rule algorithm that computes cost gradients from the output layer backwards |
| **Bias Unit** | A fixed input of 1 that allows a neuron to shift its activation threshold |
| **Chain Rule** | Calculus rule used to differentiate a sequence of dependent functions |
| **Cost Function** | A measure of how far model predictions are from their targets |
| **Deep Neural Network** | A neural network containing multiple hidden layers |
| **Dense Layer** | A layer in which every input is connected to every neuron |
| **Epoch** | One complete pass through the training dataset |
| **Feature Learning** | Automatically learning useful intermediate representations from data |
| **Forward Propagation** | Computing activations from the input layer to the output layer |
| **Gradient** | The derivative of the cost with respect to a parameter |
| **Hidden Layer** | A layer between the input and output that learns intermediate features |
| **Learning Rate** | Step size $\alpha$ used when updating weights |
| **Non-Convex Function** | A function whose cost surface can contain several local minima and saddle points |
| **One-Hot Encoding** | A target vector with 1 at the correct class position and 0 elsewhere |
| **Output Layer** | The final layer that produces the network prediction |
| **Random Initialization** | Starting weights at small random values to break neuron symmetry |
| **Symmetry Problem** | Identically initialized neurons learn identical features and remain redundant |
| **Weight** | A trainable parameter controlling the strength of a connection |
| **Weight Matrix** | A matrix containing all weights connecting one layer to the next |
| **Weighted Sum** | The linear combination $z=\theta^Ta$ computed before activation |

---

> [!TIP]
> **Study tip for Week 4:** Make sure you can:
> 1. Explain why polynomial features become impractical for image classification
> 2. Label the input, hidden, output, and bias units in a network diagram
> 3. Determine the dimension of $\Theta^{(l)}$ from two layer sizes
> 4. Perform one complete forward-propagation calculation by hand
> 5. Convert a multiclass label into a one-hot vector
> 6. Explain forward propagation and backpropagation in one sentence each
> 7. State the output and hidden-layer error equations
> 8. Explain why zero initialization fails
> 9. Compare logistic regression with ANN and select an appropriate model

