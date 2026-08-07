# 📘 Week 4 Part 2 Study Notes: Support Vector Machine (SVM)

> **Course:** COS30082 — Applied Machine Learning  
> **Topic:** Support Vector Machine (SVM)  
> **Prerequisite:** Week 3 (Logistic Regression), Week 4 Part 1 (Artificial Neural Networks)

---

## Table of Contents

1. [What Is a Support Vector Machine?](#1-what-is-a-support-vector-machine)
2. [Why a Large-Margin Classifier?](#2-why-a-large-margin-classifier)
3. [Support Vectors](#3-support-vectors)
4. [From Logistic Regression to the SVM Cost Function](#4-from-logistic-regression-to-the-svm-cost-function)
5. [The Hinge Loss](#5-the-hinge-loss)
6. [The SVM Hypothesis](#6-the-svm-hypothesis)
7. [Math Intuition Behind the Margin](#7-math-intuition-behind-the-margin)
8. [The Regularized SVM Cost Function](#8-the-regularized-svm-cost-function)
9. [The Regularization Parameter C](#9-the-regularization-parameter-c)
10. [SVM vs Logistic Regression Terminology](#10-svm-vs-logistic-regression-terminology)
11. [Non-Linear SVM and the Need for Kernels](#11-non-linear-svm-and-the-need-for-kernels)
12. [The Kernel Trick and Similarity Functions](#12-the-kernel-trick-and-similarity-functions)
13. [Landmarks](#13-landmarks)
14. [The Gaussian (RBF) Kernel](#14-the-gaussian-rbf-kernel)
15. [Worked Example: Classifying with Kernels](#15-worked-example-classifying-with-kernels)
16. [The Influence of σ²](#16-the-influence-of-σ²)
17. [SVM vs Logistic Regression: When to Use Each](#17-svm-vs-logistic-regression-when-to-use-each)
18. [SVM vs ANN](#18-svm-vs-ann)
19. [Key Takeaways](#19-key-takeaways)
20. [Glossary](#20-glossary)

---

## 1. What Is a Support Vector Machine?

### A Supervised Classification Algorithm

A **Support Vector Machine (SVM)** is a supervised machine learning algorithm best known for solving **classification** problems.

SVM is also called a **large-margin classifier**, because — unlike many other classifiers — it does not just look for *a* decision boundary that separates the classes. It looks for the boundary that separates them **as widely as possible**.

<div align="center">

```text
Training Data → SVM Optimization → Maximum-Margin Decision Boundary
```

</div>

> **Analogy:** Imagine drawing a road between two neighbourhoods so that no house on either side gets too close to the road. SVM doesn't just draw *any* road — it finds the widest possible road that still keeps both neighbourhoods on their own side.

---

## 2. Why a Large-Margin Classifier?

### The Obese / Non-Obese Example

Suppose we are given $N$ training samples, and we want to classify a person as **obese** or **non-obese** ($y$) based on their **height** ($x_1$) and **weight** ($x_2$).

Several different straight lines could separate the two classes:

<p align="center">
  <img src="https://github.com/user-attachments/assets/eee09a21-f7e7-4ece-8fd8-7dddbd931203" width=600>
</p>

Any of these lines technically separates the two groups, but they are not all equally good. Some lines pass very close to a few data points, which makes the classifier more fragile to new, unseen data.

### Distance to the Threshold

If we simplify the problem to a single dimension (a number line), the boundary (threshold) between the two classes can be placed anywhere between the closest obese and non-obese points. However, the **best** threshold is the one that is **equidistant** from the nearest point of each class:

$$
\text{distance(threshold, nearest obese point)} = \text{distance(threshold, nearest non-obese point)}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/76c4ab88-f9b7-4c86-9d04-9e5e015143ab" width=600>
</p>

This equal distance on both sides is what we call the **margin**.

### The Answer: Maximum Margin

The objective of the SVM algorithm is to find a decision boundary that has the **maximum margin** — that is, the maximum distance from the data point that is closest to the decision boundary.

$$
\boxed{\text{SVM objective: maximize the margin between the decision boundary and the nearest data points}}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/5069a388-722c-48c0-8949-b1c1c9a6663d" width=600>
</p>

> [!IMPORTANT]
> Out of infinitely many lines that could separate two linearly separable classes, SVM specifically selects the one that maximizes the distance to the nearest data point on either side.

---

## 3. Support Vectors

### Definition

A **support vector** is a data point that is either:

- **Incorrectly classified**, or
- **Close to the decision boundary** (i.e. sitting on or near the margin)

$$
\boxed{\text{Support vectors are the data points that define the hyperplane}}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/58448ed1-8b11-4705-9eda-0a4a062e5e79" width=600>
</p>

| Component | Description |
|-----------|-------------|
| **Decision boundary** | The hyperplane that separates the two classes |
| **Margin** | The region between the decision boundary and the two parallel boundary lines through the nearest points |
| **Support vectors** | The data points that lie exactly on the margin (or violate it) |

> [!NOTE]
> Data points that are far away from the decision boundary and correctly classified have **no influence** on where the boundary is placed. Only the support vectors matter — this is what makes SVM memory-efficient at prediction time.

> **Analogy:** Think of the margin as two people pulling a rope taut between the two groups of points. Only the points that are actually touching the rope (the support vectors) determine how tight and where the rope sits — everyone standing further back doesn't affect the rope at all.

---

## 4. From Logistic Regression to the SVM Cost Function

### Recall: Logistic Regression's Cost Function

Recall the sigmoid hypothesis:

$$
z(x)=\theta^Tx \qquad h_\theta(x)=g(z)=\frac{1}{1+e^{-z}}
$$

$$
\text{predict } y=1 \text{ if } h_\theta(x)\geq0.5 \;\Longleftrightarrow\; \theta^Tx\geq0
$$

$$
\text{predict } y=0 \text{ if } h_\theta(x)<0.5 \;\Longleftrightarrow\; \theta^Tx<0
$$


| | $y=1$ | $y=0$ |
|---|---|---|
| **Equation** | $$cost(x,y) = -\log\left(\dfrac{1}{1+e^{-(\theta^Tx)}}\right)$$ | $$cost(x,y) = -\log\left(1-\dfrac{1}{1+e^{-(\theta^Tx)}}\right)$$ |
| **Graph** | <p align="center"><img src="https://github.com/user-attachments/assets/57607697-1094-4535-b99e-ce90a3bb7cf1" width=600></p> | <p align="center"><img src="https://github.com/user-attachments/assets/aa49e836-1aa7-42c0-b40f-499f305e5fa7" width=600></p> |
| **Interpretation** | For $y=1$, we want $\theta^Tx \geq 0$ | For $y=0$, we want $\theta^Tx < 0$ |
---

## 5. The Hinge Loss

### SVM's Cost Function

SVM's cost function is very similar in shape to logistic regression's, but it replaces the smooth logarithmic curve with two straight-line segments. This is called the **hinge loss**.

$$
\text{if } y=1:\quad cost_1(\theta^Tx)=\max\left(0,\;1-\theta^Tx\right)
$$

$$
\text{if } y=0:\quad cost_0(\theta^Tx)=\max\left(0,\;1+\theta^Tx\right)
$$

<div align="center">

```text
cost₁(θᵀx)                         cost₀(θᵀx)
   │╲                                  │       ╱
   │ ╲                                 │      ╱
   │  ╲___________ θᵀx                 │_____╱______ θᵀx
        1                                  -1
```

</div>

| Function | Zero when | Behaviour |
|----------|-----------|-----------|
| $cost_1(\theta^Tx)$ | $\theta^Tx \geq 1$ | Linearly increasing penalty as $\theta^Tx$ falls below 1 |
| $cost_0(\theta^Tx)$ | $\theta^Tx \leq -1$ | Linearly increasing penalty as $\theta^Tx$ rises above $-1$ |

> [!NOTE]
> The hinge loss is piecewise-linear rather than smooth, which is part of what gives SVM its large-margin behaviour — it stops rewarding a prediction the moment it is "confidently correct" (past the margin), instead of continuing to reward it all the way to infinity like logistic regression's log-loss does.

---

## 6. The SVM Hypothesis

$$
h_\theta(x)=
\begin{cases}
1 & \text{if } \theta^Tx\geq0\\
0 & \text{otherwise}
\end{cases}
$$

However, for the **loss interpretation** used during training, SVM demands a stricter margin condition than logistic regression:

$$
cost_1(\theta^Tx)=0 \quad\text{only when}\quad \theta^Tx\geq1
$$

$$
cost_0(\theta^Tx)=0 \quad\text{only when}\quad \theta^Tx\leq-1
$$

Hence:

| Class | Logistic Regression requires | SVM requires |
|-------|-------------------------------|--------------|
| $y=1$ | $\theta^Tx\geq0$ | $\theta^Tx\geq1$ |
| $y=0$ | $\theta^Tx<0$ | $\theta^Tx\leq-1$ |

> [!IMPORTANT]
> SVM builds in an extra **safety margin**. It is not satisfied with a data point just barely crossing the boundary — it wants the point to be at least 1 unit away (in the $\theta^Tx$ sense) on the correct side. This built-in buffer is exactly what produces the large margin.

---

## 7. Math Intuition Behind the Margin

### Three Parallel Hyperplanes

Assume three hyperplanes:

- **Decision boundary:** $\theta^Tx=0$
- **Margin boundaries:** $\theta^Tx=\pm1$

<div align="center">

```text
x₁
 │           θᵀx = 1  (positive margin)
 │          ⋰
 │        ⋰   θᵀx = 0  (decision boundary)
 │      ⋰
 │    ⋰      θᵀx = -1 (negative margin)
 └───────────────────────── x₂
```

</div>

- **Positive domain:** when a point sits exactly on the margin, $\theta^Tx=1$. When it sits between the boundary and the margin, $0<\theta^Tx<1$.
- **Negative domain:** when a point sits exactly on the margin, $\theta^Tx=-1$. When it sits between the boundary and the margin, $-1<\theta^Tx<0$.

For a point $x$ belonging to class $y=1$, the ideal condition is:

$$
y(\theta^Tx)\geq1
$$

This equation states that the product of the actual label and the hyperplane output must be at least 1 — meaning the point is correctly classified **and** sits outside (or on) the margin.

### Worked Numerical Walkthrough

Let $\theta_0=-1,\ \theta_1=1,\ \theta_2=-1$, so that:

$$
\theta^Tx=\theta_0+\theta_1x_1+\theta_2x_2
$$

For a series of candidate points with $y=1$:

| Point $x=(x_1,x_2)$ | $\theta^Tx$ calculation | $\theta^Tx$ | Interpretation |
|---|---|---|---|
| $(6,\,2)$ | $-1+1(6)-1(2)$ | $3$ | Well past the margin → $cost_1=0$ |
| $(3,\,1)$ | $-1+1(3)-1(1)$ | $1$ | Exactly on the margin → $cost_1=0$ |
| $(2.5,\,1)$ | $-1+1(2.5)-1(1)$ | $0.5$ | Between boundary and margin → $cost_1\neq0$ |
| $(2,\,1)$ | $-1+1(2)-1(1)$ | $0$ | Exactly on the decision boundary → $cost_1\neq0$ |
| $(2,\,2.5)$ | $-1+1(2)-1(2.5)$ | $-1.5$ | On the wrong side entirely → large $cost_1$ |

> [!NOTE]
> Only the points with $\theta^Tx<1$ (the last three rows) incur a penalty. The reason the cost only becomes zero from $\theta^Tx=1$ onward — instead of $\theta^Tx=0$ — is precisely to **penalize points that are correctly classified but too close to the boundary**, forcing the algorithm to prefer a wider margin.

### Why This Creates a Large Margin

- The decision boundary is **unaffected** by points that are correctly classified and far from it.
- The cost only reacts to points that are misclassified **or** close to the boundary.
- Minimizing this cost therefore naturally pushes the boundary away from the nearest points on both sides — producing the maximum-margin hyperplane.

---

## 8. The Regularized SVM Cost Function

Putting the hinge loss together with a regularization term:

$$
\min_\theta J(\theta) = C\left[\sum_{n=1}^{N}y^{(n)}cost_1\!\left(\theta^Tx^{(n)}\right)+\left(1-y^{(n)}\right)cost_0\!\left(\theta^Tx^{(n)}\right)\right]+\frac{1}{2}\sum_{j=1}^{M}\theta_j^2
$$

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of training data points |
| $M$ | Number of parameters |
| $C$ | Regularization parameter for SVM |
| $\frac{1}{2}\sum\theta_j^2$ | Regularization term that controls the size of the weights |

---

## 9. The Regularization Parameter C

### Large C Effect

If $C$ is **very large**, minimizing $J(\theta)$ effectively forces the hinge-loss term toward zero (since it is multiplied by such a large constant), leaving:

$$
\min_\theta J(\theta) = \min_\theta \underbrace{C[\cdots]}_{\text{very large}} + \underbrace{\frac{1}{2}\sum_{j=1}^{M}\theta_j^2}_{0}
\;\;\Longrightarrow\;\;
\min_\theta J(\theta) \approx \min_\theta \frac{1}{2}\sum_{j=1}^{M}\theta_j^2
$$

The mathematical intuition behind $\min_\theta \frac{1}{2}\sum_{j=1}^{M}\theta_j^2$ is to let the SVM choose the decision boundary that results in the **large-margin classification effect**.

However, a model with a large $C$ value is **sensitive to outliers**, similar to having no regularization at all. Such a model is prone to **overfitting**.

| Large C | Effect |
|---------|--------|
| Effect of noisy points | Large |
| Precedence given to | A plane with very few misclassifications |
| Bias / Variance | Lower bias, higher variance |

### Small C Effect

A smaller $C$ allows a bigger margin for model regularization. It is useful for **non-linearly separable** datasets, tolerating some data points that are **misclassified** or that have **margin violation**.

However, a model with a **too-small** $C$ value is prone to **underfitting**.

| Small C | Effect |
|---------|--------|
| Effect of noisy points | Low |
| Precedence given to | Planes that separate the points well, even with some misclassifications |
| Bias / Variance | Higher bias, lower variance |

> [!WARNING]
> Choosing $C$ is a bias–variance trade-off, just like choosing $\lambda$ in regularized logistic regression. Too large risks overfitting to outliers; too small risks underfitting the true pattern.

---

## 10. SVM vs Logistic Regression Terminology

Both models share a very similar cost function structure — only the loss term and the position of the regularization constant differ.

**Logistic Regression:**

$$
\min_\theta J(\theta) = \frac{1}{N}\left[\sum_{n=1}^{N}y^{(n)}\{-\log h_\theta(x^{(n)})\} + (1-y^{(n)})\{-\log(1-h_\theta(x^{(n)}))\} \right] + \frac{\lambda}{2N}\sum_{j=1}^{M}\theta_j^2
$$

**SVM:**

$$
\min_\theta J(\theta) = C\left[\sum_{n=1}^{N}y^{(n)}cost_1\!\left(\theta^Tx^{(n)}\right)+\left(1-y^{(n)}\right)cost_0\!\left(\theta^Tx^{(n)}\right)\right]+\frac{1}{2}\sum_{j=1}^{M}\theta_j^2
$$

where $M$ = number of parameters and $N$ = number of data points (training data).

$$
\boxed{C \text{ plays a role similar to } \frac{1}{\lambda}}
$$

| Aspect | Logistic Regression | SVM |
|--------|---------------------|-----|
| Loss term | Log-loss (cross-entropy) | Hinge loss |
| Regularization placement | Multiplies the regularization term ($\lambda$) | Multiplies the loss term ($C$) |
| Larger regularization strength | Smaller $\lambda$ | Larger $C$ |

---

## 11. Non-Linear SVM and the Need for Kernels

### When a Straight Line Isn't Enough

The simplest way to separate two groups of data is with a straight line (1 dimension), a flat plane (2 dimensions), or an N-dimensional hyperplane.

However, there are situations where a **nonlinear region** can separate the groups far more efficiently — for example, when one class is surrounded by the other:

<div align="center">

```text
x₁
 │      ● ●
 │   ●  ╭───╮   ●
 │     │ ○ ○ │
 │   ● │○   ○│  ●
 │      ╰───╯
 └───────────────── x₂
        ● class 1     ○ class 2
```

</div>

In this case, a **non-linear decision boundary** is necessary to correctly separate the data points.

---

## 12. The Kernel Trick and Similarity Functions

### Hypothesis and Cost Function with a Kernel

The hypothesis is the same as the linear SVM, except that the input data $x$ is replaced with a function $f$:

$$
h_\theta(x)=
\begin{cases}
1 & \text{if } \theta^Tf(x)\geq0\\
0 & \text{otherwise}
\end{cases}
$$

The cost function is updated the same way:

$$
J(\theta) = C\left[\sum_{n=1}^{N}y^{(n)}cost_1\!\left(\theta^Tf(x^{(n)})\right)+\left(1-y^{(n)}\right)cost_0\!\left(\theta^Tf(x^{(n)})\right)\right]+\frac{1}{2}\sum_{j=1}^{M}\theta_j^2
$$

### What Is $f$?

$f$ is a **similarity function**, called a **kernel function**. It is used to map the data into a different (usually higher-dimensional) space when a hyperplane (linear boundary) cannot separate the data in the original space.

A non-linear function is effectively learned by a linear learning machine operating in this new, high-dimensional feature space, while the capacity of the system is controlled by a parameter that does **not** depend on the dimensionality of that space.

This is called the **kernel trick**: the kernel function transforms the data into a higher-dimensional feature space, making it possible to perform **linear** separation there — even though the original data was not linearly separable.

<div align="center">

```text
Original space (x₁, x₂)        Transformed space (x₁', x₂')
   curved boundary        →        straight boundary
```

</div>

> **Analogy:** Imagine trying to separate red and blue marbles that are arranged in two concentric circles on a flat table — no straight line works. But if you lift the inner circle of marbles up into the air (adding a third dimension), a flat sheet of glass can now separate them perfectly. The kernel trick is like finding that extra dimension mathematically, without physically computing every new coordinate.

---

## 13. Landmarks

### Computing New Features from Proximity

The kernel function computes a new feature of $x$ based on its **proximity to landmarks**:

$$
f_1 = \text{similarity}(x,l^{(1)}) \text{ or } k(x,l^{(1)})
$$

$$
f_2 = \text{similarity}(x,l^{(2)}) \text{ or } k(x,l^{(2)})
$$

$$
f_3 = \text{similarity}(x,l^{(3)}) \text{ or } k(x,l^{(3)})
$$

### How Are Landmarks Chosen?

Given $n$ training samples, the location of the landmarks is chosen to be **exactly the location of the $n$ training samples**:

$$
\text{Training samples: } (x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\ldots,(x^{(n)},y^{(n)})
$$

$$
\text{Landmarks chosen: } l^{(1)}=x^{(1)},\ l^{(2)}=x^{(2)},\ \ldots,\ l^{(n)}=x^{(n)}
$$

For one sample $(x^{(i)},y^{(i)})$, new features are created by comparing that sample against **every** landmark:

$$
f^{(i)}(x^{(i)}) =
\begin{bmatrix}
f_1^{(i)} \\ f_2^{(i)} \\ f_3^{(i)} \\ \vdots \\ f_n^{(i)}
\end{bmatrix}
\quad\text{where } x^{(i)}=l^{(i)} \Rightarrow f^{(i)}=1
$$

| Property | Description |
|----------|-------------|
| Each landmark | Defines one new feature |
| Number of new features for one sample | Equal to the number of training samples $n$ |
| Feature value when $x=l$ | Exactly $1$ (maximum similarity) |

---

## 14. The Gaussian (RBF) Kernel

The Gaussian kernel is one of the best-known kernels, also called the **Radial Basis Function (RBF)** kernel:

$$
f(x)=\exp\left(-\frac{\|x-l\|^2}{2\sigma^2}\right)
$$

where $\|x-l\|^2$ is the squared **Euclidean distance** between $x$ and the landmark $l$.

For multiple landmarks:

$$
f_1 = \exp\left(-\frac{\|x-l^{(1)}\|^2}{2\sigma^2}\right) \text{ or } \exp\left(-\frac{\sum_{j=1}^{M}(x_j-l_j^{(1)})^2}{2\sigma^2}\right)
$$

$$
f_2 = \exp\left(-\frac{\|x-l^{(2)}\|^2}{2\sigma^2}\right) \text{ or } \exp\left(-\frac{\sum_{j=1}^{M}(x_j-l_j^{(2)})^2}{2\sigma^2}\right)
$$

$$
f_3 = \exp\left(-\frac{\|x-l^{(3)}\|^2}{2\sigma^2}\right) \text{ or } \exp\left(-\frac{\sum_{j=1}^{M}(x_j-l_j^{(3)})^2}{2\sigma^2}\right)
$$

### The Core Idea

$$
\text{If } x \approx l \;\Longrightarrow\; f = \exp\left(-\frac{\|0\|^2}{2\sigma^2}\right)\approx1
$$

$$
\text{If } x \text{ is far from } l \;\Longrightarrow\; f = \exp\left(-\frac{\|\text{large number}\|^2}{2\sigma^2}\right)\approx0
$$

| Distance between $x$ and $l$ | Resulting feature value |
|-------------------------------|--------------------------|
| Very small (close) | $f \approx 1$ |
| Very large (far) | $f \approx 0$ |

---

## 15. Worked Example: Classifying with Kernels

Given three landmarks $l^{(1)},l^{(2)},l^{(3)}$ and learned parameters:

$$
\theta_0=-0.3,\quad \theta_1=1,\quad \theta_2=1,\quad \theta_3=0
$$

The prediction rule is:

$$
\theta^Tf(x)=\theta_0+\theta_1f_1(x)+\theta_2f_2(x)+\theta_3f_3(x)
$$

The model predicts $h_\theta(x)=1$ when $\theta^Tf(x)\geq0$.

### Case 1: $x$ is right on top of landmark $l^{(1)}$

$$
f_1(x)=1,\quad f_2(x)=0,\quad f_3(x)=0
$$

$$
\theta^Tf(x) = -0.3+1(1)+0+0 = 0.7 \geq 0 \;\Longrightarrow\; h_\theta(x)=1
$$

### Case 2: $x$ is right on top of landmark $l^{(2)}$

$$
f_1(x)=0,\quad f_2(x)=1,\quad f_3(x)=0
$$

$$
\theta^Tf(x) = -0.3+0+1(1)+0 = 0.7 \geq 0 \;\Longrightarrow\; h_\theta(x)=1
$$

### Case 3: $x$ is right on top of landmark $l^{(3)}$

$$
f_1(x)=0,\quad f_2(x)=0,\quad f_3(x)=1
$$

$$
\theta^Tf(x) = -0.3+0+0+0(1) = -0.3 < 0 \;\Longrightarrow\; h_\theta(x)=0
$$

### Summary Table

| Position of $x$ | $f_1$ | $f_2$ | $f_3$ | $\theta^Tf(x)$ | Prediction |
|---|---|---|---|---|---|
| Near $l^{(1)}$ | 1 | 0 | 0 | $0.7$ | $h_\theta(x)=1$ |
| Near $l^{(2)}$ | 0 | 1 | 0 | $0.7$ | $h_\theta(x)=1$ |
| Near $l^{(3)}$ | 0 | 0 | 1 | $-0.3$ | $h_\theta(x)=0$ |

### Resulting Decision Region

With these learned parameters, any data point $x$ **near landmarks $l^{(1)}$ and $l^{(2)}$** will be predicted as class 1; otherwise it is predicted as class 0. This naturally traces out a **non-linear** decision boundary around the two relevant landmarks:

<div align="center">

```text
x₁
 │        h_θ(x) = 1
 │         ╭────╮
 │        │  ●l⁽²⁾│
 │        │╭──╮   │
 │        │●l⁽¹⁾   │      h_θ(x) = 0
 │         ╰────╯       ●l⁽³⁾
 └──────────────────────────── x₂
```

</div>

---

## 16. The Influence of σ²

$$
f(x)=\exp\left(-\frac{\|x-l\|^2}{2\sigma^2}\right)
$$

Besides the regularization term $C$, we can also adjust $\sigma^2$ from the Gaussian kernel to find the balance between **bias and variance**.

With a **fixed** distance between $x$ and $l$:

| $\sigma^2$ | Effect on similarity $f_j$ | Behaviour |
|---|---|---|
| **Large** $\sigma^2$ | Makes $x$ and $l$ appear "closer" ($f_j\approx1$ over a wider range) | $f_j$ varies **more smoothly** → higher bias, lower variance |
| **Small** $\sigma^2$ | Makes $x$ and $l$ appear "further apart" ($f_j\approx0$ quickly) | $f_j$ varies **less smoothly** → lower bias, higher variance |

<div align="center">

```text
Large σ²:  wide, smooth bump centred at l   (underfitting risk)
Small σ²:  narrow, sharp spike centred at l (overfitting risk)
```

</div>

> [!TIP]
> Think of $\sigma^2$ as a "zoom level" for similarity. A large $\sigma^2$ zooms out, so many points look similar to a landmark. A small $\sigma^2$ zooms in, so only points extremely close to a landmark are considered similar.

---

## 17. SVM vs Logistic Regression: When to Use Each

### Decision Boundary Behaviour

| | SVM | Logistic Regression (LR) |
|---|---|---|
| **Boundary chosen** | Finds *the* maximum-margin boundary (distance between the decision boundary and the support vectors) | Can settle on different decision boundaries depending on the parameters $\theta$ that happen to be near the optimal solution |
| **Uniqueness** | Effectively one well-defined "best" boundary | Several near-optimal boundaries are possible |

### Choosing Between LR and SVM

The choice depends on the number of training samples ($n$) and the number of features ($m$):

| Scenario | Recommendation |
|---|---|
| $m$ is **large** and $n$ is **small** ($m \gg n$) | Linear SVM or Logistic Regression |
| $m$ is **small** and $n$ is **intermediate** ($n > m$) | **SVM with Gaussian kernel** |
| $m$ is **small** and $n$ is **large** ($n \gg m$) | Linear SVM or Logistic Regression (preferable to add more features) |

> [!NOTE]
> The Gaussian-kernel SVM shines specifically in the "middle" regime — enough data to learn a rich non-linear boundary via kernels, but not so many features that a simple linear model is already sufficient.

---

## 18. SVM vs ANN

| Aspect | SVM | ANN |
|--------|-----|-----|
| **Optimization landscape** | Convex optimization problem — guaranteed to reach the global optimum | Non-convex — can suffer from multiple local minima |
| **Number of classes** | Inherently binary — finds a hyperplane separating two classes | Handles many classes by design |
| **Performance on very large datasets** | Can be outperformed | Generally performs better on very large datasets |
| **Overall winner** | Depends on the problem and dataset — neither is universally best | Depends on the problem and dataset — neither is universally best |

> [!IMPORTANT]
> There is no single answer to "SVM or ANN?" — the right choice depends on dataset size, number of classes, feature dimensionality, and computational constraints.

---

## 19. Key Takeaways

> [!IMPORTANT]
> **The 14 things to remember from Week 4 Part 2:**

1. SVM is a supervised algorithm also known as a **large-margin classifier**
2. The objective is to find the decision boundary with the **maximum margin** to the nearest data points
3. **Support vectors** are the (misclassified or boundary-adjacent) points that actually define the hyperplane
4. SVM's cost function replaces logistic regression's log-loss with the **hinge loss**: $cost_1=\max(0,1-\theta^Tx)$ and $cost_0=\max(0,1+\theta^Tx)$
5. SVM requires $\theta^Tx\geq1$ for $y=1$ and $\theta^Tx\leq-1$ for $y=0$ — a stricter condition than plain logistic regression
6. The regularized cost is $\min_\theta C[\sum \cdots]+\frac{1}{2}\sum\theta_j^2$, where $C$ behaves like $\frac{1}{\lambda}$
7. **Large $C$** → sensitive to outliers, lower bias, higher variance (overfitting risk)
8. **Small $C$** → tolerant of margin violations, higher bias, lower variance (underfitting risk)
9. Non-linear data requires a **non-linear decision boundary**, which linear SVM cannot produce directly
10. The **kernel trick** maps data into a higher-dimensional space so a linear separator becomes possible
11. **Landmarks** are placed at the location of every training sample; new features measure similarity to each landmark
12. The **Gaussian (RBF) kernel** produces similarity $\approx1$ when $x$ is close to a landmark, and $\approx0$ when far away
13. **$\sigma^2$** controls the smoothness of the similarity function — large $\sigma^2$ underfits, small $\sigma^2$ overfits
14. Choosing between SVM, logistic regression, and ANN depends on the number of features $m$, the number of samples $n$, and the dataset/problem itself

---

## 20. Glossary

| Term | Definition |
|------|-----------|
| **Support Vector Machine (SVM)** | A supervised large-margin classification algorithm |
| **Large-Margin Classifier** | A classifier that chooses the decision boundary maximizing the distance to the nearest data points |
| **Decision Boundary** | The hyperplane $\theta^Tx=0$ that separates the two classes |
| **Margin** | The region between the decision boundary and the parallel hyperplanes $\theta^Tx=\pm1$ |
| **Support Vector** | A data point that is misclassified or lies close to the decision boundary; defines the hyperplane |
| **Hinge Loss** | The piecewise-linear cost function used by SVM: $\max(0,1-\theta^Tx)$ or $\max(0,1+\theta^Tx)$ |
| **Regularization Parameter $C$** | Controls the trade-off between margin size and misclassification tolerance in SVM; analogous to $\frac{1}{\lambda}$ |
| **Overfitting (Large C)** | A boundary too sensitive to individual (possibly noisy) points; low bias, high variance |
| **Underfitting (Small C)** | A boundary too tolerant of misclassification; high bias, low variance |
| **Non-Linear Decision Boundary** | A curved boundary needed when classes cannot be separated by a straight hyperplane |
| **Kernel Function** | A similarity function $f$ used to transform data so it becomes linearly separable |
| **Kernel Trick** | Mapping data into a higher-dimensional feature space to enable linear separation, without needing to compute that space explicitly |
| **Landmark** | A reference point (typically each training sample) used to compute similarity-based features |
| **Gaussian Kernel / RBF Kernel** | A kernel of the form $\exp(-\|x-l\|^2/2\sigma^2)$ measuring similarity via Euclidean distance |
| **Euclidean Distance** | The straight-line distance $\|x-l\|$ between two points in feature space |
| **$\sigma^2$ (Kernel Bandwidth)** | Parameter controlling how quickly similarity decays with distance in the Gaussian kernel |
| **Bias–Variance Trade-off** | The balance between a model being too simple (underfitting) or too complex (overfitting) |
| **Convex Optimization** | An optimization problem with a single global optimum — a property of the standard SVM formulation |

---

> [!TIP]
> **Study tip for Week 4 Part 2:** Make sure you can:
> 1. Explain why SVM is called a "large-margin classifier"
> 2. Identify the support vectors in a simple 2D scatter plot
> 3. Write down the hinge loss and explain how it differs from log-loss
> 4. Explain why SVM requires $\theta^Tx\geq1$ (not just $\geq0$) for $y=1$
> 5. Describe the effect of large vs small $C$ on bias, variance, and outlier sensitivity
> 6. Explain the kernel trick and how landmarks are chosen
> 7. Compute a Gaussian kernel similarity value given $x$, $l$, and $\sigma^2$
> 8. Decide whether to use logistic regression, linear SVM, or a Gaussian-kernel SVM based on $m$ and $n$
> 9. Compare SVM and ANN in terms of convexity, class handling, and dataset size
