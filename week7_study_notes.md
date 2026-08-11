# 📘 Week 7 Study Notes: Object Detection with Deep Learning

> **Course:** COS30082 — Applied Machine Learning  
> **Topic:** Object Detection with Deep Learning  
> **Prerequisite:** Week 5–6 (Convolutional Neural Networks, Transfer Learning)

---

## Table of Contents

1. [What Is Object Detection?](#1-what-is-object-detection)
2. [Object Detection vs Image Classification](#2-object-detection-vs-image-classification)
3. [Three Related but Different Problems](#3-three-related-but-different-problems)
4. [Milestones of Object Detection and Recognition](#4-milestones-of-object-detection-and-recognition)
5. [Why a Standard CNN Framework Fails at Detection](#5-why-a-standard-cnn-framework-fails-at-detection)
6. [Two Approaches to Object Detection](#6-two-approaches-to-object-detection)
7. [Approach 1: Traditional Object Detection](#7-approach-1-traditional-object-detection)
8. [Approach 2: Pre-trained Network as a Base Network](#8-approach-2-pre-trained-network-as-a-base-network)
9. [Base Networks](#9-base-networks)
10. [Object Detection Frameworks: One-Stage vs Two-Stage](#10-object-detection-frameworks-one-stage-vs-two-stage)
11. [Two-Stage Detector 1: R-CNN](#11-two-stage-detector-1-r-cnn)
12. [Two-Stage Detector 2: Fast R-CNN](#12-two-stage-detector-2-fast-r-cnn)
13. [Two-Stage Detector 3: Faster R-CNN](#13-two-stage-detector-3-faster-r-cnn)
14. [One-Stage Detector 1: YOLO](#14-one-stage-detector-1-yolo)
15. [One-Stage Detector 2: SSD](#15-one-stage-detector-2-ssd)
16. [Common Object Detection Tricks](#16-common-object-detection-tricks)
17. [Summary: One-Stage vs Two-Stage, All Models Side by Side](#17-summary-one-stage-vs-two-stage-all-models-side-by-side)
18. [Key Takeaways](#18-key-takeaways)
19. [Glossary](#19-glossary)
20. [Study Tips for Week 7](#20-study-tips-for-week-7)

---

## 1. What Is Object Detection?

**Object Detection** is a computer vision task where a model must **find** where objects are in an image (with a bounding box) **and** say **what** each object is (a class label).

$$
\boxed{\text{Object Detection} = \text{Localization (where?)} + \text{Classification (what?)}}
$$

> **Analogy:** Imagine handing a photo of a busy street to two different people. One person (an **image classifier**) can only shout out a single word describing the whole photo, like *"street!"*. The other person (an **object detector**) walks up to the photo with a marker pen, draws a box around every car, every pedestrian, and every traffic light, and labels each box individually. The second person is doing exactly what an object detector does — locating **and** naming multiple things at once.

---

## 2. Object Detection vs Image Classification

| | **Image Classification** | **Object Detection** |
|---|---|---|
| **Example** | <p align="center"><img src="https://github.com/user-attachments/assets/c1dd3565-23f7-49a7-8613-319e6bf25236" width=600></p> | <p align="center"><img src="https://github.com/user-attachments/assets/bc9229e7-9801-481f-a399-f8cf63f290f8" width=600></p> |
| **Goal** | Predict the type/class of an object in an image | Locate objects with a bounding box **and** predict the class of each one |
| **Output** | A single class label for the whole image | One or more bounding boxes, each with its own class label |
| **Example output** | `cat` | `cat (x, y, w, h)`, `duck (x, y, w, h)`, `dog (x, y, w, h)` |

> **Analogy:** Image classification is like being asked "What's the theme of this photo?" — you give one answer for the whole picture. Object detection is like being asked "Point to every animal in this photo and tell me what each one is" — you give **many** answers, each tied to a specific location.

---

## 3. Three Related but Different Problems

Object detection is best understood as the **combination** of two simpler problems:

| Problem | Input | Output |
|---|---|---|
| **Image Classification** | An image | A class label |
| **Object Localization** | An image with one or more objects | One or more bounding boxes (e.g. defined by a point, width, and height) |
| **Object Detection** | An image with one or more objects | One or more bounding boxes **+** a class label for each box |

<div align="left">

```text
                    ┌── Image classification  (what is it?)
Object detection ───┤
                    └── Object localization   (where is it?)
```

</div>

> **Analogy:** Think of a food delivery app. **Classification** is like the app just telling you "there's food in this bag." **Localization** is like the app telling you "the food is on shelf 3, item 2" without saying what it is. **Object detection** does both together: "on shelf 3, item 2, there's a burger; on shelf 5, item 1, there are fries."

---

## 4. Milestones of Object Detection and Recognition

Before 2012, object detection relied on **handcrafted features** — features that humans manually designed (SIFT, HOG, SURF, DPM, etc.). The field went through a major shift in **2012**, when **Krizhevsky et al.** introduced **Deep CNNs (AlexNet)** for image classification. Every major detector after 2012 (R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD, ResNet, DenseNet, Mask R-CNN...) is built on deep networks.

<div align="center">

```text
1999 ───────────────────────── 2012 ───────────────────────── 2016+

Handcrafted-feature era          Transition:              Deep-learning era
(SIFT, HOG, DPM,                 Deep CNN AlexNet          (R-CNN, Fast R-CNN,
 Selective Search...)            (Krizhevsky et al.)        Faster R-CNN, YOLO,
                                                              SSD, ResNet...)
```

</div>

> **Analogy:** Handcrafted-feature detection was like a chef who follows a **fixed recipe card** someone else wrote — they can only cook what's written down. Deep learning detection is like a chef who has **tasted thousands of dishes** and learned, on their own, which flavour combinations work — they can generalise to new dishes nobody explicitly wrote a recipe for.

---

## 5. Why a Standard CNN Framework Fails at Detection

A standard CNN (convolution layers → fully connected layer) works great for classification, but it **cannot** be directly reused for detection.

**The core problem:** the length of the output layer must be **fixed**, but the number of objects in an image is **variable** — one image might have 0 objects, another might have 12.

$$
\boxed{\text{Fixed-size output} \neq \text{Variable number of objects}}
$$

> **Analogy:** Imagine a form that only has **3 blank lines** for "list all attendees at the meeting." That form works fine if exactly 3 people show up — but it completely breaks down if 1 person shows up (2 blank lines wasted) or if 10 people show up (nowhere to write the other 7). A plain CNN + fully connected layer is exactly this rigid form: it can't adapt its output size to however many objects actually appear in the picture.

This is precisely **why** object detection needs specialised architectures instead of "just add more classification heads."

---

## 6. Two Approaches to Object Detection

| Approach | Description |
|---|---|
| **Method 1: Traditional object detection** | - Non-deep-learning, computer-vision-based methods (e.g. **sliding windows** + **image pyramids**) <br>- Typically paired with **HOG + Linear SVM** <br>- Slow, tedious, error-prone |
| **Method 2: Pre-trained network as a base network + detection framework** | - Use a pre-trained classification network (e.g. VGG, ResNet) as a **base network** inside a deep-learning detection framework (Faster R-CNN, SSD, YOLO) |

> **Analogy:** Method 1 is like searching for your keys by checking **every single spot in your house one at a time**, room by room, drawer by drawer — thorough, but painfully slow. Method 2 is like already having a rough memory of "I usually leave my keys near the door or the kitchen counter" (a pre-trained network's learned visual knowledge), so you jump straight to the likely spots.

---

## 7. Approach 1: Traditional Object Detection

The traditional pipeline has **4 steps**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ae7093e-0a0e-4879-8ec9-5f091b0083d1" width=600>
</p>

1. **Fixed-size sliding windows** slide left-to-right, top-to-bottom to localise objects at different positions.
2. An **image pyramid** resizes the image at multiple scales, so objects of different sizes can still be caught by the fixed-size window.
3. Each region of interest (ROI) is **classified** — either with a pre-trained CNN or a traditional method like SVM.
4. If the classification probability of label $L$ satisfies $P(L) > T$ (some threshold), the window is marked with label $L$.

> **Analogy:** This is like searching for a friend's face in a crowd by holding up a **small picture-frame cut-out** and physically moving it across the entire crowd, inch by inch, then **zooming your camera in and out** repeatedly to check for both close and far-away faces. It works, but it's mind-numbingly repetitive — you're re-examining the same pixels dozens of times at different positions and scales.

---

## 8. Approach 2: Pre-trained Network as a Base Network

Instead of manually sliding windows, we treat a pre-trained classification network as a **base network** plugged into a deep-learning detection framework such as **Faster R-CNN**, **SSD**, or **YOLO**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c44dbe11-8d6f-44b1-8cb4-635f724bfc48" width=600>
</p>


> **Example:** The diagram above shows **VGG16 (up to the Conv5_3 layer)** acting as the base network for the **SSD** framework, followed by extra feature layers, classifiers, and non-maximum suppression to produce final detections.

> **Analogy:** This is like a construction company that doesn't pour a brand-new **foundation** for every single building — they reuse a proven, pre-tested foundation design (the base network) and just build different structures (the detection framework: classifiers, regressors) on top of it.

---

## 9. Base Networks

**Base networks** are the common CNN classification architectures, typically pre-trained on a large dataset like **ImageNet** to learn a rich set of general, discriminating filters.

| Base Network | Common Role |
|---|---|
| **VGGNet** | Simple, uniform 3×3 convolutions; commonly paired with SSD |
| **ResNet** | Uses residual (skip) connections; strong general-purpose backbone |
| **MobileNet** | Lightweight, designed for mobile/embedded devices |
| **DenseNet** | Densely connects every layer to every other layer for feature reuse |

> **Analogy:** Base networks are like **pre-trained apprentices** who already know how to recognise edges, textures, shapes, and colours from years of general training (ImageNet). When you hire one for a specific job (detection), you don't need to teach them "what an edge looks like" from zero — you only need to teach them the job-specific skill on top of their existing general knowledge. This is essentially **transfer learning** (Week 6) applied to detection.

---

## 10. Object Detection Frameworks: One-Stage vs Two-Stage

| | **Two-Stage Detector** | **One-Stage Detector** |
|---|---|---| 
| **Visual Flowchart** | <p align="center"><img src="https://github.com/user-attachments/assets/5eb7c04e-fbe9-45b1-b985-c9c5a4a449d0" width=600></p> | <p align="center"><img src="https://github.com/user-attachments/assets/0230b229-8331-410d-bf70-bcd598f5cdf8" width=600></p>
| **How it works** | (1) Propose regions of interest via a region-proposal method/network <br>(2) then classify only those region candidates | A **single** convolutional network directly predicts bounding boxes **and** class probabilities in one pass |
| **Models** | R-CNN, Fast R-CNN, Faster R-CNN | YOLO, SSD |
| **Speed** | Slower — must run predictions for every selected region | Faster — commonly used for **real-time** detection |
| **Accuracy** | Generally **higher** accuracy | Trades a bit of accuracy for **large** speed gains |

> **Analogy:** A **two-stage detector** is like a hiring process with **two separate rounds**: first a recruiter skims resumes to shortlist promising candidates (region proposal), then a specialist panel carefully interviews only the shortlisted candidates (classification + refinement). It's thorough but takes longer. A **one-stage detector** is like a single interviewer who looks at *every* candidate once and makes an instant decision on the spot — much faster, but occasionally less accurate because there's no second, more careful look.

---

## 11. Two-Stage Detector 1: R-CNN

**R-CNN (Regions with CNN features)** was the first major deep-learning object detector. Its pipeline has **4 main blocks**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/b870ce9e-08d5-4b83-893a-0ccc94c043f5">
</p>

> **Source:** Girshick, R., Donahue, J., Darrell, T. and Malik, J., 2014. *Rich feature hierarchies for accurate object detection and semantic segmentation.* CVPR.

### 11.1 CNN Classification Model Pre-training

**Pre-train** a CNN (e.g. VGG or ResNet) on an image classification task, typically ImageNet. The network learns from the following phases:

<p align="center">
  <img src="https://github.com/user-attachments/assets/fe0c65e6-fdc0-4ff4-9873-241cf40425f6">
</p>

> **Analogy:** This is the "general education" phase — like a new employee going through onboarding training that teaches broad skills useful in almost any role at the company, before being assigned to a specific project.

### 11.2 Region Proposal

We first generate **region proposals** using an algorithm such as **Edge Boxes** or **Selective Search** (~**2,000 candidate proposals per image**). Those regions may contain target objects, and they come in different sizes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fb126803-afde-4af1-824f-adb8179bc1bc" width=600>
</p>

- Region proposals with **≥ 0.5 IoU** overlap with a ground-truth box are treated as **positives** for that box's class; the rest are **negatives**.

> **Analogy:** Region proposal is like a talent scout at a stadium **circling roughly 2,000 spots in the crowd** that *might* contain something interesting (a face, a mascot, a sign), without yet knowing exactly what's there. It's a rough first filter — cheap and fast — that narrows down the search before anyone spends real effort examining each one closely.

### 11.3 Worked Example: Understanding Intersection over Union (IoU)

**IoU** is one of the most important concepts in this entire topic — it measures how well a **predicted** bounding box overlaps with the **ground-truth** bounding box. Let's read it step by step, the same way we'd interpret any evaluation metric.

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/598ae6dd-d655-408e-86ce-3df276fbf213" width=600>
</p>

**How to read this scale:**

| IoU range | Interpretation |
|---|---|
| **Close to 0** | Predicted box barely overlaps the true object — a bad prediction |
| **Around 0.4–0.6** | Some overlap, but the box is clearly misaligned ("Poor" in the example) |
| **Around 0.7–0.8** | The box roughly matches the object — usually the threshold used to count a detection as "correct" in training/evaluation ("Good") |
| **Close to 1.0** | Near-perfect match between prediction and ground truth ("Excellent") |

<p align="center">
  <img src="https://github.com/user-attachments/assets/d42dc1fc-1aa4-4e52-8e1c-f096094921fb" width=600>
</p>

<p align="center">
  <em>Example of IoU on a STOP sign</em>
</p>

> **Analogy:** IoU is like grading how well someone parked a car inside a parking space by comparing **how much of the space is covered by the car** versus **how much total space the car + the empty space together take up**. If they parked perfectly inside the lines, IoU ≈ 1. If they're halfway on the line into the next spot, IoU drops a lot — even though "most" of the car is still roughly in the right area.

### 11.4 CNN Classification Model Fine-tuning

The proposed regions are **cropped**, **resized**, and **warped** to a fixed size (as required by the CNN). The CNN is then **fine-tuned** on these warped regions for **K + 1 classes** (the "+1" is the **background** class).

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d807179-168c-4919-8db5-b43be43f20cd" width=1000>
</p>

- A much **smaller learning rate** is used during fine-tuning.
- The mini-batch **oversamples positive cases**, because most proposed regions are just background.

> **Analogy:** This is like re-training that general-education employee (Section 11.1) with a **short, focused refresher course** specific to the new job (e.g. "here's what our specific products look like"), rather than sending them through onboarding all over again. Because most of what they'll see day-to-day is "nothing relevant" (background), the refresher course makes sure to include plenty of real examples so they don't forget to recognise the actual product.

### 11.5 Object Classification (Binary SVM)

For every image region, one forward pass through the CNN generates a **feature vector**. This feature vector is fed into a **binary SVM** trained **independently for each class**.

- **Positive samples:** proposed regions with IoU ≥ 0.3 with a ground-truth box.
- **Negative samples:** everything else (irrelevant regions).
- A subset of **"hard negatives"** (regions that look deceptively similar to the positive class) is specifically used to sharpen the SVM's decision boundary.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa4036cd-7271-4386-94cb-c055c50664cc" width=1000>
</p>

> **Analogy:** Training one binary SVM per class is like hiring a **separate specialist gatekeeper for each category** — one gatekeeper only asks "is this a dog or not?", another only asks "is this a cat or not?" — rather than one gatekeeper trying to juggle every category at once.

### 11.6 Bounding Box Regression

The bounding-box regression stage **improves localization** — it predicts a refined bounding box for each detection, using the CNN features. Only boxes with **IoU ≥ 0.6** are used to train the class-specific regressor.

<p align="center">
  <img src="https://github.com/user-attachments/assets/23712abb-e51d-45bd-9149-f0b77ab1c4e0" width=1000>
</p>

**Goal:** learn a transformation that maps a proposal box $P = (P_x, P_y, P_h, P_w)$ to a ground-truth box $G = (g_x, g_y, g_h, g_w)$.

$$
L_{reg} = \sum_{i \in \{x,y,w,h\}} (t_i - d_i(P))^2 + \lambda \lVert w \rVert^2
$$

where the scale-invariant regression targets are:

$$
t_x = \frac{g_x - P_x}{P_w}, \quad t_y = \frac{g_y - P_y}{P_h}, \quad t_w = \log\left(\frac{g_w}{P_w}\right), \quad t_h = \log\left(\frac{g_h}{P_h}\right)
$$

> **Analogy:** Bounding-box regression is like a photo-editing app's "**auto-align**" feature — you've already roughly cropped a photo (the region proposal), and instead of manually dragging the corners pixel-by-pixel, the app **nudges** the crop box slightly left/right/up/down and slightly resizes it to snugly fit the actual subject.

### 11.7 Problems with R-CNN

| Problem | Explanation |
|---|---|
| **Slow at test-time** | Needs an **independent full forward pass of the CNN for every region proposal** (~2,000 passes per image!) |
| **Post-hoc training of SVMs/regressors** | The CNN features are **not updated** in response to the SVM/regressor — this can lead to sub-optimal region proposals |
| **Complex, multi-stage pipeline** | Three separate models trained with **little shared computation**: (1) CNN for classification/feature extraction, (2) SVM for object classification, (3) regression model for box correction |

> **Analogy:** R-CNN is like re-reading the **entire textbook from page one** every single time you want to answer just **one practice question**, and doing this separately for **2,000 different questions**, using three different unconnected study methods (memorising, flashcards, and practice tests) that never talk to each other. Technically it works — but it's an enormous, wasteful amount of repeated effort.

---

## 12. Two-Stage Detector 2: Fast R-CNN

<p align="center">
  <img src="https://github.com/user-attachments/assets/ecf33525-0310-4b74-81a3-4dfbc874acd4" width=1000>
</p>

> **Source:** Girshick, R., 2015. *Fast R-CNN.* ICCV.

**Fast R-CNN** makes R-CNN **faster** by **unifying the three independent models** (CNN, binary SVM, regressor) from the existing CNN problem: "Complex, multi-stage pipeline" into **one jointly trained framework**, sharing computation.

### 12.1 Solving R-CNN's Problems

| R-CNN Problem | Fast R-CNN's Solution |
|---|---|
| **#1: Slow test-time** (independent forward pass per region) | **Share computation** of convolutional layers across all proposals for an image — run the CNN **once per image**, not once per region |
| **#2: Post-hoc training** (CNN not updated by the classifier/regressor) | Train the **whole system end-to-end**, all at once |
| **#3: Complex multi-stage pipeline** | Train the whole system **end-to-end**, all at once |

> **Analogy:** Instead of re-reading the whole textbook 2,000 times (R-CNN), Fast R-CNN **reads the textbook once**, remembers the general "notes" (the shared feature map), and then quickly flips to the relevant highlighted section for each individual question. Everything also happens in **one integrated study session** instead of three disconnected methods.

### 12.2 Model Workflow

<p align="center">
  <img src="https://github.com/user-attachments/assets/77f26666-67ce-4f9b-a792-c773db6c5607" width=600>
</p>

### 12.3 Worked Example: Reading the ROI Pooling Diagram

**ROI Pooling** is the key trick that lets Fast R-CNN turn region proposals of *different sizes* into a **fixed-size** feature vector, so they can all be fed into the same fully connected layers. Let's walk through the example, the same way we would trace through a convolution calculation.

**Setup:** An 8×8 input feature map, one region proposal, and a desired output size of **2×2**.

1. 8x8 input feature map
2. Region proposal (7x5 box)
3. Pooling sections
4. MAX pixel value in each of the 4 sections obtained

<p align="center">
  <img src="https://github.com/user-attachments/assets/b45f13c1-0fd6-42db-bab4-a32aa3008d48" width=1500>
</p>

**How to read this:** no matter whether the input region is 7×5, 12×9, or 3×3, ROI Pooling always slices it into the **same fixed grid** (here 2×2) and takes the **max value** in each cell. The output is *always* the same shape, ready to feed into a fully connected layer.

> **Analogy:** ROI Pooling is like taking photographs of people of **wildly different heights and widths** and squeezing them all into the **same-sized ID photo frame** by dividing each photo into a fixed grid and picking the brightest pixel from each grid cell. No matter how tall or short the original person was, the final ID photo is always the same standard size — which is exactly what the next layer (the classifier) needs to work consistently.

### 12.4 Loss Function

Fast R-CNN uses a **multi-task loss** that sums classification and bounding-box loss:

$$
L(p, u, t^u, v) = \underbrace{L_{cls}(p, u)}_{-\log p_u} + \mathbb{1}(u \geq 1)\underbrace{L_{box}(t^u, v)}_{\sum_{i \in \{x,y,w,h\}} L_1^{smooth}(t_i^u - v_i)}
$$

where $u$ is the true class label (background class $u = 0$), $p$ is the predicted class-probability distribution, $v$ is the true bounding box, and $t^u$ is the predicted box correction.

The **Smooth L1 loss** is used instead of plain L2 (as in R-CNN) because it's **less sensitive to outliers**:

$$
L_1^{smooth}(x) =
\begin{cases}
0.5x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}
$$

> **Analogy:** L2 loss is like a strict teacher who **punishes a wildly wrong answer exponentially harder** than a slightly wrong one — great for encouraging precision, but it can get thrown off badly by a single extreme mistake (an "outlier" region). Smooth L1 loss is like a more balanced teacher who penalises small mistakes gently (like L2) but switches to a milder, steady penalty for very large mistakes — so one bad outlier doesn't dominate the entire training process.

### 12.5 Speed Bottleneck

Fast R-CNN is much faster than R-CNN in both training and testing — **but the improvement isn't dramatic**, because **region proposals are still generated separately** by an external algorithm (Selective Search), which is itself expensive.

> **Analogy:** Fast R-CNN is like upgrading your kitchen equipment so you cook each dish much faster — but you're **still waiting on a slow, external grocery delivery service** to bring you ingredients before you can start cooking at all. The kitchen isn't the bottleneck anymore; the delivery is.

### 12.6 Worked Example: Reading the Performance Comparison Table

| Metric | R-CNN | Fast R-CNN | How to read it |
|---|---|---|---|
| **Training Time** | 84 hours | **9.5 hours** | Fast R-CNN trains **8.8× faster** — shared computation avoids recomputing CNN features for every region |
| **Test time per image** | 47 seconds | **0.32 seconds** | Fast R-CNN is **146× faster** at test time — again thanks to computing the feature map only once per image |
| **mAP (VOC 2007)** | 66.0 | **66.9** | Accuracy is *not sacrificed* for speed — it's even slightly **better**, since the whole network is trained end-to-end |

> **How to interpret a table like this in general:** when comparing detectors, always check **three axes together**: (1) how long it takes to *train*, (2) how long it takes to *run on one image* (inference speed), and (3) how *accurate* it is (mAP). A model can look impressive on speed alone but be misleading if its accuracy dropped — the useful comparison is always speed **and** accuracy side by side, not just one number in isolation.

---

## 13. Two-Stage Detector 3: Faster R-CNN

<p align="center">
  <img src="https://github.com/user-attachments/assets/2d0ff64e-f686-42ae-a128-f0dc03c9cdb2" width=600>
</p>

> **Source:** Ren, S., He, K., Girshick, R. and Sun, J., 2016. *Faster R-CNN: Towards real-time object detection with region proposal networks.* IEEE TPAMI.

**Faster R-CNN** removes the last remaining bottleneck from **Fast R-CNN**: the external region-proposal algorithm. It inserts a **Region Proposal Network (RPN)** directly **after the last convolutional layer**, so region proposals are generated **inside the network itself** — no external Selective Search needed.

> **Analogy:** If Fast R-CNN was like upgrading the kitchen but still relying on a slow external grocery delivery (Section 12.5), Faster R-CNN is like **growing your own vegetable garden right next to the kitchen** — the ingredients (region proposals) are now produced **in-house**, instantly, using the same infrastructure (the shared feature map) that's already there for cooking.

### 13.1 Region Proposal Network (RPN) and Anchor Boxes

The RPN slides over the feature map and, at every location (called an **anchor point**), proposes a set of **anchor boxes** — predefined, fixed-size boxes of different **scales** and **aspect ratios**, spread across the whole image.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a62d86a4-577a-4e51-8771-a0424669aa88" width=1000>
</p>

- **Anchors are translation invariant** — the *same* set of anchor shapes is reused at every location.
- For each anchor box, the RPN outputs:
  - **2k scores** → is there an object here or not? (`cls` layer)
  - **4k coordinates** → how should the box be adjusted? (`reg` layer)
- Typically **9 anchor boxes per anchor point** (3 scales × 3 aspect ratios), to capture objects of different sizes and shapes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d8426421-d529-4484-b45c-63fe08806478" width=600>
</p>

> **Analogy:** Anchor boxes are like a photographer who, before even looking at the scene, sets up **9 different picture-frame stencils** of various shapes and sizes (tall/thin, short/wide, square) and holds each one up at every spot in the photo to check "does something interesting fit inside this particular frame shape, here?" Having multiple frame shapes ready in advance means the photographer doesn't miss a tall giraffe just because they were only checking with a wide, short frame.

### 13.2 Training of the RPN

Each anchor box is assigned a label (**object** or **not**) based on its IoU with the ground truth:

| Condition | Label |
|---|---|
| IoU > **0.7** with any ground-truth box | **Positive** |
| IoU < **0.3** with **all** ground-truth boxes | **Negative** |

A mini-batch of **256 randomly picked anchor boxes** (from the same image) is used per training step, aiming for a **1:1 ratio** of positive to negative anchors. If there are fewer than 128 positives, the batch is **padded with negatives**. The RPN is then trained end-to-end with backpropagation and SGD, minimising the loss with the loss function, where $cls$ is classification and $reg$ is regression:

$$
L = L_{cls} + L_{reg}
$$

> **Analogy:** This is like grading a multiple-choice test where you only keep the **clearly correct answers** (IoU > 0.7) and the **clearly wrong answers** (IoU < 0.3) for training purposes, and simply **throw out the ambiguous, borderline ones** (0.3–0.7) — because training on confusing, ambiguous examples would just muddy the signal.

### 13.3 Training of Faster R-CNN

**Original paper's approach — "ugly" alternating optimisation** (4 steps):

<p align="center">
  <img src="https://github.com/user-attachments/assets/508bcfda-cc94-4d69-97d7-acf5cef296ed" width=800>
</p>

1. Train the RPN end-to-end (CNN initialised from an ImageNet pre-trained model).
2. Train a **separate** Fast R-CNN, also initialised from an ImageNet pre-trained model. At this point, RPN and Fast R-CNN **share no convolution layers**.
3. Initialise a new RPN from the Fast R-CNN's weights, **freeze** the shared convolution layers, and fine-tune only the RPN-specific layers.
4. Keep the shared convolution layers frozen and fine-tune only the Fast-R-CNN-specific layers.

**Since publication — joint training (simpler & preferred):** train **one network with four losses simultaneously**:
- RPN classification (anchor good/bad)
- RPN regression (anchor → proposal)
- Fast R-CNN classification (over classes)
- Fast R-CNN regression (proposal → box)

> **Analogy:** The original 4-step process is like two roommates (RPN and Fast R-CNN) who **each cook a separate meal from scratch**, then awkwardly try to combine leftovers afterward. Joint training is like both roommates **cooking together in the same kitchen at the same time**, sharing ingredients (the shared convolutional layers) from the very start — much more efficient, and it's the version almost everyone uses today.

### 13.4 Worked Example: Reading the Full Performance Comparison Table

| Metric | R-CNN | Fast R-CNN | Faster R-CNN |
|---|---|---|---|
| **Test time per image (with proposals)** | 50 seconds | 2 seconds | **0.2 seconds** |
| **Speedup** | 1× | 25× | **250×** |
| **mAP (VOC 2007)** | 66.0 | 66.9 | **66.9** |

**How to read this progression:** each new model in the R-CNN family solves the **specific bottleneck** left by the previous one, **without losing accuracy**:
- R-CNN → Fast R-CNN: solved the "recompute CNN features per region" bottleneck.
- Fast R-CNN → Faster R-CNN: solved the "external, slow region-proposal algorithm" bottleneck.
- Notice that **accuracy stays essentially flat (66.9)** the whole way — the story here is purely about removing wasted computation, step by step, not about a smarter model that predicts "better."

> **Analogy:** Think of upgrading transportation between two cities: R-CNN is walking, Fast R-CNN is driving a car with someone constantly getting out to check every side street on foot (still slow), and Faster R-CNN is driving straight through on a **highway built specifically for this trip** — same destination (accuracy), *far* less wasted time getting there.

---

## 14. One-Stage Detector 1: YOLO

> **Source:** Redmon, J., Divvala, S., Girshick, R. and Farhadi, A., 2016. *You Only Look Once: Unified, real-time object detection.* CVPR.

**YOLO** was the **first attempt** at a fast, real-time object detector. Instead of proposing regions and then classifying them (two stages), YOLO looks at the image **only once** and directly predicts everything in one pass.

### 14.1 How YOLO Works

The image is split into an **S × S grid**. Within each grid cell, the model proposes **B bounding boxes** (anchor boxes). For each bounding box, the network outputs:

- **B bounding boxes** (position + size)
- A **confidence score** for each box: $\text{Confidence}(C) = p(\text{object}) \times \text{IOU}(\text{pred}, \text{truth})$
- **Class probabilities** $p(c)$ for that grid cell

Boxes with a confidence score above a threshold are kept and used to locate objects.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d13379d4-0fa5-4992-b280-90f8c42b57a9" width=600>
</p>

> **Analogy:** Imagine dividing a classroom photo into a **7×7 seating chart grid**. Each "seat" in the grid is responsible for reporting: "is there a student sitting roughly here, and if so, who is it and how confident am I?" Combine every seat's report, throw out the low-confidence ones, and you get the full roster of who's in the photo — all figured out in **one single glance** across the whole room, rather than interviewing each seat individually.

### 14.2 The "Responsible" Predictor

At one grid cell $i$, the model proposes **B** bounding-box candidates. The one with the **highest IoU** (overlaps the most with the ground truth) becomes the **"responsible" predictor** for that object — it's the only box whose prediction gets trained against that ground truth.

> **Analogy:** If several friends all guess where a hidden treasure is buried near the same general area, only the friend whose guess is **closest** to the actual treasure gets "credit" (positive reinforcement) for the next round — the others aren't punished as harshly, since they weren't the best guess.

### 14.3 Network Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/0096dc8f-7330-400b-a158-0401a07fc4d2" width=600>
</p>

The base model resembles **GoogLeNet**, with the inception module replaced by 1×1 and 3×3 conv layers. Two fully connected layers over the whole feature map produce a final prediction of shape:

$$
S \times S \times (B \times 5 + p(c))
$$

**YOLO's original configuration:** $S = 7$, $B = 2$, with **20 classes** → final output is a **7 × 7 × 30 tensor**.

### 14.4 Loss Function

$$
L = L_{cls} + L_{loc}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/0cd68831-a5ba-4fe9-bf23-ea0044d2f330" width=1000>
</p>

The loss is a sum of squared errors, using two scaling parameters:

- $\lambda_{coord}$ — controls how much to **increase** the loss from bounding-box coordinate predictions.
- $\lambda_{noobj}$ — controls how much to **decrease** the loss from confidence-score predictions on boxes **without** objects.

> **Why down-weight background boxes?** Most grid cells in a typical image contain **no object at all** (empty background). Without down-weighting, the huge number of "no object" boxes would **drown out** the learning signal from the few boxes that actually contain something important.

> **Analogy:** $\lambda_{noobj}$ is like telling a teacher grading a class quiz: "please don't spend equal energy praising or correcting the 90% of blank answer boxes — focus your grading effort on the handful of boxes where students actually wrote something." Otherwise the grading effort (and the model's learning signal) gets wasted on the overwhelming majority of "nothing here" cases.

### 14.5 YOLO vs Faster R-CNN

| Metric | YOLO | Faster R-CNN |
|---|---|---|
| **Speed** | **45 fps** | 7 fps |
| **Accuracy (mAP)** | 63.4% | **73.2%** |
| **Weakness** | Struggles with **small objects** or objects **close together**, since each grid cell only has 2 anchor boxes predicting **one** class | Detects small objects well (9 anchors per location), but **too slow** for real-time use |

> **Analogy:** YOLO is like a fast-talking auctioneer who calls out bids at lightning speed but occasionally mishears or lumps two nearby bidders together when the crowd is packed tightly. Faster R-CNN is like a careful, methodical auctioneer who double-checks every single bid individually — accurate, but far too slow to keep up in a live, fast-paced setting.

---

## 15. One-Stage Detector 2: SSD

<p align="center">
  <img src="https://github.com/user-attachments/assets/da0dc9a8-a3da-44c3-9ac0-0dd0228fe5e8" width=900>
</p>

> **Source:** Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y. and Berg, A.C., 2016. *SSD: Single Shot MultiBox Detector.* ECCV.

**SSD (Single Shot Detector)** was one of the first models to use a CNN's **pyramidal feature hierarchy** to efficiently detect objects of **many different sizes** in a single pass — and it ran **both faster and more accurately** than YOLO.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0f6debe3-37b6-440d-99a3-9b21f979d2e7" width=1000>
</p>

Don't worry if "pyramidal feature extraction" and "detection head" sound abstract right now — the next three subsections trace through this pipeline **one arrow at a time**, using the exact diagrams from the slides, so you can see precisely what shape of data flows between each box.

### 15.1 Pyramidal Feature Extraction

This is the **"Image → Feature maps"** part of the pipeline. The goal is to turn one 300×300 image into **several feature maps of different sizes**, so that later stages can search for objects at multiple scales at once.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0aca974a-5bce-4614-a8ea-6f9f52456411" width=1000>
</p>

**How to read this diagram, step by step:**
 
1. The 300×300 image first passes through **VGG** (a familiar base network from Section 9), which outputs a **38×38** feature map at the `Conv4_3` layer. This is immediately **tapped off** — copied sideways — as the **first** feature map SSD will use for detection, *before* the network goes any deeper.
2. The network keeps going through a couple more convolution blocks (labelled `Conv6`/`FC6` and `Conv7`/`FC7` in the full diagram), shrinking the map down to **19×19**, which is tapped off as the **second** feature map.
3. From here, SSD adds its own **"Extra Feature Layers"** — small conv blocks that each **halve the spatial size** roughly, producing **10×10**, then **5×5**, then **3×3** feature maps, each one tapped off in turn.
4. The `/2` you see in `3×3×512/2` simply means **stride 2** — the convolution slides two pixels at a time instead of one, which is what shrinks the grid (38→19→10→5→3) at each stage.
**The key insight:** every one of these five tapped feature maps is kept and used **simultaneously** for detection — the network doesn't throw away the 38×38 map just because it later computes a 3×3 map. Early, large feature maps (38×38) have a **small receptive field** per cell, so they're great at spotting **small** objects. Later, small feature maps (3×3) have a **large receptive field** per cell (each cell "sees" a huge chunk of the original image), so they're great at spotting **large** objects that fill much of the frame.

> **Analogy:** Think of a photographer taking the **same scene** at several different zoom levels — a wide shot (38×38, catches small objects with lots of detail), a medium shot (10×10), and a tight zoom (3×3, catches large objects that fill the frame). By checking for objects at **every** zoom level instead of just one, SSD doesn't miss a tiny object in a wide shot **or** a huge object that only fits in a zoomed-out view.

### 15.2 Detection Head & Default Boxes

This is the **"Feature maps → Detection head → Scores & boxes"** part of the pipeline. Now that we have several feature maps of different sizes, each one needs to actually **make predictions**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d719421e-e298-4862-b642-c9a9349cd277" width=1000>
</p>

**How to read this diagram, step by step:**
 
1. Each tapped feature map (from Section 15.1) is fed into its **own small detection head** — a tiny convolutional layer with kernel size **3×3**.
2. This conv layer doesn't just output "yes/no, object here" — for **every single grid cell**, and for **every one of the k default boxes** anchored at that cell, it outputs a full bundle of numbers: **c** class scores (one score per class, e.g. "cat: 0.1, dog: 0.7, background: 0.2") **plus 4** numbers describing how to nudge that default box's position and size (the same $t_x, t_y, t_w, t_h$-style offsets you saw in Sections 11.6 and 12.4).
3. That's exactly what `conv 3×3 × k(c+4)` means: **k** boxes, each carrying **(c+4)** numbers → **k(c+4)** output channels in total, produced by one 3×3 convolution.
4. Notice in the diagram that **different feature maps use a different number of default boxes k** — for example, one map's detection head might be `conv 3×3 × 4(c+4)` (4 boxes per cell) while another's is `conv 3×3 × 6(c+4)` (6 boxes per cell, usually because that scale uses more aspect ratios). Every feature map still follows the *same general formula*, just with its own value of k.
5. The output of every detection head is collectively labelled **"Scores & Boxes"** — a big pile of candidate predictions, one bundle per (grid cell, default box) pair.
   
At each feature-map location, a small `conv 3×3 × k(c+4)` layer predicts, for each of **k** default (anchor) boxes: **c** class scores + **4** box-offset coordinates.

**Why does this matter?** Multiply "grid cells" × "boxes per cell" across **all five (or six) tapped feature maps**, and you get a *huge* number of raw candidate boxes for one image — in the standard SSD300 configuration, this adds up to **8,732 candidate detections**. That's far too many to show the user directly — which is exactly why the pipeline's last step, **NMS** (Section 16.2), is needed to whittle this down to a small, clean set of final predictions.

### 15.3 Worked Example: Default Boxes and Aspect Ratios (Reading the Dog/Cat Diagram)

Now let's connect Sections 15.1 and 15.2 by tracing the **complete** network end to end, the same way the slides present it as one unified diagram.

<p align="center">
  <img src="https://github.com/user-attachments/assets/437364d1-48a0-4714-b123-8f28cc21039b" width=600>
</p>

A picture with both a **dog** and a **cat** are shown, alongside two different feature-map grids: an **8×8** grid and a **4×4** grid.

- Feature maps at **different levels have different receptive-field sizes**. Anchor boxes on each level are rescaled so that **one feature map is responsible for objects of roughly one particular scale**.
- In the example: the **dog** (a larger object filling more of the image) can only be detected in the **4×4 feature map** (a "higher," more zoomed-out level, where each cell "sees" a larger patch of the original image).
- The **cat** (a smaller object) is only captured by the **8×8 feature map** (a "lower," more zoomed-in level, where each cell covers a smaller patch, ideal for catching small details).

**How to read this kind of diagram in general:** larger, coarser grids (like 4×4) are naturally suited to **larger objects**, because each grid cell corresponds to a bigger region of the original image. Finer, denser grids (like 8×8 or 38×38) are naturally suited to **smaller objects**, because each cell corresponds to a smaller, more zoomed-in region.

> **Analogy:** This is like using **different-sized nets** to catch different-sized fish. A wide-mesh net (coarse grid, e.g. 4×4) is great for catching a big fish (the dog) but would let a tiny minnow (the cat) slip straight through. A fine-mesh net (dense grid, e.g. 8×8) catches the tiny minnow easily, but you'd need several passes to net the big fish since it barely fits any single mesh cell. SSD cleverly uses **both net sizes at once**, so nothing slips through regardless of size.

### 15.4 Loss Function

Same overall idea as YOLO — sum of localization and classification loss:

$$
L = \frac{1}{N} L_{cls} + \alpha L_{loc}
$$

where $N$ is the number of matched bounding boxes (IoU > 0.5) and $\alpha$ balances the two loss terms. Only **positive matches** contribute to the localization loss $L_{loc}$ (it only "penalizes predictions from positive matches").

$$
L_{cls} = -\sum_{i \in Pos} x_{ij}^p \log(\hat{c}_i^p) - \sum_{i \in Neg} \log(\hat{c}_i^0), \quad \hat{c}_i^p = \frac{\exp(c_i^p)}{\sum_p \exp(c_i^p)}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/a9c685b5-6d48-4018-becb-5d45fdbcbaa4">
</p>

- For **positive matches**, the loss penalises according to the confidence score of the **correct class**.
- For **negative matches**, the loss penalises according to the confidence score of class **"0"** — meaning "no object here."

### 15.5 Worked Example: Matching Strategy (Reading the Bicycle Scene)

A scene with a **person on a bicycle** is taken here to illustrate how SSD decides which predicted boxes actually get to influence training.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f1dadbd6-0861-4daf-abcb-d3ad08dd8178" width=1000>
</p>

- SSD predictions are classified as **positive matches** or **negative matches**.
- A prediction is a **positive match** only if its corresponding **default box** (not the predicted box) has IoU > 0.5 with the ground truth.
- In the example, out of three candidate boxes (labelled 1, 2, 3) near the cyclist, only **boxes 1 and 2** turn out to be positive matches — box 3's default box doesn't overlap the ground truth enough, so it is discarded as a negative match and does **not** contribute to the localization loss.

**How to read this kind of diagram in general:** just because a box is drawn near an object doesn't automatically make it "responsible" for that object during training — always check the box's IoU against the ground truth **first**, using the *original*, un-adjusted default box position, not the (possibly already-nudged) prediction.

> **Analogy:** This is like a group project where only students who **signed up early enough** (default box IoU > 0.5) get graded on that specific project — students who showed up too late or too far off-topic (IoU ≤ 0.5) simply aren't counted for that assignment at all, even if their work might look decent. This keeps the grading clean and focused on genuinely relevant contributions.

---

## 16. Common Object Detection Tricks

Several tricks show up repeatedly across R-CNN, Fast/Faster R-CNN, YOLO, and SSD.

### 16.1 Hard Negative Mining

Not all negative examples are **equally hard** to classify:

| Type of negative | Difficulty |
|---|---|
| **Pure empty background** | "Easy negative" — trivially easy to reject |
| **Weird noisy texture / partial object** | "Hard negative" — could easily be mistaken for the real object |

<p align="center">
  <img src="https://github.com/user-attachments/assets/60fca0b5-4f23-4df0-8dd1-85dd1c527565" width=1000>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/fb1cdca1-a971-4456-918a-7afb0cca54fa" width=600>
</p>

We can explicitly **find these false-positive "hard negative" samples during training** and deliberately include more of them in the training data, to sharpen the classifier's decision boundary.

> **Analogy:** Imagine training airport security to spot fake passports. Showing them **1,000 obviously blank sheets of paper** (easy negatives) teaches them very little new. But showing them a **cleverly forged passport that almost looks real** (a hard negative) is exactly the kind of tricky example that actually improves their detection skill. Hard negative mining is about **deliberately seeking out the tricky almost-fooled-me examples** and training on those.

### 16.2 Non-Maximum Suppression (NMS)

After a detector proposes many overlapping boxes for the **same object**, **NMS** picks the single best box and removes ("suppresses") the rest.

**NMS considers two things:** the **confidence score** given by the model, and the **IoU** between overlapping bounding boxes.

**Algorithm (step by step):**

1. Get the set of matched bounding boxes for the **same object category**.
2. **Sort** all the bounding boxes by confidence score.
3. **Discard** boxes with low confidence, and boxes that overlap (high IoU) with an already-kept, higher-confidence box.

<p align="center">
  <img src="https://github.com/user-attachments/assets/97826e20-fb61-4ab7-b51b-15c360f9a7df" width=1000>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/85e6be4b-eef3-4915-9f6d-56fe3bf06902" width=1000>
</p>

> **Analogy:** NMS is like a photo-tagging app where **four different friends all tag the exact same person in a group photo with slightly different bounding boxes**. Instead of showing four redundant tags for one person, the app keeps only the **most confident** tag and quietly deletes the overlapping, lower-confidence duplicates — but if there's a *second, separate* person elsewhere in the photo, that person's tag is kept independently, since it doesn't overlap with the first.

---

## 17. Summary: One-Stage vs Two-Stage, All Models Side by Side

| Model | Type | Region proposals? | Speed | Accuracy | Key idea |
|---|---|---|---|---|---|
| **R-CNN** | Two-stage | External (Selective Search / Edge Boxes) | Very slow (47s/image) | Baseline | CNN feature extraction + SVM + bbox regression, trained separately |
| **Fast R-CNN** | Two-stage | External (Selective Search) | Fast (0.32s/image) | Slightly better | Shares CNN computation via ROI Pooling; end-to-end training |
| **Faster R-CNN** | Two-stage | **Internal** (Region Proposal Network) | Very fast (0.2s/image) | Same as Fast R-CNN | Learns region proposals inside the network — no external algorithm |
| **YOLO** | One-stage | None — direct grid prediction | Real-time (45 fps) | Lower (63.4%) | Splits image into S×S grid; one pass predicts everything |
| **SSD** | One-stage | None — direct multi-scale prediction | Real-time (59 fps) | Higher than YOLO (74.3 mAP) | Uses a pyramidal feature hierarchy to catch objects at multiple scales |

> **Analogy for the whole family tree:** Think of this like the evolution of a review process at a company. **R-CNN** is a slow, fully manual multi-department approval chain. **Fast R-CNN** merges the departments into one meeting but still waits on an outside consultant for the initial shortlist. **Faster R-CNN** brings that shortlisting **in-house**, cutting the wait entirely. **YOLO** and **SSD** skip the whole "shortlist then review" process altogether and make the final call in **one single glance** — YOLO glances at one zoom level, while SSD glances at **several zoom levels at once**, which is part of why it edges out YOLO on accuracy.

---

## 18. Key Takeaways

> [!IMPORTANT]
> **The core things to remember from Week 7:**

1. **Object detection** = localization (where?) + classification (what?) — unlike image classification, which only answers "what?" for the whole image.
2. A plain CNN + fully connected layer **cannot** do detection directly, because the number of objects per image is **variable**, but a standard output layer is **fixed-size**.
3. There are two broad approaches: **traditional** (sliding window + image pyramid, non-deep-learning) and **pre-trained base network + detection framework** (deep learning).
4. Detection frameworks split into **two-stage detectors** (R-CNN family — propose regions, then classify) and **one-stage detectors** (YOLO, SSD — predict boxes and classes directly, in one pass).
5. **IoU (Intersection over Union)** measures how well a predicted box overlaps a ground-truth box; it's used both as a **training signal** (deciding positive/negative examples) and an **evaluation metric**.
6. **R-CNN** trains three separate models (CNN, SVM, regressor) — accurate but painfully slow, since it runs the CNN once per region proposal (~2,000× per image).
7. **Fast R-CNN** fixes this by sharing CNN computation across all proposals using **ROI Pooling**, and training everything **end-to-end** with a single multi-task loss.
8. **Faster R-CNN** removes the last external bottleneck by learning region proposals **inside the network** via a **Region Proposal Network (RPN)** and **anchor boxes**.
9. **YOLO** splits the image into an **S×S grid**, predicting bounding boxes, confidence scores, and class probabilities per cell in a single pass — very fast, but weaker on small/crowded objects.
10. **SSD** improves on YOLO by predicting from **multiple feature-map scales** (a pyramidal hierarchy), so it naturally handles objects of many different sizes — faster **and** more accurate than YOLO.
11. **Hard negative mining** focuses training on the *confusing* negative examples (not the trivially easy ones), which sharpens the classifier's decision boundary.
12. **Non-Maximum Suppression (NMS)** removes duplicate/overlapping detections of the same object, keeping only the highest-confidence box per object.
13. Across the whole R-CNN family, each new model mainly removes a **specific speed bottleneck** left by the previous one — often **without sacrificing accuracy**.

---

## 19. Glossary

| Term | Definition |
|------|-----------|
| **Object Detection** | Locating objects with bounding boxes **and** classifying each one |
| **Object Localization** | Locating objects with bounding boxes only (no class label) |
| **Image Classification** | Predicting a single class label for the whole image |
| **Bounding Box** | A rectangle (defined by position + width/height) marking where an object is |
| **Sliding Window** | A fixed-size window scanned across an image to search for objects |
| **Image Pyramid** | Resizing an image at multiple scales, so a fixed-size window can catch objects of different sizes |
| **Base Network** | A pre-trained classification CNN (e.g. VGG, ResNet) reused as the backbone of a detection framework |
| **Region Proposal** | A candidate bounding box that might contain an object, generated before classification |
| **Selective Search / Edge Boxes** | Classical (non-deep-learning) algorithms used to generate region proposals |
| **Intersection over Union (IoU)** | Overlap ratio between a predicted box and the ground-truth box: Area of Overlap ÷ Area of Union |
| **Two-Stage Detector** | Detector that first proposes regions, then classifies/refines them (e.g. R-CNN family) |
| **One-Stage Detector** | Detector that predicts boxes and classes directly in a single pass (e.g. YOLO, SSD) |
| **R-CNN** | First deep-learning detector: CNN feature extraction + SVM classification + bbox regression, trained separately |
| **Fast R-CNN** | Unifies R-CNN's three models into one end-to-end trained network using ROI Pooling |
| **ROI Pooling** | Converts region proposals of varying sizes into a fixed-size feature map by max-pooling over a fixed grid |
| **Faster R-CNN** | Adds a Region Proposal Network (RPN) so region proposals are generated inside the network |
| **Region Proposal Network (RPN)** | A small network that slides over the feature map and proposes region candidates using anchor boxes |
| **Anchor Box / Anchor Point** | Predefined boxes of various scales/ratios placed at each location ("anchor point") on the feature map |
| **Smooth L1 Loss** | A regression loss that behaves like L2 for small errors but like L1 for large errors, reducing sensitivity to outliers |
| **YOLO (You Only Look Once)** | A one-stage detector that splits the image into a grid and predicts boxes + classes per cell in one pass |
| **Grid Cell** | A section of the S×S grid in YOLO, responsible for predicting objects centred within it |
| **"Responsible" Predictor** | The bounding box (out of B candidates in a cell) with the highest IoU to the ground truth, used for training |
| **SSD (Single Shot Detector)** | A one-stage detector that predicts from multiple feature-map scales (a pyramidal hierarchy) |
| **Pyramidal Feature Hierarchy** | Using feature maps of decreasing size (from a CNN) to detect objects at multiple scales |
| **Default Box** | SSD's term for an anchor box at a given feature-map location and scale |
| **Matching Strategy** | The rule (usually IoU > 0.5 with the default box) used to decide whether a prediction is a positive or negative match |
| **Hard Negative Mining** | Deliberately training on the negative examples that are hardest to classify correctly, not just random/easy ones |
| **Non-Maximum Suppression (NMS)** | Removing duplicate, overlapping detections of the same object, keeping only the highest-confidence box |
| **mAP (mean Average Precision)** | A standard accuracy metric for object detection, averaged across classes |
| **fps (frames per second)** | A standard speed metric for detectors, especially relevant for real-time use cases |

---

> [!TIP]
> **Study tip for Week 7:** Make sure you can:
> 1. Explain, in your own words, why a plain CNN + fully connected layer **cannot** do object detection directly.
> 2. Walk through the **IoU worked example** and explain what counts as a "poor," "good," and "excellent" overlap, and why.
> 3. Trace through the **ROI Pooling worked example** step by step — explain why the output is always a fixed size, regardless of the input region's shape.
> 4. Explain what specific bottleneck each model in the R-CNN family solves compared to the one before it (R-CNN → Fast R-CNN → Faster R-CNN).
> 5. Compare **YOLO** and **SSD**: what do they have in common (one-stage, single pass), and what's the key architectural difference (single grid vs pyramidal multi-scale) that makes SSD more accurate?
> 6. Explain **Non-Maximum Suppression** step by step, and why it's necessary even after a detector already predicts "good" boxes.
> 7. Give your own analogy for **hard negative mining**, and explain why "easy" negatives aren't very useful for training.
> 8. Be able to read a **performance comparison table** (like the R-CNN family speed/accuracy tables) and explain what trade-off (if any) is being made between speed and accuracy.
