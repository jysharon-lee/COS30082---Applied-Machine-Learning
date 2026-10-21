import json
import time
from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchvision import models, transforms

# -----------------------------------------------------------------------------
# 1. Model & Label Loading (Robust path resolution)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "src" else BASE_DIR

CLASS_MAP_PATH = BASE_DIR / "classes.json" if (BASE_DIR / "classes.json").exists() else PROJECT_ROOT / "classes.json"
CKPT_PATH = BASE_DIR / "swin_t_best.pth" if (BASE_DIR / "swin_t_best.pth").exists() else PROJECT_ROOT / "swin_t_best.pth"

if not CLASS_MAP_PATH.exists():
    raise FileNotFoundError(f"Missing 'classes.json'. Looked in: {CLASS_MAP_PATH}")

with open(CLASS_MAP_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


def load_classifier(ckpt_path: Path):
    model = models.swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))

    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint '{ckpt_path}' not found! Running in demo uninitialized mode.")

    model.eval()
    return model


MODEL = load_classifier(CKPT_PATH)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# 2. Explainable AI: Grad-CAM Saliency Heatmap for Swin-T
# -----------------------------------------------------------------------------
def compute_attention_heatmap(model, input_tensor, target_class_idx, orig_image):
    """
    Computes class-activation map (Grad-CAM) at Stage 4 of Swin-T
    and overlays it onto the original image.
    """
    features, gradients = [], []

    def f_hook(module, inp, out):
        features.append(out)

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook into deepest Swin Transformer block (Stage 4)
    target_layer = model.features[7]
    h_f = target_layer.register_forward_hook(f_hook)
    h_b = target_layer.register_full_backward_hook(b_hook)

    model.zero_grad()
    x = input_tensor.clone().detach().requires_grad_(True)
    out = model(x)
    loss = out[0, target_class_idx]
    loss.backward()

    h_f.remove()
    h_b.remove()

    if not features or not gradients:
        return orig_image.resize((224, 224))

    feat = features[0]   # Shape: (1, 7, 7, 768)
    grad = gradients[0]  # Shape: (1, 7, 7, 768)

    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * feat).sum(dim=-1).squeeze(0)
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Upsample heatmap from 7x7 to 224x224
    cam_up = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze().detach().numpy()

    # Map to heatmap colors (inferno colormap)
    colormap = cm.get_cmap("inferno")
    heatmap = colormap(cam_up)[:, :, :3]

    # Blend with original resized image
    base_img = np.array(orig_image.resize((224, 224)).convert("RGB")) / 255.0
    overlay = 0.52 * base_img + 0.48 * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


# -----------------------------------------------------------------------------
# 3. Main Inference Logic (with OOD Uncertainty Detection)
# -----------------------------------------------------------------------------
OOD_CONFIDENCE_THRESHOLD = 30.0  # Percentage threshold for OOD warning

def classify_bird(image):
    if image is None:
        return None, None, {}, "No image provided."

    start_time = time.perf_counter()
    rgb_image = image.convert("RGB")
    tensor = EVAL_TRANSFORM(rgb_image).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    top5_probs, top5_indices = torch.topk(probs, k=5)
    top1_idx = top5_indices[0].item()
    top1_species = CLASS_NAMES[top1_idx]
    top1_conf = top5_probs[0].item() * 100

    # 1. Generate Grad-CAM Heatmap
    heatmap_img = compute_attention_heatmap(MODEL, tensor, top1_idx, rgb_image)

    # 2. Top-5 Distribution Dictionary
    predictions = {
        CLASS_NAMES[idx.item()]: float(prob.item())
        for idx, prob in zip(top5_indices, top5_probs)
    }

    # 3. Out-of-Distribution (OOD) / Uncertainty Check
    if top1_conf < OOD_CONFIDENCE_THRESHOLD:
        status_html = f"""
        <div class="card-ood">
            <div class="callout-header">Uncertainty Notice</div>
            <div class="ood-title">Potential Out-of-Distribution Input</div>
            <div class="ood-desc">
                Maximum model confidence is {top1_conf:.1f}%, which falls below the reliability threshold ({OOD_CONFIDENCE_THRESHOLD:.0f}%).
                The input may not belong to the 200 North American CUB-200 species, or contains heavy visual occlusion.
            </div>
            <div class="ood-sub">Top candidate: <strong>{top1_species}</strong> ({top1_conf:.1f}%)</div>
        </div>
        """
    else:
        status_html = f"""
        <div class="card-success">
            <div class="callout-header">Primary Taxonomy Classification</div>
            <div class="species-title">{top1_species}</div>
            <div class="conf-badge">{top1_conf:.2f}% Confidence Interval</div>
        </div>
        """

    latency_str = f"Inference Latency: {elapsed_ms:.1f} ms (PyTorch CPU)"
    return status_html, heatmap_img, predictions, latency_str


# -----------------------------------------------------------------------------
# 4. Curated Example Images (Discovered dynamically)
# -----------------------------------------------------------------------------
def get_example_images():
    search_dirs = [PROJECT_ROOT / "AML Assignment 1" / "Train", BASE_DIR / "AML Assignment 1" / "Train"]
    target_names = ["Cardinal", "Blue_Jay", "Baltimore_Oriole", "Mallard", "Laysan_Albatross"]
    found_examples = []

    for s_dir in search_dirs:
        if s_dir.exists():
            for target in target_names:
                matches = list(s_dir.glob(f"{target}*"))
                if matches and matches[0].exists():
                    found_examples.append([str(matches[0])])
            if found_examples:
                break
    return found_examples


EXAMPLE_IMAGES = get_example_images()

# -----------------------------------------------------------------------------
# 5. UI Layout & High-Contrast Adaptive CSS
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
/* Global Typography */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Header Banner with high contrast */
.header-box {
    padding: 12px 0 22px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    margin-bottom: 22px;
}
.header-title {
    font-size: 1.95rem;
    font-weight: 800;
    color: #F8FAFC !important;
    letter-spacing: -0.02em;
    margin: 0;
}
.header-sub {
    font-size: 0.98rem;
    color: #94A3B8 !important;
    margin-top: 6px;
    line-height: 1.5;
}

/* Metric Cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.metric-box {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 14px 16px;
    text-align: left;
}
.metric-num {
    font-size: 1.55rem;
    font-weight: 800;
    color: #38BDF8 !important;
    line-height: 1.2;
}
.metric-desc {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #CBD5E1 !important;
    margin-top: 4px;
    font-weight: 600;
}

/* Primary Result Callout */
.card-success {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.card-ood {
    background: rgba(45, 20, 20, 0.85);
    border: 1px solid rgba(239, 68, 68, 0.45);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.callout-header {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    font-weight: 700;
}
.species-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #F8FAFC !important;
    margin: 6px 0;
}
.conf-badge {
    font-size: 0.95rem;
    color: #38BDF8;
    font-weight: 700;
}
.ood-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #FCA5A5 !important;
    margin: 4px 0;
}
.ood-desc {
    font-size: 0.88rem;
    color: #E2E8F0;
    line-height: 1.45;
    margin-top: 4px;
}
.ood-sub {
    font-size: 0.88rem;
    color: #F87171;
    margin-top: 8px;
}

@media (max-width: 768px) {
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
}
"""

THEME = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="CUB-200 Bird Classifier") as demo:

    # Header with high-contrast text
    gr.HTML("""
    <div class="header-box">
        <h1 class="header-title">Caltech-UCSD Birds 200 Classification System</h1>
        <div class="header-sub">Fine-grained multi-class taxonomy inference using a hierarchical Swin Transformer (Swin-T).</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-num">86.05%</div>
            <div class="metric-desc">Test Top-1 Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-num">85.78%</div>
            <div class="metric-desc">Per-Class Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-num">96.10%</div>
            <div class="metric-desc">Test Top-5 Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-num">Swin-T</div>
            <div class="metric-desc">Backbone Architecture</div>
        </div>
    </div>
    """)

    # Interactive Workbench
    with gr.Row():
        with gr.Column(scale=5):
            input_image = gr.Image(
                type="pil",
                label="Input Image (Upload or Select Benchmark Example)",
                sources=["upload", "clipboard"],
                height=320,
            )
            with gr.Row():
                btn_clear = gr.Button("Clear", variant="secondary")
                btn_submit = gr.Button("Classify Image", variant="primary")

            # Feature #5: Clickable Example Gallery
            if EXAMPLE_IMAGES:
                gr.Examples(
                    examples=EXAMPLE_IMAGES,
                    inputs=input_image,
                    label="Curated Benchmark Examples (Click to Evaluate)",
                    examples_per_page=5,
                )

        with gr.Column(scale=5):
            # Feature #3: Top result callout or OOD Uncertainty Warning
            output_status = gr.HTML(label="Classification Status")

            with gr.Tabs():
                with gr.TabItem("Top 5 Predictions"):
                    output_label = gr.Label(num_top_classes=5, label="Probability Distribution")
                with gr.TabItem("Model Attention (XAI)"):
                    # Feature #1: Attention / Saliency Heatmap
                    output_heatmap = gr.Image(type="pil", label="Swin-T Stage 4 Attention Heatmap (Grad-CAM)")
                    gr.Markdown("Visual representation of where the Swin Transformer focused attention to distinguish fine-grained features (beak, crown, plumage).")

            output_latency = gr.Markdown(value="", elem_id="latency-text")

    # Wire Events
    btn_submit.click(
        fn=classify_bird,
        inputs=[input_image],
        outputs=[output_status, output_heatmap, output_label, output_latency],
    )
    input_image.change(
        fn=classify_bird,
        inputs=[input_image],
        outputs=[output_status, output_heatmap, output_label, output_latency],
    )
    btn_clear.click(
        fn=lambda: (None, None, None, {}, ""),
        inputs=[],
        outputs=[input_image, output_status, output_heatmap, output_label, output_latency],
    )

    # Technical Notes Drawer
    with gr.Accordion("Technical Specifications & Model Benchmarks", open=False):
        gr.Markdown(r"""
        ### Experimental Results Summary (CUB-200-2011)
        The following benchmarks were evaluated across 200 fine-grained bird species under identical data partitions:

        | Architecture | Parameters | Resolution | Test Top-1 | Test Avg Per-Class | Test Top-5 |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **ResNet-101 (Baseline)** | 44.5 M | 224 px | 84.50% | 84.10% | 95.50% |
        | **EfficientNet-B2 (Native)** | 8.0 M | 260 px | 82.06% | 81.59% | 95.02% |
        | **Swin-T (Champion Model)** | **27.7 M** | **224 px** | **86.05%** | **85.78%** | **96.10%** |

        ### Optimization Details
        - **Objective:** Cross-Entropy Loss with Label Smoothing ($\epsilon = 0.1$).
        - **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, Weight Decay $= 0.05$).
        - **Scheduler:** Cosine Annealing with 2-epoch linear warmup.
        """)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS, share=True)
