import json
import time
from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchvision import models, transforms


# 1. Path & Label Setup

BASE_DIR = Path(__file__).resolve().parent

def locate_file(filename):
    for p in [BASE_DIR / filename, BASE_DIR.parent / filename, Path(filename)]:
        if p.exists():
            return p
    return BASE_DIR / filename

CLASS_MAP_PATH = locate_file("classes.json")
if not CLASS_MAP_PATH.exists():
    raise FileNotFoundError(f"Missing 'classes.json'. Looked in: {CLASS_MAP_PATH}")

with open(CLASS_MAP_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


# 2. Multi-Model Architecture Loader & Registry

MODEL_SPECS = {
    "Swin-T (Champion ViT)": {
        "file": "swin_t_best.pth",
        "res": 224,
        "acc": "86.05%",
        "builder": lambda: models.swin_t(weights=None),
        "head_attr": "head",
        "cam_target": lambda m: m.features[7],
        "is_vit": True,
    },
    "ResNet-101 (Deep CNN)": {
        "file": "resnet101_best.pth",
        "res": 224,
        "acc": "84.50%",
        "builder": lambda: models.resnet101(weights=None),
        "head_attr": "fc",
        "cam_target": lambda m: m.layer4,
        "is_vit": False,
    },
    "EfficientNet-B2 (Native 260px)": {
        "file": "efficientnet_b2_native260_best.pth",
        "alt_file": "efficientnet_b2_best.pth",
        "res": 260,
        "acc": "84.70%",
        "builder": lambda: models.efficientnet_b2(weights=None),
        "head_attr": "classifier",
        "cam_target": lambda m: m.features[-1],
        "is_vit": False,
    },
}

LOADED_MODELS = {}

def get_model(arch_name):
    if arch_name in LOADED_MODELS:
        return LOADED_MODELS[arch_name]

    spec = MODEL_SPECS[arch_name]
    m = spec["builder"]()

    # Modify classification head for 200 classes
    head_attr = spec["head_attr"]
    if head_attr == "classifier":
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASS_NAMES))
    elif head_attr == "fc":
        m.fc = nn.Linear(m.fc.in_features, len(CLASS_NAMES))
    elif head_attr == "head":
        m.head = nn.Linear(m.head.in_features, len(CLASS_NAMES))

    # Locate checkpoint
    ckpt_file = locate_file(spec["file"])
    if not ckpt_file.exists() and "alt_file" in spec:
        ckpt_file = locate_file(spec["alt_file"])

    is_loaded = False
    if ckpt_file.exists():
        state_dict = torch.load(ckpt_file, map_location=torch.device("cpu"), weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        m.load_state_dict(state_dict)
        is_loaded = True
        print(f"Loaded {arch_name} from {ckpt_file}")
    else:
        print(f"Notice: Checkpoint for {arch_name} not found at {ckpt_file}.")

    m.eval()
    LOADED_MODELS[arch_name] = (m, is_loaded, spec["res"])
    return LOADED_MODELS[arch_name]

# Preload default champion model
get_model("Swin-T (Champion ViT)")

# Transforms
def get_transform(resolution):
    return transforms.Compose([
        transforms.Resize(int(resolution * 1.14)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

TRANSFORMS = {
    224: get_transform(224),
    260: get_transform(260),
}


# 3. Explainable AI: Universal Grad-CAM Heatmap

def compute_attention_heatmap(model, input_tensor, target_class_idx, orig_image, spec):
    try:
        features, gradients = [], []

        def f_hook(module, inp, out):
            features.append(out)

        def b_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = spec["cam_target"](model)
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
            return orig_image.resize((spec["res"], spec["res"]))

        if spec["is_vit"]:
            weights = gradients[0].mean(dim=(1, 2), keepdim=True)
            cam = (weights * features[0]).sum(dim=-1).squeeze(0)
        else:
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam = (weights * features[0]).sum(dim=1).squeeze(0)

        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam_up = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(spec["res"], spec["res"]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().detach().numpy()

        try:
            colormap = matplotlib.colormaps["inferno"]
        except Exception:
            colormap = cm.get_cmap("inferno")
        heatmap = colormap(cam_up)[:, :, :3]

        base_img = np.array(orig_image.resize((spec["res"], spec["res"])).convert("RGB")) / 255.0
        overlay = 0.52 * base_img + 0.48 * heatmap
        overlay = np.clip(overlay, 0.0, 1.0)
        return Image.fromarray((overlay * 255).astype(np.uint8))
    except Exception as e:
        print(f"Heatmap computation notice: {e}")
        return None


# 4. Inference Logic

def classify_bird(image, selected_arch):
    if image is None:
        return "", None, None, ""

    model, is_loaded, res = get_model(selected_arch)
    spec = MODEL_SPECS[selected_arch]

    if not is_loaded:
        warn_html = f"""
        <div style="background-color: rgba(234, 179, 8, 0.15); border: 1px solid rgba(234, 179, 8, 0.5); border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; color: #FEF08A;">
            <strong>Checkpoint Notice:</strong> The checkpoint for <strong>{selected_arch}</strong> (<code>{spec['file']}</code>) was not found in <code>src/</code>.
            Please download <code>{spec['file']}</code> from your Google Drive into the <code>src/</code> directory to evaluate this model.
        </div>
        """
        return warn_html, None, {}, f"Model {selected_arch} weights missing in src/"

    start_time = time.perf_counter()
    rgb_image = image.convert("RGB")
    tf = TRANSFORMS[res]
    tensor = tf(rgb_image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    top5_probs, top5_indices = torch.topk(probs, k=5)
    top1_species = CLASS_NAMES[top5_indices[0].item()]
    top1_conf = top5_probs[0].item() * 100

    # Predictions dictionary for Gradio Label
    predictions = {
        CLASS_NAMES[idx.item()]: float(prob.item())
        for idx, prob in zip(top5_indices, top5_probs)
    }

    # Generate Attention Heatmap for the selected architecture
    heatmap_img = compute_attention_heatmap(model, tensor, top5_indices[0].item(), rgb_image, spec)

    callout_html = f"""
    <div style="background-color: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 8px; padding: 18px 20px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Top Prediction</span>
            <span style="font-size: 0.75rem; letter-spacing: 0.05em; background: rgba(56, 189, 248, 0.15); color: #38BDF8; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{selected_arch.split()[0]}</span>
        </div>
        <div style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin: 4px 0;">{top1_species}</div>
        <div style="font-size: 1.05rem; color: #38BDF8; font-weight: 700;">{top1_conf:.2f}% Confidence Score</div>
    </div>
    """

    latency_str = f"Architecture: {selected_arch} | Resolution: {res}px | Latency: {elapsed_ms:.1f} ms (PyTorch CPU)"
    return callout_html, heatmap_img, predictions, latency_str


# 5. High-Contrast Dark-Mode Compatible CSS & UI

CUSTOM_CSS = """
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
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
        <div class="header-sub">Fine-grained multi-class taxonomy inference with Explainable AI and multi-architecture evaluation.</div>
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
            <div class="metric-desc">Champion Backbone</div>
        </div>
    </div>
    """)

    # Interactive Workbench
    with gr.Row():
        with gr.Column(scale=5):
            input_image = gr.Image(
                type="pil",
                label="Input Image",
                sources=["upload", "clipboard"],
                height=320,
            )
            # Multi-Model Architecture Comparator Dropdown
            selected_model = gr.Dropdown(
                choices=list(MODEL_SPECS.keys()),
                value="Swin-T (Champion ViT)",
                label="Select Architecture for Inference",
                info="Toggle between distinct model families evaluated in the assignment."
            )
            with gr.Row():
                btn_clear = gr.ClearButton(value="Clear", variant="secondary")
                btn_submit = gr.Button("Classify Image", variant="primary")

        with gr.Column(scale=5):
            output_callout = gr.HTML(label="Primary Result")
            with gr.Tabs():
                with gr.TabItem("Top 5 Predictions"):
                    output_label = gr.Label(num_top_classes=5, label="Probability Distribution")
                with gr.TabItem("Model Attention (XAI)"):
                    output_heatmap = gr.Image(type="pil", label="Visual Attention Saliency Heatmap (Grad-CAM)")
                    gr.Markdown("Visual saliency representation illustrating the anatomical regions (e.g. beak shape, wing bars, crown crest) that drove the selected model's prediction.")

            output_latency = gr.Markdown(value="", elem_id="latency-text")

    # Wire Events
    btn_submit.click(
        fn=classify_bird,
        inputs=[input_image, selected_model],
        outputs=[output_callout, output_heatmap, output_label, output_latency],
    )
    input_image.change(
        fn=classify_bird,
        inputs=[input_image, selected_model],
        outputs=[output_callout, output_heatmap, output_label, output_latency],
    )
    selected_model.change(
        fn=classify_bird,
        inputs=[input_image, selected_model],
        outputs=[output_callout, output_heatmap, output_label, output_latency],
    )
    # Wire ClearButton to reset input and all output panels cleanly
    btn_clear.add([input_image, output_callout, output_heatmap, output_label, output_latency])

    # Technical Notes Drawer
    with gr.Accordion("Technical Specifications & Model Benchmarks", open=False):
        gr.Markdown("""
        ### Experimental Results Summary (CUB-200-2011)
        The following benchmarks were evaluated across 200 fine-grained bird species under identical data partitions:

        | Architecture | Parameters | Resolution | Test Top-1 | Test Avg Per-Class | Test Top-5 |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **ResNet-101 (Baseline)** | 44.5 M | 224 px | 84.50% | 84.10% | 95.50% |
        | **EfficientNet-B2 (Native)** | 8.0 M | 260 px | 82.06% | 81.59% | 95.02% |
        | **Swin-T (Champion Model)** | **27.7 M** | **224 px** | **86.05%** | **85.78%** | **96.10%** |

        ### Optimization Details
        - **Objective:** Cross-Entropy Loss with Label Smoothing (smoothing factor: 0.1)
        - **Optimizer:** AdamW (beta1: 0.9, beta2: 0.999, weight decay: 0.05)
        - **Scheduler:** Cosine Annealing with 2-epoch linear warmup
        """)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS, share=True)
