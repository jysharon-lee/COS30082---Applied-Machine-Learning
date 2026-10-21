import json
import time
from pathlib import Path
from PIL import Image
import gradio as gr
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
        print(f"Warning: Checkpoint '{ckpt_path}' not found! Running in uninitialized state.")

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
# 2. Inference Logic
# -----------------------------------------------------------------------------
def classify_bird(image):
    if image is None:
        return None, {}, "No image provided."

    start_time = time.perf_counter()
    rgb_image = image.convert("RGB")
    tensor = EVAL_TRANSFORM(rgb_image).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    top5_probs, top5_indices = torch.topk(probs, k=5)
    top1_species = CLASS_NAMES[top5_indices[0].item()]
    top1_conf = top5_probs[0].item() * 100

    predictions = {
        CLASS_NAMES[idx.item()]: float(prob.item())
        for idx, prob in zip(top5_indices, top5_probs)
    }

    callout_html = f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
        <span style="font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; color: #64748B; font-weight: 600;">Top Prediction</span>
        <div style="font-size: 1.4rem; font-weight: 700; color: #0F172A; margin: 4px 0;">{top1_species}</div>
        <div style="font-size: 0.95rem; color: #2563EB; font-weight: 600;">{top1_conf:.2f}% Confidence Score</div>
    </div>
    """

    latency_str = f"Inference Latency: {elapsed_ms:.1f} ms (PyTorch CPU)"
    return callout_html, predictions, latency_str


# -----------------------------------------------------------------------------
# 3. UI Styling & Layout
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
/* Clean Typography & Container */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0F172A;
}

/* Metric Cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: left;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
.metric-num {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1E293B;
    line-height: 1.2;
}
.metric-desc {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-top: 4px;
    font-weight: 600;
}

/* Section Header */
.header-box {
    padding: 10px 0 24px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 20px;
}
.header-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0;
}
.header-sub {
    font-size: 0.95rem;
    color: #475569;
    margin-top: 6px;
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

    # Header
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
            <div class="metric-desc">Model Architecture</div>
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
            with gr.Row():
                btn_clear = gr.Button("Clear", variant="secondary")
                btn_submit = gr.Button("Classify Image", variant="primary")

        with gr.Column(scale=5):
            output_callout = gr.HTML(label="Primary Result")
            output_label = gr.Label(num_top_classes=5, label="Top 5 Predicted Species Distribution")
            output_latency = gr.Markdown(value="", elem_id="latency-text")

    # Wire Events
    btn_submit.click(
        fn=classify_bird,
        inputs=[input_image],
        outputs=[output_callout, output_label, output_latency],
    )
    input_image.change(
        fn=classify_bird,
        inputs=[input_image],
        outputs=[output_callout, output_label, output_latency],
    )
    btn_clear.click(
        fn=lambda: (None, None, {}, ""),
        inputs=[],
        outputs=[input_image, output_callout, output_label, output_latency],
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
