"""Streamlit app: upload a chest X-ray → pneumonia prediction + Grad-CAM."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cxr_pneumonia.config import load_config
from cxr_pneumonia.explain import predict_with_gradcam

st.set_page_config(page_title="CXR Pneumonia Detector", layout="wide")


@st.cache_resource
def get_cfg_and_checkpoint():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.data_dir = str((ROOT / cfg.data_dir).resolve())
    cfg.artifacts_dir = str((ROOT / cfg.artifacts_dir).resolve())
    ckpt = cfg.artifacts_path / "best.pt"
    return cfg, ckpt


def main() -> None:
    st.title("Chest X-Ray Pneumonia Detector")
    st.caption("Upload a chest X-ray to get a Normal / Pneumonia prediction and a Grad-CAM explanation.")

    cfg, ckpt = get_cfg_and_checkpoint()
    if not ckpt.exists():
        st.error(
            f"Missing trained model at `{ckpt}`. "
            "Run `python scripts/train.py --config configs/default.yaml` first."
        )
        st.stop()

    uploaded = st.file_uploader(
        "Chest X-ray image",
        type=["jpeg", "jpg", "png"],
        help="Use a frontal chest X-ray (JPEG/PNG).",
    )

    col_left, col_right = st.columns([1, 1])

    if uploaded is None:
        st.info("Upload an image to run detection.")
        with col_left:
            st.subheader("Example from test set")
            sample_dir = Path(cfg.data_dir) / "test" / "PNEUMONIA"
            samples = sorted(sample_dir.glob("*.jpeg"))[:1]
            if samples:
                st.image(str(samples[0]), caption=samples[0].name, use_container_width=True)
                if st.button("Analyze this example"):
                    st.session_state["example_path"] = str(samples[0])
            else:
                st.write("No sample images found. Download data first.")
        image = None
        if "example_path" in st.session_state:
            image = Image.open(st.session_state["example_path"]).convert("RGB")
    else:
        image = Image.open(uploaded).convert("RGB")
        with col_left:
            st.subheader("Uploaded X-ray")
            st.image(image, use_container_width=True)

    if image is None:
        return

    with st.spinner("Running model + Grad-CAM..."):
        result = predict_with_gradcam(cfg, ckpt, image)

    with col_right:
        st.subheader("Prediction")
        label = result["pred_name"]
        prob = result["prob"]
        if label == "PNEUMONIA":
            st.error(f"**{label}** — confidence {prob:.1%}")
        else:
            st.success(f"**{label}** — confidence {prob:.1%}")

        st.write("Class probabilities")
        for name, p in result["probs"].items():
            st.progress(min(max(p, 0.0), 1.0), text=f"{name}: {p:.1%}")

        st.caption(
            "This is a research demo, not a medical diagnosis. "
            "Always consult a qualified clinician."
        )

    st.subheader("Grad-CAM explanation")
    st.write("Hotter regions are where the model focused for its prediction.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(result["rgb"])
    axes[0].set_title("Original")
    axes[1].imshow(result["cam"], cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(result["overlay"])
    axes[2].set_title(f"Overlay — {result['pred_name']} ({result['prob']:.2f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
