import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation
)


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BhesBhusa AI",
    page_icon="👘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

    /* ================================
       COLOR PALETTE (single source)
       ================================ */

    :root {
        --ink-900: #0f172a;   /* strongest headings */
        --ink-700: #1e293b;   /* labels, card titles */
        --ink-500: #475569;   /* body text */
        --ink-400: #64748b;   /* muted / subtitle */
        --line: #e2e8f0;      /* borders */
        --surface: #ffffff;
        --accent: #334155;
    }

    /* ================================
       MAIN PAGE
       ================================ */

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }


    /* ================================
       HEADER
       ================================ */

    .header-wrap {
        text-align: center;
        padding: 0.3rem 0 1.3rem 0;
    }

    .main-title {
        display: block;
        font-size: clamp(2rem, 4.8vw, 2.6rem);
        font-weight: 800;
        line-height: 1.45;          /* extra room so descenders never clip */
        padding: 0.2em 0.1em 0.1em 0.1em;
        margin: 0;
        letter-spacing: -0.3px;
        color: var(--ink-900);
        overflow: visible;
    }

    .main-title-np {
        display: block;
        font-size: clamp(1.15rem, 3vw, 1.4rem);
        font-weight: 700;
        line-height: 1.6;
        margin: 0.1rem 0 0 0;
        color: var(--accent);
    }

    .subtitle {
        text-align: center;
        font-size: clamp(0.85rem, 2vw, 1.02rem);
        color: var(--ink-500);
        margin-top: 0.45rem;
        margin-bottom: 0;
        font-weight: 600;
        letter-spacing: 0.1px;
    }


    /* ================================
       SECTION TITLES
       ================================ */

    .section-title {
        font-size: 1.05rem;
        font-weight: 750;
        color: var(--ink-900);
        margin-top: 1.35rem;
        margin-bottom: 0.55rem;
        border-left: 3px solid var(--accent);
        padding-left: 0.55rem;
    }

    .section-title:first-of-type {
        margin-top: 0.4rem;
    }


    /* ================================
       ABOUT BOX
       ================================ */

    .about-box {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.7rem;
        color: var(--ink-700);
        font-size: 0.9rem;
        line-height: 1.6;
    }


    /* ================================
       FILE UPLOADER
       ================================ */

    [data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed #94a3b8;
        border-radius: 10px;
        padding: 0.35rem 0.6rem;
    }


    /* ================================
       BUTTON
       ================================ */

    .stButton > button {
        border-radius: 9px;
        min-height: 2.8rem;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        margin-bottom: 0.2rem;
    }


    /* ================================
       RESULT CARDS
       ================================ */

    .result-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.6rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    .result-title {
        text-align: center;
        font-size: 0.9rem;
        font-weight: 750;
        color: var(--ink-700);
        margin-bottom: 0.45rem;
        letter-spacing: 0.1px;
    }


    /* ================================
       DETECTED PARTS
       ================================ */

    .detected-box {
        background: var(--surface);
        border: 1px solid #dbe3ed;
        border-radius: 9px;
        padding: 0.6rem;
        text-align: center;
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--ink-700);
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    }


    /* ================================
       LEGEND
       ================================ */

    .legend-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 9px;
        padding: 0.6rem;
        text-align: center;
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 0.1px;
    }

    .legend-daura {
        color: #15803d;
    }

    .legend-surwal {
        color: #1d4ed8;
    }

    .legend-topi {
        color: #b91c1c;
    }


    /* ================================
       FOOTER
       ================================ */

    .footer-line {
        margin-top: 1.6rem;
        margin-bottom: 0.7rem;
        border-top: 1px solid var(--line);
    }

    .footer-title {
        text-align: center;
        color: var(--ink-700);
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .footer-subtitle {
        text-align: center;
        color: var(--ink-400);
        font-size: 0.78rem;
        font-weight: 500;
        padding-bottom: 0.4rem;
    }


    /* ================================
       MOBILE RESPONSIVE
       ================================ */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-top: 0.8rem;
        }

        .main-title {
            font-size: 1.9rem;
            line-height: 1.4;
        }

        .main-title-np {
            font-size: 1.05rem;
        }

        .subtitle {
            font-size: 0.82rem;
            margin-bottom: 0.3rem;
        }

        .section-title {
            font-size: 1rem;
            margin-top: 1.1rem;
        }

        .about-box {
            font-size: 0.83rem;
        }

    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL SETTINGS
# =========================================================
MODEL_PATH = "best_model.pth"

IMAGE_SIZE = 256

CLASS_NAMES = [
    "background",
    "Daura",
    "Surwal",
    "Topi"
]


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="header-wrap">'
    '<div class="main-title">BhesBhusa AI</div>'
    '<div class="main-title-np">भेषभूषा एआई</div>'
    '<div class="subtitle">'
    'AI-Powered Nepali Traditional Costume Segmentation'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ABOUT THE SYSTEM
# =========================================================
st.markdown(
    '<div class="section-title">About the System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="about-box">'
    'BhesBhusa AI is an AI-based image segmentation system '
    'developed for identifying and segmenting Nepali traditional '
    'costume components from images.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():

    config = SegformerConfig.from_pretrained(
        "nvidia/mit-b2",
        num_labels=4,
        id2label={
            0: "background",
            1: "Daura",
            2: "Surwal",
            3: "Topi"
        },
        label2id={
            "background": 0,
            "Daura": 1,
            "Surwal": 2,
            "Topi": 3
        }
    )

    model = SegformerForSemanticSegmentation(
        config
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    return model


# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_np = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image_np = (
        image_np - mean
    ) / std

    tensor = torch.from_numpy(
        image_np.transpose(2, 0, 1)
    ).unsqueeze(0)

    return tensor


# =========================================================
# MODEL PREDICTION
# =========================================================
@torch.no_grad()
def predict(image, model):

    tensor = preprocess_image(
        image
    )

    outputs = model(
        pixel_values=tensor
    )

    logits = outputs.logits

    logits = torch.nn.functional.interpolate(
        logits,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )

    prediction = torch.argmax(
        logits,
        dim=1
    ).squeeze(0).cpu().numpy()

    return prediction


# =========================================================
# CREATE SEGMENTATION MASK
# =========================================================
def create_mask(prediction):

    mask = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        dtype=np.uint8
    )

    # Daura
    mask[prediction == 1] = [
        60,
        180,
        100
    ]

    # Surwal
    mask[prediction == 2] = [
        60,
        130,
        220
    ]

    # Topi
    mask[prediction == 3] = [
        220,
        60,
        60
    ]

    return mask


# =========================================================
# CREATE OVERLAY
# =========================================================
def create_overlay(
    image,
    prediction
):

    original = image.convert(
        "RGB"
    ).resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    original_np = np.array(
        original
    ).astype(
        np.float32
    )

    mask = create_mask(
        prediction
    ).astype(
        np.float32
    )

    overlay = original_np.copy()

    foreground = (
        prediction != 0
    )

    overlay[foreground] = (
        0.55 *
        original_np[foreground]
        +
        0.45 *
        mask[foreground]
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    return overlay


# =========================================================
# GET DETECTED COSTUME PARTS
# =========================================================
def get_detected_classes(
    prediction
):

    detected = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        # Do not show background
        if class_id == 0:
            continue

        if np.any(
            prediction == class_id
        ):

            detected.append(
                class_name
            )

    return detected


# =========================================================
# UPLOAD IMAGE
# =========================================================
st.markdown(
    '<div class="section-title">Upload an Image</div>',
    unsafe_allow_html=True
)

uploader_col1, uploader_col2, uploader_col3 = st.columns(
    [1, 2, 1]
)

with uploader_col2:

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


# =========================================================
# MAIN PROCESS
# =========================================================
if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # =============================================
    # INPUT IMAGE
    # =============================================
    st.markdown(
        '<div class="section-title">Input Image</div>',
        unsafe_allow_html=True
    )

    # Keep preview compact
    preview_col1, preview_col2, preview_col3 = st.columns(
        [1, 2, 1]
    )

    with preview_col2:

        st.image(
            image,
            width="stretch"
        )


    # =============================================
    # START SEGMENTATION
    # =============================================
    if st.button(
        "Start Segmentation",
        type="primary",
        width="stretch"
    ):

        try:

            with st.spinner(
                "Segmenting costume..."
            ):

                model = load_model()

                prediction = predict(
                    image,
                    model
                )

                mask = create_mask(
                    prediction
                )

                overlay = create_overlay(
                    image,
                    prediction
                )

                detected = get_detected_classes(
                    prediction
                )


            st.success(
                "Segmentation completed successfully."
            )


            # =============================================
            # RESULTS
            # =============================================
            st.markdown(
                '<div class="section-title">'
                'Segmentation Results'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(
                3,
                gap="small"
            )


            # ---------------------------------------------
            # ORIGINAL
            # ---------------------------------------------
            with col1:

                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-title">'
                    'Original Image'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image.resize(
                        (
                            IMAGE_SIZE,
                            IMAGE_SIZE
                        ),
                        Image.Resampling.BILINEAR
                    ),
                    width="stretch"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


            # ---------------------------------------------
            # MASK
            # ---------------------------------------------
            with col2:

                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-title">'
                    'Segmentation Mask'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    mask,
                    width="stretch"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


            # ---------------------------------------------
            # OVERLAY
            # ---------------------------------------------
            with col3:

                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-title">'
                    'Segmentation Overlay'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    overlay,
                    width="stretch"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


            # =============================================
            # DETECTED COSTUME PARTS
            # =============================================
            st.markdown(
                '<div class="section-title">'
                'Detected Costume Parts'
                '</div>',
                unsafe_allow_html=True
            )


            if detected:

                detected_cols = st.columns(
                    len(detected),
                    gap="small"
                )

                for col, item in zip(
                    detected_cols,
                    detected
                ):

                    with col:

                        st.markdown(
                            f'<div class="detected-box">'
                            f'{item}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            else:

                st.info(
                    "No costume components were detected."
                )


            # =============================================
            # LEGEND
            # =============================================
            st.markdown(
                '<div class="section-title">'
                'Segmentation Legend'
                '</div>',
                unsafe_allow_html=True
            )


            legend_col1, legend_col2, legend_col3 = st.columns(
                3,
                gap="small"
            )


            with legend_col1:

                st.markdown(
                    '<div class="legend-card '
                    'legend-daura">'
                    '🟢 Daura'
                    '</div>',
                    unsafe_allow_html=True
                )


            with legend_col2:

                st.markdown(
                    '<div class="legend-card '
                    'legend-surwal">'
                    '🔵 Surwal'
                    '</div>',
                    unsafe_allow_html=True
                )


            with legend_col3:

                st.markdown(
                    '<div class="legend-card '
                    'legend-topi">'
                    '🔴 Topi'
                    '</div>',
                    unsafe_allow_html=True
                )


        except Exception as e:

            st.error(
                "An error occurred during segmentation."
            )

            st.exception(e)


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer-line"></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer-title">'
    'BhesBhusa AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer-subtitle">'
    'Nepali Traditional Costume Segmentation &amp; Identification'
    '</div>',
    unsafe_allow_html=True
)