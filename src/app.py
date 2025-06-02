import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn

from cnn import BaseCNN
from utils import (
    get_image_from_tensor,
    transform_image_to_tensor,
)


logger = logging.getLogger(__name__)


# choose model by exposed hyperparameters / accuracy on test set
HOT_SWAP_SCALE_FACTOR = 16
HOT_SWAP_EPOCHS = 20
HOT_SWAP_ACC = 99
MODEL_WEIGHTS_PATH = (
    f"models/mnist_cnn.{HOT_SWAP_SCALE_FACTOR}.{HOT_SWAP_EPOCHS}.{HOT_SWAP_ACC}.pth"
)

MNIST_IMG_DIMENSIONS = (28, 28)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DEBUG_MODE = os.getenv("DEBUG_MODE") is not None


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str = MODEL_WEIGHTS_PATH) -> nn.Module:
    """Load trained weights once and cache the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseCNN(factor=HOT_SWAP_SCALE_FACTOR).to(device)
    if Path(weights_path).is_file():
        state = torch.load(weights_path, device)
        model.load_state_dict(state)
    else:
        raise ValueError(f"⚠️ Weight file not found at {MODEL_WEIGHTS_PATH}")
    # ensure model is in evaluation mode before handing off to Streamlit
    model.eval()
    return model


def get_greyscale_image(img: np.ndarray) -> Image.Image:
    """Convert a RGBA image from canvas to a greyscale PIL image."""
    # select red channel (all channels should be identical so any will do) and make greyscale
    return Image.fromarray(img[:, :, 0]).convert("L")


def preprocess(img: np.ndarray) -> torch.Tensor:
    """Canvas RGBA → 1-batch 1x28x28 tensor ready for the CNN."""
    return transform_image_to_tensor(
        img=get_greyscale_image(img),
        resize=MNIST_IMG_DIMENSIONS,
    ).unsqueeze(0)  # add singleton batch dimension


# load model with pre-trained parameters/state
model = load_model()

# define UI layout
st.set_page_config(
    page_title="Digit recogniser",
    page_icon="🎱",
    layout="centered",
)

st.title("Guess my digits")
st.write("Can this machine recognise your mouse-writing? 🤔")
st.write("Draw a single digit (0-9) in the box, and we'll try and guess it!")

"---"

# prepare to keep submission history in session state (does not persist)
if "history" not in st.session_state:
    st.session_state.history = []

# define structure of page as 2 columns
col_left, col_right = st.columns([1, 1])

with col_left:
    # canvas component
    canvas_result = st_canvas(
        # black background, white ink as per MNIST data
        fill_color="rgba(0,0,0,1)",
        stroke_color="rgba(255,255,255,1)",
        # use thick brush stroke to preserve detail after downsizing
        stroke_width=16,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict"):
        if canvas_result.image_data is None:
            logger.debug(
                "Predict button clicked, but no image data available from canvas."
            )
            st.warning("Sorry, we can't guess at a blank slate!")
        else:
            tensor = preprocess(canvas_result.image_data.astype("uint8"))
            # we include some logic to visualise what the model will see, for dev/debugging purposes
            if DEBUG_MODE:
                img = get_image_from_tensor(tensor)
                st.image(
                    img,
                    caption="Preprocessed image to be fed to model (de-normalised for inspection)",
                )
            with torch.no_grad():
                logger.debug("Running inference")
                logits = model(tensor)
                # use softmax to convert logits to a list of probabilities per class (read: digit)
                probs = torch.softmax(logits, dim=1)[0].numpy()
            # get the digit with the highest probability (which we also interpret as 'confidence')
            logger.debug(f"Predicted probabilities: {probs}")
            prediction = int(probs.argmax())
            confidence = float(probs[prediction])

            # save results to state right column and history
            st.session_state.current_pred = prediction
            st.session_state.current_conf = confidence
            st.session_state.current_img = get_greyscale_image(
                canvas_result.image_data.astype("uint8")
            )
            # (re)set feedback submission state to allow feedback
            st.session_state.feedback_submitted = False

with col_right:
    if "current_pred" in st.session_state:
        st.subheader("Prediction")
        st.html(
            f"<h1 style='text-align: center;'>{st.session_state.current_pred}</h1>",
        )
        st.html(
            f"<p style='text-align: center; color: #10B981;'>↑ {st.session_state.current_conf:.1%}</p>",
        )

        truth_label = st.number_input(
            "Please indicate which number you drew:",
            min_value=0,
            max_value=9,
            step=1,
            value=st.session_state.current_pred,
            key="truth_label_input",
        )

        # determine if the feedback button should be disabled (i.e. waiting on new prediction)
        disable_feedback_button = st.session_state.get("feedback_submitted", False)
        if st.button(
            "Submit feedback",
            use_container_width=True,
            disabled=disable_feedback_button,
        ):
            dt = datetime.datetime.now()
            st.session_state.history.insert(
                0,
                {
                    "Timestamp": dt.strftime(TIMESTAMP_FORMAT),
                    "Prediction": st.session_state.current_pred,
                    "Truth": truth_label,
                    "Confidence": f"{st.session_state.current_conf:.1%}",
                },
            )
            # mark feedback as submitted to disable button and rerun to reflect immediately
            st.session_state.feedback_submitted = True
            st.rerun()

if "feedback_submitted" in st.session_state and st.session_state.feedback_submitted:
    st.success("Feedback recorded - thanks!", icon="✅")
    if st.button("Go again!", use_container_width=True):
        # clear most of the session state (but keep history)
        for key in (
            "canvas",
            "current_pred",
            "current_conf",
            "current_img",
            "feedback_submitted",
        ):
            del st.session_state[key]
        st.rerun()

# show submissions for current session in a table
if st.session_state.history:
    st.subheader("Recent submissions")
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

if DEBUG_MODE:
    st.session_state
