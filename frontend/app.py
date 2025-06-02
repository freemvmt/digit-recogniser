import datetime
import logging

import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import (
  get_prediction,
  put_submission,
)


logger = logging.getLogger(__name__)


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


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
            logger.info(
                "Predict button clicked, but no image data available from canvas."
            )
            st.warning("Sorry, we can't guess at a blank slate!")
        else:
            result = get_prediction(canvas_result.image_data.astype("uint8"))
            # save results to state right column and history
            st.session_state.current_pred = result["prediction"]
            st.session_state.current_conf = result["confidence"]
            st.session_state.current_img = canvas_result.image_data.astype("uint8")
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

        true_label = st.number_input(
            "Please indicate which number you drew:",
            min_value=0,
            max_value=9,
            step=1,
            value=st.session_state.current_pred,
            key="true_label_input",
        )

        # determine if the feedback button should be disabled (i.e. waiting on new prediction)
        disable_feedback_button = st.session_state.get("feedback_submitted", False)
        if st.button(
            "Submit feedback",
            use_container_width=True,
            disabled=disable_feedback_button,
        ):
            dt = datetime.datetime.now()
            # add the submission to session history, to be displayed on the page
            st.session_state.history.insert(
                0,
                {
                    "Timestamp": dt.strftime(TIMESTAMP_FORMAT),
                    "Prediction": st.session_state.current_pred,
                    "Correct digit": true_label,
                    "Confidence": f"{st.session_state.current_conf:.1%}",
                },
            )
            # also log the submission to the database via backend API
            # TODO: also save the drawn digit (e.g. as base64), for future training runs!
            result = put_submission(
                prediction=st.session_state.current_pred,
                true_label=true_label,
                confidence=st.session_state.current_conf,
            )

            # mark feedback as submitted to disable button and rerun to reflect immediately
            st.session_state.feedback_submitted = True
            st.rerun()

if "feedback_submitted" in st.session_state and st.session_state.feedback_submitted:
    st.success("Feedback recorded - thanks!", icon="✅")
    if st.button("Go again!", use_container_width=True):
        # clear most of the session state (but keep history)
        # FIXME: this doesn't clear the actual canvas, user has to additionally click the trash can
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
    # create dataframe with 10 most recent submissions
    df_hist = pd.DataFrame(st.session_state.history[:10])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
