import io

import numpy as np
import requests


BACKEND_BASE_URL = "http://backend:8000"
PREDICTION_ENDPOINT = "/predict"
SUBMISSION_ENDPOINT = "/submit"


def get_image_as_blob(img: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    # dump image-as-numpy-array in .npy format to preserve dtype/shape
    np.save(buffer, img)
    buffer.seek(0)
    return buffer.read()


def get_prediction(img: np.ndarray) -> dict[str, int | float]:
    """Query the backend container with the image data."""
    res = requests.post(
        url=BACKEND_BASE_URL + PREDICTION_ENDPOINT,
        data=get_image_as_blob(img),
        headers={
            "Content-Type": "application/octet-stream",
            "Accept": "application/json",
        },
    )
    if res.status_code != 200:
        raise ValueError(
            f"Backend {PREDICTION_ENDPOINT} returned error {res.status_code}: {res.text}"
        )
    return res.json()


def put_submission(
    prediction: int,
    true_label: int,
    confidence: float,
) -> dict[str, str]:
    """Send a user submission to the db (via backend) for persistence."""
    res = requests.put(
        url=BACKEND_BASE_URL + SUBMISSION_ENDPOINT,
        json={
            "prediction": prediction,
            "true_label": true_label,
            "confidence": confidence,
        },
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    if res.status_code != 200:
        raise ValueError(
            f"Backend {SUBMISSION_ENDPOINT} returned error {res.status_code}: {res.text}"
        )
    return res.json()
