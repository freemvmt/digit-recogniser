import io
import logging
import os

from fastapi import FastAPI, Request
import numpy as np
import psycopg

from infer import infer


logger = logging.getLogger(__name__)


DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")


api = FastAPI()


# TODO: supply types so that Swagger docs make sense
@api.post("/predict")
async def predict(req: Request) -> dict:
    # read the raw body and rehydrate as numpy array
    blob: bytes = await req.body()
    img = np.load(io.BytesIO(blob))
    logger.info(f"Received image data of shape {img.shape} and dtype {img.dtype}")
    pred, conf = infer(img)
    logger.info(f"Completed inferrence: prediction is {pred}, with confidence: {conf:.2f}")
    return {
        "prediction": pred,
        "confidence": conf,
    }


@api.put("/submit")
async def send_prediction_to_db(req: Request) -> dict:
    """Send submission data to the postgres db."""
    data = await req.json()
    try:
        with psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
        ) as conn:
            with conn.cursor() as cur:
                # insert the submission into the submissions table
                cur.execute(
                    "INSERT INTO submissions (prediction, true_label, confidence) VALUES (%s, %s, %s)",
                    (data["prediction"], data["true_label"], data["confidence"]),
                )
                # commit the transaction
                conn.commit()
    except psycopg.Error as e:
        msg = f"Error connecting to the database or executing query: {e}"
        logger.error(msg)
        return {"status": "error", "message": msg}
    except KeyError as e:
        msg = f"Error with data supplied in request: {e}"
        logger.error(msg)
        return {"status": "error", "message": msg}
    msg = f"Logged prediction {data['prediction']} with label {data['true_label']} and confidence {data['confidence']:.2f} to the database"
    logger.info(msg)
    return {"status": "success", "message": msg}
