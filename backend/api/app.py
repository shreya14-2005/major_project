from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from testing.disease_inference import (
    load_model_bundle,
    load_config,
    predict_image,
)
from predictions.nutrient_detection.predict_nutrient import (
    load_model_and_classes,
    predict_nutrient_deficiency,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("planthealthai")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"
}

disease_lock = Lock()
nutrient_lock = Lock()


@contextmanager
def temporary_upload(upload: UploadFile):
    filename = upload.filename or "image.jpg"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format."
        )

    contents = upload.file.read(MAX_UPLOAD_BYTES + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit."
        )

    temp_path = None

    try:
        with NamedTemporaryFile(
            suffix=suffix,
            prefix="planthealthai_",
            delete=False
        ) as temp:
            temp.write(contents)
            temp_path = Path(temp.name)

        yield temp_path

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading PlantHealthAI models...")

    # Load each model once during application startup.
    app.state.disease_bundle = load_model_bundle()
    app.state.disease_config = load_config()

    app.state.nutrient_model, app.state.nutrient_classes = (
        load_model_and_classes()
    )

    app.state.models_ready = True
    logger.info("Both prediction models are ready.")

    yield

    app.state.models_ready = False


app = FastAPI(
    title="PlantHealthAI API",
    description=(
        "Maize disease classification and visual nutrient "
        "deficiency prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "PlantHealthAI API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health")
def health():
    ready = getattr(app.state, "models_ready", False)

    return {
        "status": "healthy" if ready else "starting",
        "models": {
            "disease": "loaded" if ready else "not_ready",
            "nutrient": "loaded" if ready else "not_ready",
        },
    }


@app.post("/api/v1/disease/predict")
def disease_prediction(file: UploadFile = File(...)):
    if not getattr(app.state, "models_ready", False):
        raise HTTPException(status_code=503, detail="Models are not ready.")

    with temporary_upload(file) as image_path:
        try:
            with disease_lock:
                result = predict_image(
                    image_path,
                    bundle=app.state.disease_bundle,
                    config=app.state.disease_config,
                    explanation_context=None,
                )

            if result["status"] == "rejected":
                raise HTTPException(status_code=400, detail=result["error"])

            return result

        except HTTPException:
            raise
        except Exception:
            logger.exception("Disease prediction failed.")
            raise HTTPException(
                status_code=500,
                detail="Disease prediction failed. Check the backend logs.",
            )


@app.post("/api/v1/nutrients/predict")
def nutrient_prediction(file: UploadFile = File(...)):
    if not getattr(app.state, "models_ready", False):
        raise HTTPException(status_code=503, detail="Models are not ready.")

    with temporary_upload(file) as image_path:
        try:
            with nutrient_lock:
                result = predict_nutrient_deficiency(image_path)

            return result

        except (ValueError, TypeError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error))

        except Exception:
            logger.exception("Nutrient prediction failed.")
            raise HTTPException(
                status_code=500,
                detail="Nutrient prediction failed. Check the backend logs.",
            )
