import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import traceback

# Use absolute imports
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_path = str(Path(__file__).parent.parent)
if backend_path not in sys.path:
    sys.path.append(backend_path)

from app.config import settings
from app.utils.logging_config import configure_logging
from app.models.inference_model import load_model
from app.routers import health, predict

# Configure logging
configure_logging(log_level="DEBUG" if settings.DEBUG else "INFO")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="API for detecting sickle cells in blood smear images",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Model loading check
def check_model_loaded():
    """Check if the model is loaded and log detailed information."""
    from app.models.inference_model import model as ml_model
    import traceback
    
    if not hasattr(ml_model, 'model') or ml_model.model is None:
        logger.error("Model is not loaded!")
        return False
    
    try:
        # Test if model is callable
        import numpy as np
        test_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
        _ = ml_model.model.predict(test_input)
        logger.info("Model is loaded and callable")
        return True
    except Exception as e:
        logger.error(f"Model loading test failed: {str(e)}\n{traceback.format_exc()}")
        return False

# Startup event
@app.on_event("startup")
async def startup():
    logger.info("Starting up application...")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    
    # Check if model file exists
    import os
    if not os.path.exists(settings.MODEL_PATH):
        logger.error(f"Model file not found at: {settings.MODEL_PATH}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Absolute model path: {os.path.abspath(settings.MODEL_PATH)}")
    else:
        logger.info(f"Model file found at: {settings.MODEL_PATH}")
        logger.info(f"Model file size: {os.path.getsize(settings.MODEL_PATH) / (1024*1024):.2f} MB")
    
    try:
        logger.info("Attempting to load model...")
        load_model()
        if check_model_loaded():
            logger.info("✅ Model loaded successfully")
        else:
            logger.error("❌ Model failed to load properly")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}\n{traceback.format_exc()}")
        logger.error("Application started with model loading error!")

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "docs": "/docs",
    }

def start():
    uvicorn.run(
        "app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
    )

if __name__ == "__main__":
    start()
