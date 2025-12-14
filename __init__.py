from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the predict router
from app.routers.predict import router as predict_router


# create the FastAPI app
app = FastAPI(
    title="Sickle Cell Anemia Detection API",
    version="1.0.0",
)

# CORS so React (localhost:5173) can call the API
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

# simple health-check endpoint (optional, but nice)
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# include the /predict router
app.include_router(predict_router)
