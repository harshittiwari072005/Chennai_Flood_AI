from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict, resources, zones

app = FastAPI(
    title="Chennai Flood Risk AI",
    description="AI-driven flood prediction and resource allocation",
    version="1.0.0",
)

# Allow React frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route groups
app.include_router(predict.router,   prefix="/api")
app.include_router(resources.router, prefix="/api")
app.include_router(zones.router,     prefix="/api")

# Health check endpoint
@app.get("/")
def root():
    return {
        "status"  : "running",
        "project" : "Chennai Flood Risk AI",
        "version" : "1.0.0",
        "endpoints": [
            "POST /api/predict-risk",
            "POST /api/estimate-resources",
            "POST /api/optimize-allocation",
            "GET  /api/zones",
        ]
    }