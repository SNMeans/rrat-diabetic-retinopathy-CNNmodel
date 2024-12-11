from fastapi import FastAPI
from analyze import router as analyze_router
from root import router as root_router

# Create the FastAPI instance
app = FastAPI()

# Include the routers
app.include_router(root_router, prefix="/")
app.include_router(analyze_router, prefix="/analyze")

