from fastapi import FastAPI
import uvicorn
from routes import router

app = FastAPI(
    title="G19 Studio Chat API",
    description="API for interacting with G19 Studio's RAG-based chatbot",
    version="1.0.0"
)

# Include the router
app.include_router(router, prefix="/api", tags=["chat"])

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)