from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# TODO: Import your route modules here
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI Research Assistant",
    version="1.0.0",
    description="Multi-agent AI assistant for internal knowledge retrieval"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agentic AI Research Assistant", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# TODO: Implement your /query or /chat endpoint here
# This should accept a question and run your multi-agent workflow

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
