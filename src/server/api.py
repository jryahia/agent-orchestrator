"""FastAPI application for the agent orchestrator API."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.server.routes import router

app = FastAPI(
    title="Agent Orchestrator API",
    description="Multi-agent workflow orchestration system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "agent-orchestrator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server.api:app", host="0.0.0.0", port=8000, reload=True)
