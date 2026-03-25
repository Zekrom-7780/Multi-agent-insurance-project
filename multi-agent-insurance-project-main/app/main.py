import logging
import uuid
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.exceptions import InsuranceAgentError
from app.logging_config import conversation_id_var, setup_logging
from app.routes.chat import router as chat_router
from app.routes.events import router as events_router
from app.routes.health import router as health_router
from app.services.context import ContextManager
from app.services.database import init_db
from app.services.llm import LLMClient
from app.services.rag import RAGService
from app.services.session import SessionManager
from app.tools import default_registry
from app.agents.graph import build_graph

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info("Starting Insurance Agent API")

    # Database
    await init_db()

    # LLM client
    llm = LLMClient()
    app.state.llm = llm

    # RAG service
    rag = RAGService()
    try:
        rag.init()
    except Exception as exc:
        logger.warning("RAG service init failed (FAQ queries will be empty): %s", exc)
    app.state.rag = rag

    # Context manager
    context_mgr = ContextManager(llm=llm)
    app.state.context_mgr = context_mgr

    # Session manager
    app.state.session_mgr = SessionManager()

    # Tool registry
    app.state.registry = default_registry

    # Build LangGraph
    app.state.graph = build_graph(
        llm=llm,
        registry=default_registry,
        rag=rag,
        context_mgr=context_mgr,
    )

    logger.info("All services initialised")
    yield
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title="Insurance Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request-ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Conversation-ID", str(uuid.uuid4()))
        request.state.conversation_id = cid
        conversation_id_var.set(cid)
        response = await call_next(request)
        response.headers["X-Conversation-ID"] = cid
        return response


app.add_middleware(RequestIDMiddleware)


# Exception handlers
@app.exception_handler(InsuranceAgentError)
async def insurance_error_handler(request: Request, exc: InsuranceAgentError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "detail": exc.message},
    )


# Routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(events_router)


@app.get("/")
async def root():
    return RedirectResponse("/static/index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
