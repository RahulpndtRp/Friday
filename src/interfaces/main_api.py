from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pathlib import Path

# Import routers
from interfaces.api import chat_router
from interfaces.websocket import chat_websocket
from core.telemetry.logger import StructuredLogger

logger = StructuredLogger("api.main")

# Global application state
app_state = {"chat_managers": {}, "services_initialized": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("🚀 Starting FRIDAY API Server...")
    app_state["services_initialized"] = True
    logger.info("✅ FRIDAY API Server ready")

    yield

    # Shutdown
    logger.info("🔄 Shutting down FRIDAY API Server...")
    app_state["services_initialized"] = False
    logger.info("✅ FRIDAY API Server shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="FRIDAY Personal Assistant API",
    description="Complete API for FRIDAY Personal AI Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
frontend_path = Path(__file__).parent.parent.parent / "frontend"
app.mount(
    "/static", StaticFiles(directory=str(frontend_path / "static")), name="static"
)
templates = Jinja2Templates(directory=str(frontend_path / "templates"))

# Include API routers
app.include_router(chat_router.router, prefix="/api")
app.include_router(chat_websocket.router)


# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "FRIDAY Assistant",
            "api_base_url": str(request.base_url),
        },
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat interface page."""
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "title": "Chat - FRIDAY Assistant",
            "api_base_url": str(request.base_url),
            "user_id": "friday_user_001",
        },
    )


# API routes (keep existing health endpoint structure)
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if app_state["services_initialized"] else "starting",
        "service": "FRIDAY API",
        "version": "1.0.0",
        "active_chat_managers": len(app_state["chat_managers"]),
        "components": {
            "api": "healthy",
            "websocket": "healthy",
            "chat_service": "healthy" if app_state["chat_managers"] else "no_users",
        },
    }


# Function to register chat managers (called by main app)
def register_chat_manager(user_id: str, chat_manager):
    """Register chat manager for a user."""
    app_state["chat_managers"][user_id] = chat_manager

    # Register with routers
    chat_router.set_chat_manager(user_id, chat_manager)
    chat_websocket.set_chat_manager(user_id, chat_manager)

    logger.info(f"Chat manager registered for user", user_id=user_id)
