from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.ask import router as ask_router
from app.api.documents import router as documents_router
from app.config import Settings, get_settings
from app.models.schemas import HealthResponse
from app.services.runtime import Runtime


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or get_settings()
    application = FastAPI(
        title=settings.app_name,
        description="Grounded question answering over user-uploaded documents.",
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    application.state.runtime = runtime or Runtime.create(settings)
    application.include_router(documents_router)
    application.include_router(ask_router)

    @application.get("/api/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse()

    frontend_dir = settings and (settings.upload_dir.parent.parent / "frontend")
    if frontend_dir and frontend_dir.exists():
        application.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return application


app = create_app()
