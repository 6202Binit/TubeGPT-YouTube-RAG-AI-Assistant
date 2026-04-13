"""Application entrypoint and environment bootstrap."""

import os

from dotenv import load_dotenv


def _assert_env() -> None:
    """Load the .env file and ensure required configuration is present."""

    load_dotenv()

    required_vars = [
        "MONGO_CORE_URI",
        "MONGO_DB_NAME",
        "PGUSER",
        "PGPASSWORD",
        "PGHOST",
        "PGPORT",
        "PGDATABASE",
        "PINECONE_API_KEY",
        # "GEMINI_API_KEY",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
    ]

    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")


_assert_env()

from api import app as api_app

# Import new Keycloak middleware
from middlewares.auth import AIBuddyAuthMiddleware

# Apply middleware
api_app.add_middleware(AIBuddyAuthMiddleware)

app = api_app
__all__ = ["app"]


