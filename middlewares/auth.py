import os
import time
import httpx
import requests
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import Response
from jose import jwt

KEYCLOAK_BASE = os.getenv("KEYCLOAK_BASE_URL")
REALM = os.getenv("KEYCLOAK_REALM_NAME")
CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

AUTH_BASE_URL = os.getenv("AUTH_BASE_URL")
AUTH_BASE_KEY = os.getenv("AUTH_BASE_KEY")

COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN")

ISSUER = f"{KEYCLOAK_BASE}/realms/{REALM}"
JWKS_URL = f"{ISSUER}/protocol/openid-connect/certs"
ALGORITHMS = ["RS256"]

# =====================================================
# JWKS Cache
# =====================================================

_JWKS_CACHE = None
_JWKS_CACHE_TIME = 0
_JWKS_TTL = 3600  # 1 hour


async def get_jwks():
    """
    Async + cached JWKS fetch.
    Never blocks the event loop.
    """
    global _JWKS_CACHE, _JWKS_CACHE_TIME

    now = time.time()
    if _JWKS_CACHE and (now - _JWKS_CACHE_TIME) < _JWKS_TTL:
        return _JWKS_CACHE

    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(JWKS_URL)
        resp.raise_for_status()

    _JWKS_CACHE = resp.json()
    _JWKS_CACHE_TIME = now
    return _JWKS_CACHE



async def verify_and_extract_claims(token: str) -> dict:
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        jwks = await get_jwks()
        key = next(k for k in jwks["keys"] if k["kid"] == kid)

        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=ISSUER,
            options={"verify_aud": False},
        )

        return claims

    except Exception as e:
        raise Exception("JWT verification failed") from e


# =====================================================
# Keycloak Userinfo (fast path)
# =====================================================

async def keycloak_userinfo(token: str):
    url = f"{ISSUER}/protocol/openid-connect/userinfo"
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(url, headers={"Authorization": f"Bearer {token}"})
    return r.json() if r.status_code == 200 else None


# =====================================================
# Refresh Token
# =====================================================

async def keycloak_refresh(refresh_token: str):
    url = f"{ISSUER}/protocol/openid-connect/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
    }

    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.post(url, data=data)

    return r.json() if r.status_code == 200 else None


# =====================================================
# Fetch User Profile (NON-BLOCKING)
# =====================================================

async def fetch_basic_details(identifier: str):
    # Supports BOTH email & userId
    if "@" in identifier:
        url = f"{AUTH_BASE_URL}/v1/basicDetails?email={identifier}"
    else:
        url = f"{AUTH_BASE_URL}/v1/basicDetails?userId={identifier}"

    if AUTH_BASE_KEY:
        url += f"&key={AUTH_BASE_KEY}"

    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(url)

    return r.json() if r.status_code == 200 else None


# =====================================================
# Cookie Helper
# =====================================================

def set_login_cookies(response: Response, token_data: dict):
    params = dict(
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        domain=COOKIE_DOMAIN,
    )

    response.set_cookie("ElevateToken", token_data["access_token"], max_age=3600, **params)
    response.set_cookie("RefreshToken", token_data["refresh_token"], max_age=2592000, **params)


# =====================================================
# Middleware
# =====================================================

class AIBuddyAuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        # ---------------------------------------------
        # Public endpoints
        # ---------------------------------------------
        if request.url.path in {"/", "/health", "/docs", "/openapi.json"}:
            return await call_next(request)

        access = request.cookies.get("ElevateToken")
        refresh = request.cookies.get("RefreshToken")

        token = None
        new_token_data = None

        # ---------------------------------------------
        # 1️⃣ Token source
        # ---------------------------------------------
        if access:
            token = access
        elif refresh:
            new_token_data = await keycloak_refresh(refresh)
            if not new_token_data:
                return Response("Session expired", status_code=401)
            token = new_token_data["access_token"]
        else:
            auth = request.headers.get("Authorization")
            if not auth:
                return Response("Missing token", status_code=401)
            token = auth.split()[1] if " " in auth else auth

        # ---------------------------------------------
        # 2️⃣ Validate token
        # ---------------------------------------------
        kc_info = await keycloak_userinfo(token)
        if not kc_info:
            try:
                kc_info = await verify_and_extract_claims(token)
            except Exception:
                return Response("Invalid token", status_code=401)

        # ---------------------------------------------
        # 3️⃣ Identity extraction (STANDARD)
        # ---------------------------------------------
        user_id = kc_info.get("sub")
        email = kc_info.get("email") or kc_info.get("preferred_username")

        if not user_id:
            return Response("Invalid token claims", status_code=401)

        # ---------------------------------------------
        # 4️⃣ Profile enrichment (NON-BLOCKING)
        # ---------------------------------------------
        details = await fetch_basic_details(email or user_id) or {}

        # ---------------------------------------------
        # 5️⃣ Attach request context
        # ---------------------------------------------
        request.state.token = token
        request.state.user_id = (
            details.get("_id")
            or details.get("user_id")
            or user_id
        )
        request.state.email = email
        request.state.name = details.get("name")
        request.state.phone = details.get("phone")
        request.state.kc_user = kc_info

        # ---------------------------------------------
        # 6️⃣ Continue pipeline
        # ---------------------------------------------
        response: Response = await call_next(request)

        if new_token_data:
            set_login_cookies(response, new_token_data)

        return response
