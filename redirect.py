from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

app = FastAPI()

TARGET_BASE_URL = "http://prof-navigator.site"


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def redirect_all(request: Request, path: str):
    # сохраняем query параметры
    query = request.url.query

    # формируем новый URL
    target_url = f"{TARGET_BASE_URL}/{path}"
    if query:
        target_url += f"?{query}"

    return RedirectResponse(url=target_url, status_code=307)