import uvicorn


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("server.main:app", host="127.0.0.1", port=8000, reload=True)
