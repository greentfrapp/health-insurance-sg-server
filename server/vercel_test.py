import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
origins = [
    "http://localhost:5173",
    "https://health-insurance-sg.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def get_status():
    return {"message": "Success"}


if __name__ == "__main__":
    uvicorn.run(app)
