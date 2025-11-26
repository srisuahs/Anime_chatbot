from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend import models
from backend.database import engine
from backend.routers import users, chat, mal

# --- Database Initialization ---
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# --- Middleware ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(users.router)
app.include_router(chat.router)
app.include_router(mal.router)

@app.get("/api")
def read_root():
    return {"message": "Hello from the Anime Chatbot Backend!"}
