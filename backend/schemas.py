from pydantic import BaseModel, EmailStr
from typing import Optional

# --- Base User Schemas ---
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: Optional[str] = None

class User(UserBase):
    id: int

    class Config:
        from_attributes = True

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# --- MAL Token Schemas ---
class MalTokenData(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    username: str | None = None

# --- Conversation History Schemas ---
class ConversationHistoryBase(BaseModel):
    speaker: str
    message_text: str

class ConversationHistory(ConversationHistoryBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True

