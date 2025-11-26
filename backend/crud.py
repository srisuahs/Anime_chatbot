from sqlalchemy.orm import Session
from backend import models, schemas, auth

# --- User GET operations ---
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

# --- User CREATE operation ---
def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = auth.get_password_hash(user.password) if user.password else None
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- User AUTHENTICATION operation ---
def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email=email)
    if not user or not user.hashed_password:
        return None
    if not auth.verify_password(password, user.hashed_password):
        return None
    return user

# --- MAL Token CRUD operations ---
def get_mal_tokens_by_user_id(db: Session, user_id: int):
    return db.query(models.MalTokens).filter(models.MalTokens.user_id == user_id).first()

def get_mal_tokens(db: Session, user_id: int):
    return db.query(models.MalTokens).filter(models.MalTokens.user_id == user_id).first()

def create_or_update_mal_tokens(
    db: Session,
    user_id: int,
    mal_token_data: schemas.MalTokenData
):
    db_mal_tokens = get_mal_tokens_by_user_id(db, user_id=user_id)
    if db_mal_tokens:
        db_mal_tokens.access_token = mal_token_data.access_token
        db_mal_tokens.refresh_token = mal_token_data.refresh_token
        db_mal_tokens.token_type = mal_token_data.token_type
        db_mal_tokens.expires_in = mal_token_data.expires_in
        db_mal_tokens.mal_username = mal_token_data.username
    else:
        db_mal_tokens = models.MalTokens(
            user_id=user_id,
            access_token=mal_token_data.access_token,
            refresh_token=mal_token_data.refresh_token,
            token_type=mal_token_data.token_type,
            expires_in=mal_token_data.expires_in,
            mal_username=mal_token_data.username
        )
        db.add(db_mal_tokens)
    db.commit()
    db.refresh(db_mal_tokens)
    return db_mal_tokens

# --- Conversation History CRUD operations ---
def create_conversation_message(
    db: Session,
    user_id: int,
    message: schemas.ConversationHistoryBase
):
    db_message = models.ConversationHistory(
        user_id=user_id,
        speaker=message.speaker,
        message_text=message.message_text
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_conversation_history(db: Session, user_id: int, limit: int = 100):
    return db.query(models.ConversationHistory).filter(models.ConversationHistory.user_id == user_id).order_by(models.ConversationHistory.id.asc()).limit(limit).all()
