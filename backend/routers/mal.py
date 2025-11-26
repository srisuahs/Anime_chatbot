import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
import urllib.parse
from backend import crud, schemas, auth, config, models
from backend.database import get_db
from backend.mal_service import MALService

router = APIRouter()

# --- MAL API Constants ---
AUTHORIZE_URL = "https://myanimelist.net/v1/oauth2/authorize"
TOKEN_URL = "https://myanimelist.net/v1/oauth2/token"

# --- Pydantic models ---
class MalTokenRequest(schemas.BaseModel):
    code: str
    code_verifier: str

class MalAuthURLRequest(schemas.BaseModel):
    code_challenge: str
    state: str

# --- 1. Generate Authorization URL ---
@router.post("/mal/auth-url")
async def get_mal_auth_url(
    request_body: MalAuthURLRequest,
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    params = {
        "response_type": "code",
        "client_id": config.MAL_CLIENT_ID,
        "state": urllib.parse.quote_plus(request_body.state),
        "code_challenge": urllib.parse.quote_plus(request_body.code_challenge),
        "code_challenge_method": "plain",
        "redirect_uri": config.MAL_REDIRECT_URI
    }
    
    auth_url = f"{AUTHORIZE_URL}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    return {"authorization_url": auth_url}

# --- 2. Handle MAL Callback ---
@router.get("/mal/callback")
async def mal_callback(code: str, state: str):
    return RedirectResponse(
        url=f"http://localhost:5173/?code={code}&state={state}",
        status_code=status.HTTP_302_FOUND
    )

# --- 3. Token Exchange ---
@router.post("/mal/token")
def exchange_mal_token(
    token_request: MalTokenRequest,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    try:
        data_payload = (
            f"client_id={urllib.parse.quote_plus(config.MAL_CLIENT_ID)}&"
            f"client_secret={urllib.parse.quote_plus(config.MAL_CLIENT_SECRET)}&"
            f"grant_type={urllib.parse.quote_plus('authorization_code')}&"
            f"code={urllib.parse.quote_plus(token_request.code)}&"
            f"code_verifier={urllib.parse.quote_plus(token_request.code_verifier)}&"
            f"redirect_uri={config.MAL_REDIRECT_URI}"
        )
        
        response = requests.post(
            TOKEN_URL,
            data=data_payload,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        print(f"DEBUG: Token Exchange Request Payload: {response.request.body}")
        print(f"DEBUG: Token Exchange Response Status: {response.status_code}")
        print(f"DEBUG: Token Exchange Response Body: {response.text}")
        
        response.raise_for_status()
        token_data = response.json()
        
        # Fetch MAL username
        mal_service_instance = MALService()
        user_profile = mal_service_instance.get_user_profile(token_data['access_token'])
        mal_username = user_profile.get('name', 'N/A')
        
        mal_token_data = schemas.MalTokenData(
            access_token=token_data['access_token'],
            refresh_token=token_data['refresh_token'],
            token_type=token_data['token_type'],
            expires_in=token_data['expires_in'],
            username=mal_username
        )
        
        crud.create_or_update_mal_tokens(
            db=db,
            user_id=current_user.id,
            mal_token_data=mal_token_data
        )
        
        print(f"Backend: User {current_user.id} successfully connected to MAL with username: {mal_username}")
        return {"message": "MyAnimeList tokens exchanged and stored successfully!", "mal_username": mal_username}
    
    except requests.exceptions.RequestException as e:
        error_details = e.response.json() if e.response else str(e)
        raise HTTPException(status_code=400, detail=f"Failed to get MAL token: {error_details}")

# --- 4. Get User's MAL Tokens ---
@router.get("/mal/tokens", response_model=schemas.MalTokenData)
def get_user_mal_tokens(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    db_mal_tokens = crud.get_mal_tokens_by_user_id(db, user_id=current_user.id)
    if not db_mal_tokens:
        raise HTTPException(status_code=404, detail="MyAnimeList tokens not found for this user.")
    
    response_data = schemas.MalTokenData(
        access_token=db_mal_tokens.access_token,
        refresh_token=db_mal_tokens.refresh_token,
        token_type=db_mal_tokens.token_type,
        expires_in=db_mal_tokens.expires_in,
        username=db_mal_tokens.mal_username
    )
    return response_data

# --- 5. Refresh MAL Access Token ---
@router.post("/mal/refresh-token", response_model=schemas.MalTokenData)
def refresh_mal_access_token(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    db_mal_tokens = crud.get_mal_tokens_by_user_id(db, user_id=current_user.id)
    if not db_mal_tokens or not db_mal_tokens.refresh_token:
        raise HTTPException(
            status_code=400,
            detail="Refresh token not found. Please re-authenticate with MyAnimeList."
        )
    
    try:
        data_payload = (
            f"client_id={urllib.parse.quote_plus(config.MAL_CLIENT_ID)}&"
            f"client_secret={urllib.parse.quote_plus(config.MAL_CLIENT_SECRET)}&"
            f"grant_type={urllib.parse.quote_plus('refresh_token')}&"
            f"refresh_token={urllib.parse.quote_plus(db_mal_tokens.refresh_token)}"
        )
        
        response = requests.post(
            TOKEN_URL,
            data=data_payload,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        response.raise_for_status()
        token_data = response.json()
        
        mal_token_data = schemas.MalTokenData(
            access_token=token_data['access_token'],
            refresh_token=token_data.get('refresh_token', db_mal_tokens.refresh_token),
            token_type=token_data['token_type'],
            expires_in=token_data['expires_in']
        )
        
        updated_tokens = crud.create_or_update_mal_tokens(
            db=db,
            user_id=current_user.id,
            mal_token_data=mal_token_data
        )
        return updated_tokens
    
    except requests.exceptions.RequestException as e:
        error_details = e.response.json() if e.response else str(e)
        raise HTTPException(status_code=400, detail=f"Failed to refresh MAL token: {error_details}")

# --- 6. Delete User's MAL Tokens ---
@router.delete("/mal/tokens")
def delete_user_mal_tokens(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    db_mal_tokens = crud.get_mal_tokens_by_user_id(db, user_id=current_user.id)
    if not db_mal_tokens:
        raise HTTPException(status_code=404, detail="MyAnimeList tokens not found for this user.")
    
    db.delete(db_mal_tokens)
    db.commit()
    return {"message": "MyAnimeList tokens deleted successfully. Account disconnected."}

# --- 7. Get User's Watching Anime List ---
@router.get("/mal/my-anime-list/watching")
def get_watching_anime_list(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    mal_tokens = crud.get_mal_tokens_by_user_id(db, user_id=current_user.id)
    if not mal_tokens:
        raise HTTPException(
            status_code=404,
            detail="MyAnimeList tokens not found for this user. Please connect your MAL account."
        )
    
    try:
        mal_service = MALService()
        watching_list_data = mal_service.get_user_anime_list(mal_tokens.access_token, status="watching")
        return watching_list_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching watching list from MAL: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch watching list from MyAnimeList: {e}")

# --- NEW: 8. Update Anime Status ---
@router.put("/mal/anime/{anime_id}/status")
def update_anime_status(
    anime_id: int,
    status: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    mal_tokens = crud.get_mal_tokens_by_user_id(db, user_id=current_user.id)
    if not mal_tokens:
        raise HTTPException(status_code=404, detail="MAL tokens not found")
    
    mal_service = MALService()
    try:
        result = mal_service.update_anime_status(mal_tokens.access_token, anime_id, status)
        return result
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
