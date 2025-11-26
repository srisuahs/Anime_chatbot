import requests
from backend import config
from datetime import datetime

MAL_API_URL = "https://api.myanimelist.net/v2"

class MALService:
    def __init__(self):
        self.client_id = config.MAL_CLIENT_ID

    def search_anime(self, access_token: str, anime_title: str):
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "q": anime_title,
            "limit": 5,
            "fields": "id,title,main_picture,alternative_titles,synopsis,mean,genres"
        }
        response = requests.get(f"{MAL_API_URL}/anime", headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def update_anime_status(self, access_token: str, anime_id: int, new_status: str, start_date: str = None, finish_date: str = None):
        """Update anime status with optional start and finish dates"""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"status": new_status}
        
        # Add dates if status is watching or completed
        if new_status == "watching" and not start_date:
            data["start_date"] = datetime.now().strftime("%Y-%m-%d")
        elif new_status == "watching" and start_date:
            data["start_date"] = start_date
        
        if new_status == "completed" and not finish_date:
            data["finish_date"] = datetime.now().strftime("%Y-%m-%d")
        elif new_status == "completed" and finish_date:
            data["finish_date"] = finish_date
        
        response = requests.put(f"{MAL_API_URL}/anime/{anime_id}/my_list_status", headers=headers, data=data)
        response.raise_for_status()
        return response.json()

    def get_user_anime_list(self, access_token: str, status: str = "watching", limit: int = 100, offset: int = 0):
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "status": status,
            "limit": limit,
            "offset": offset,
            "fields": "list_status,start_date,broadcast,main_picture,alternative_titles,num_episodes,synopsis,mean"
        }
        response = requests.get(f"{MAL_API_URL}/users/@me/animelist", headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_user_profile(self, access_token: str):
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{MAL_API_URL}/users/@me", headers=headers)
        response.raise_for_status()
        return response.json()

    def get_random_highly_rated_anime(self, access_token: str, genre: str = None, limit: int = 5):
        """Get random highly rated anime from MAL ranking endpoint"""
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "ranking_type": "all",
            "limit": limit,
            "fields": "id,title,main_picture,alternative_titles,synopsis,mean,genres,num_episodes,status"
        }
        try:
            response = requests.get(f"{MAL_API_URL}/anime/ranking", headers=headers, params=params)
            response.raise_for_status()
            anime_list = response.json().get('data', [])
            
            # Filter for highly rated anime (mean >= 7.0)
            highly_rated = [anime for anime in anime_list if anime.get('node', {}).get('mean', 0) >= 7.0]
            
            # If genre is specified, filter by genre
            if genre and highly_rated:
                filtered = []
                for anime in highly_rated:
                    anime_genres = [g.get('name', '').lower() for g in anime.get('node', {}).get('genres', [])]
                    if genre.lower() in anime_genres:
                        filtered.append(anime)
                highly_rated = filtered if filtered else highly_rated
            
            return highly_rated
        except Exception as e:
            print(f"Error fetching ranked anime: {e}")
            return []

    def get_anime_details(self, access_token: str, anime_id: int):
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "fields": "id,title,status,start_date,end_date,num_episodes,broadcast,mean,related_anime,main_picture,alternative_titles"
        }
        response = requests.get(f"{MAL_API_URL}/anime/{anime_id}", headers=headers, params=params)
        response.raise_for_status()
        return response.json()
