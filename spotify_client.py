import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()


class SpotifyClient:
    def __init__(self):
        """Initialize Spotify client with OAuth authentication."""
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8080/callback')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
        
        scope = "user-library-read playlist-read-private playlist-read-collaborative"
        
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=scope
        ))

    def get_liked_songs(self) -> List[Dict[str, Any]]:
        """
        Fetch all of the user's liked songs from Spotify.
        
        Returns:
            List[Dict[str, Any]]: List of track dictionaries with metadata
        """
        tracks = []
        offset = 0
        limit = 50
        
        while True:
            results = self.sp.current_user_saved_tracks(limit=limit, offset=offset)
            
            if not results['items']:
                break
                
            for item in results['items']:
                track = item['track']
                track_data = {
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'preview_url': track['preview_url'],
                    'external_urls': track['external_urls'],
                    'added_at': item['added_at']
                }
                tracks.append(track_data)
            
            offset += limit
            
            if len(results['items']) < limit:
                break
                
        return tracks

    def get_user_created_playlists(self) -> List[Dict[str, Any]]:
        """
        Fetch all playlists created by the current user.
        
        Returns:
            List[Dict[str, Any]]: List of playlist dictionaries
        """
        user_id = self.sp.current_user()['id']
        playlists = []
        offset = 0
        limit = 50
        
        while True:
            results = self.sp.current_user_playlists(limit=limit, offset=offset)
            
            if not results['items']:
                break
                
            for playlist in results['items']:
                # Only include playlists created by the current user
                if playlist['owner']['id'] == user_id:
                    playlist_data = {
                        'id': playlist['id'],
                        'name': playlist['name'],
                        'description': playlist['description'],
                        'tracks_total': playlist['tracks']['total'],
                        'public': playlist['public'],
                        'collaborative': playlist['collaborative'],
                        'external_urls': playlist['external_urls']
                    }
                    playlists.append(playlist_data)
            
            offset += limit
            
            if len(results['items']) < limit:
                break
                
        return playlists

    def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all tracks from a specific playlist.
        
        Args:
            playlist_id (str): The Spotify playlist ID
            
        Returns:
            List[Dict[str, Any]]: List of track dictionaries with metadata
        """
        tracks = []
        offset = 0
        limit = 100
        
        while True:
            results = self.sp.playlist_tracks(playlist_id, offset=offset, limit=limit)
            
            if not results['items']:
                break
                
            for item in results['items']:
                track = item['track']
                
                # Skip if track is None (e.g., local files or removed tracks)
                if track is None or track['id'] is None:
                    continue
                    
                track_data = {
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'preview_url': track['preview_url'],
                    'external_urls': track['external_urls'],
                    'added_at': item['added_at'],
                    'playlist_id': playlist_id
                }
                tracks.append(track_data)
            
            offset += limit
            
            if len(results['items']) < limit:
                break
                
        return tracks

    def get_all_user_tracks(self) -> List[Dict[str, Any]]:
        """
        Fetch all tracks from user's liked songs and created playlists.
        
        Returns:
            List[Dict[str, Any]]: List of all unique track dictionaries
        """
        all_tracks = []
        track_ids = set()
        
        # Get liked songs
        liked_songs = self.get_liked_songs()
        for track in liked_songs:
            if track['id'] not in track_ids:
                track['source'] = 'liked_songs'
                all_tracks.append(track)
                track_ids.add(track['id'])
        
        # Get tracks from user's created playlists
        playlists = self.get_user_created_playlists()
        for playlist in playlists:
            playlist_tracks = self.get_playlist_tracks(playlist['id'])
            for track in playlist_tracks:
                if track['id'] not in track_ids:
                    track['source'] = f"playlist_{playlist['name']}"
                    all_tracks.append(track)
                    track_ids.add(track['id'])
        
        return all_tracks