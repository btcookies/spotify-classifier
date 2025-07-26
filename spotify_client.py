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

    def get_audio_features(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch audio features for multiple tracks.
        
        Args:
            track_ids (List[str]): List of Spotify track IDs
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of track_id to audio features
        """
        features_map = {}
        
        # Spotify API allows up to 100 track IDs per request
        batch_size = 100
        
        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i:i + batch_size]
            
            try:
                features_response = self.sp.audio_features(batch_ids)
                
                for features in features_response:
                    if features is not None:  # Some tracks may not have audio features
                        features_map[features['id']] = {
                            'tempo': features.get('tempo'),
                            'energy': features.get('energy'),
                            'danceability': features.get('danceability'),
                            'valence': features.get('valence'),
                            'acousticness': features.get('acousticness'),
                            'instrumentalness': features.get('instrumentalness'),
                            'loudness': features.get('loudness'),
                            'speechiness': features.get('speechiness'),
                            'mode': features.get('mode'),
                            'key': features.get('key'),
                            'time_signature': features.get('time_signature')
                        }
            except Exception as e:
                print(f"Error fetching audio features for batch: {e}")
                continue
                
        return features_map

    def get_track_genres(self, track_ids: List[str]) -> Dict[str, List[str]]:
        """
        Get genres for tracks by looking up their artists.
        
        Args:
            track_ids (List[str]): List of Spotify track IDs
            
        Returns:
            Dict[str, List[str]]: Mapping of track_id to list of genres
        """
        track_genres = {}
        
        # Get tracks to extract artist IDs
        batch_size = 50
        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i:i + batch_size]
            
            try:
                tracks_response = self.sp.tracks(batch_ids)
                
                # Collect all unique artist IDs
                artist_ids = set()
                track_to_artists = {}
                
                for track in tracks_response['tracks']:
                    if track is not None:
                        track_id = track['id']
                        track_artist_ids = [artist['id'] for artist in track['artists']]
                        track_to_artists[track_id] = track_artist_ids
                        artist_ids.update(track_artist_ids)
                
                # Get artist information in batches
                artist_genres = {}
                artist_ids_list = list(artist_ids)
                
                for j in range(0, len(artist_ids_list), 50):  # Artists API also has 50 limit
                    artist_batch = artist_ids_list[j:j + 50]
                    
                    try:
                        artists_response = self.sp.artists(artist_batch)
                        
                        for artist in artists_response['artists']:
                            if artist is not None:
                                artist_genres[artist['id']] = artist.get('genres', [])
                    except Exception as e:
                        print(f"Error fetching artist data: {e}")
                        continue
                
                # Map track IDs to genres
                for track_id, artist_id_list in track_to_artists.items():
                    genres = []
                    for artist_id in artist_id_list:
                        if artist_id in artist_genres:
                            genres.extend(artist_genres[artist_id])
                    
                    # Remove duplicates while preserving order
                    unique_genres = []
                    seen = set()
                    for genre in genres:
                        if genre not in seen:
                            unique_genres.append(genre)
                            seen.add(genre)
                    
                    track_genres[track_id] = unique_genres
                    
            except Exception as e:
                print(f"Error fetching tracks for genre lookup: {e}")
                continue
                
        return track_genres

    def enrich_tracks_with_features(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich track data with audio features and genres.
        
        Args:
            tracks (List[Dict[str, Any]]): List of track dictionaries
            
        Returns:
            List[Dict[str, Any]]: Enriched track data with audio features and genres
        """
        if not tracks:
            return tracks
            
        track_ids = [track['id'] for track in tracks]
        
        # Get audio features and genres
        audio_features = self.get_audio_features(track_ids)
        track_genres = self.get_track_genres(track_ids)
        
        # Enrich each track
        enriched_tracks = []
        for track in tracks:
            track_id = track['id']
            enriched_track = track.copy()
            
            # Add audio features
            if track_id in audio_features:
                enriched_track['audio_features'] = audio_features[track_id]
            else:
                enriched_track['audio_features'] = {}
            
            # Add genres
            enriched_track['genres'] = track_genres.get(track_id, [])
            
            enriched_tracks.append(enriched_track)
            
        return enriched_tracks