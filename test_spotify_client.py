import pytest
from unittest.mock import Mock, patch, MagicMock
from spotify_client import SpotifyClient


class TestSpotifyClient:
    
    @patch('spotify_client.os.getenv')
    @patch('spotify_client.spotipy.Spotify')
    def test_init_success(self, mock_spotify, mock_getenv):
        """Test successful initialization of SpotifyClient."""
        mock_getenv.side_effect = lambda key, default=None: {
            'SPOTIFY_CLIENT_ID': 'test_client_id',
            'SPOTIFY_CLIENT_SECRET': 'test_client_secret',
            'SPOTIFY_REDIRECT_URI': 'http://localhost:8080/callback'
        }.get(key, default)
        
        client = SpotifyClient()
        
        assert client.client_id == 'test_client_id'
        assert client.client_secret == 'test_client_secret'
        assert client.redirect_uri == 'http://localhost:8080/callback'
        mock_spotify.assert_called_once()

    @patch('spotify_client.os.getenv')
    def test_init_missing_credentials(self, mock_getenv):
        """Test initialization fails when credentials are missing."""
        mock_getenv.side_effect = lambda key, default=None: {
            'SPOTIFY_CLIENT_ID': None,
            'SPOTIFY_CLIENT_SECRET': None
        }.get(key, default)
        
        with pytest.raises(ValueError, match="Spotify credentials not found"):
            SpotifyClient()

    @patch('spotify_client.os.getenv')
    @patch('spotify_client.spotipy.Spotify')
    def test_get_liked_songs_success(self, mock_spotify, mock_getenv):
        """Test successful retrieval of liked songs."""
        # Setup environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'SPOTIFY_CLIENT_ID': 'test_client_id',
            'SPOTIFY_CLIENT_SECRET': 'test_client_secret'
        }.get(key, default)
        
        # Mock Spotify API response
        mock_sp_instance = Mock()
        mock_spotify.return_value = mock_sp_instance
        
        # First call returns items, second call returns empty (end of pagination)
        mock_sp_instance.current_user_saved_tracks.side_effect = [
            {
                'items': [
                    {
                        'track': {
                            'id': 'track1',
                            'name': 'Test Song 1',
                            'artists': [{'name': 'Artist 1'}],
                            'album': {'name': 'Album 1'},
                            'duration_ms': 180000,
                            'popularity': 75,
                            'preview_url': 'http://preview1.mp3',
                            'external_urls': {'spotify': 'http://spotify.com/track1'}
                        },
                        'added_at': '2023-01-01T00:00:00Z'
                    }
                ]
            },
            {'items': []}  # Empty response to end pagination
        ]
        
        client = SpotifyClient()
        tracks = client.get_liked_songs()
        
        assert len(tracks) == 1
        assert tracks[0]['id'] == 'track1'
        assert tracks[0]['name'] == 'Test Song 1'
        assert tracks[0]['artists'] == ['Artist 1']
        assert tracks[0]['album'] == 'Album 1'
        assert tracks[0]['duration_ms'] == 180000
        assert tracks[0]['popularity'] == 75
        assert tracks[0]['added_at'] == '2023-01-01T00:00:00Z'

    @patch('spotify_client.os.getenv')
    @patch('spotify_client.spotipy.Spotify')
    def test_get_liked_songs_multiple_pages(self, mock_spotify, mock_getenv):
        """Test pagination handling for liked songs."""
        mock_getenv.side_effect = lambda key, default=None: {
            'SPOTIFY_CLIENT_ID': 'test_client_id',
            'SPOTIFY_CLIENT_SECRET': 'test_client_secret'
        }.get(key, default)
        
        mock_sp_instance = Mock()
        mock_spotify.return_value = mock_sp_instance
        
        # Create 75 mock tracks across two pages (50 + 25)
        page1_items = []
        for i in range(50):
            page1_items.append({
                'track': {
                    'id': f'track{i}',
                    'name': f'Song {i}',
                    'artists': [{'name': f'Artist {i}'}],
                    'album': {'name': f'Album {i}'},
                    'duration_ms': 180000,
                    'popularity': 75,
                    'preview_url': f'http://preview{i}.mp3',
                    'external_urls': {'spotify': f'http://spotify.com/track{i}'}
                },
                'added_at': '2023-01-01T00:00:00Z'
            })
        
        page2_items = []
        for i in range(50, 75):
            page2_items.append({
                'track': {
                    'id': f'track{i}',
                    'name': f'Song {i}',
                    'artists': [{'name': f'Artist {i}'}],
                    'album': {'name': f'Album {i}'},
                    'duration_ms': 180000,
                    'popularity': 75,
                    'preview_url': f'http://preview{i}.mp3',
                    'external_urls': {'spotify': f'http://spotify.com/track{i}'}
                },
                'added_at': '2023-01-01T00:00:00Z'
            })
        
        mock_sp_instance.current_user_saved_tracks.side_effect = [
            {'items': page1_items},
            {'items': page2_items},
            {'items': []}  # End pagination
        ]
        
        client = SpotifyClient()
        tracks = client.get_liked_songs()
        
        assert len(tracks) == 75
        assert tracks[0]['id'] == 'track0'
        assert tracks[49]['id'] == 'track49'
        assert tracks[74]['id'] == 'track74'

    @patch('spotify_client.os.getenv')
    @patch('spotify_client.spotipy.Spotify')
    def test_get_user_created_playlists_success(self, mock_spotify, mock_getenv):
        """Test successful retrieval of user-created playlists."""
        mock_getenv.side_effect = lambda key, default=None: {
            'SPOTIFY_CLIENT_ID': 'test_client_id',
            'SPOTIFY_CLIENT_SECRET': 'test_client_secret'
        }.get(key, default)
        
        mock_sp_instance = Mock()
        mock_spotify.return_value = mock_sp_instance
        
        # Mock current user
        mock_sp_instance.current_user.return_value = {'id': 'test_user'}
        
        # Mock playlist response
        mock_sp_instance.current_user_playlists.side_effect = [
            {
                'items': [
                    {
                        'id': 'playlist1',
                        'name': 'My Playlist',
                        'description': 'Test playlist',
                        'tracks': {'total': 25},
                        'public': True,
                        'collaborative': False,
                        'external_urls': {'spotify': 'http://spotify.com/playlist1'},
                        'owner': {'id': 'test_user'}  # User-created playlist
                    },
                    {
                        'id': 'playlist2',
                        'name': 'Someone Else Playlist',
                        'description': 'Not my playlist',
                        'tracks': {'total': 15},
                        'public': True,
                        'collaborative': False,
                        'external_urls': {'spotify': 'http://spotify.com/playlist2'},
                        'owner': {'id': 'other_user'}  # Not user-created
                    }
                ]
            },
            {'items': []}  # End pagination
        ]
        
        client = SpotifyClient()
        playlists = client.get_user_created_playlists()
        
        # Should only return user-created playlists
        assert len(playlists) == 1
        assert playlists[0]['id'] == 'playlist1'
        assert playlists[0]['name'] == 'My Playlist'
        assert playlists[0]['tracks_total'] == 25

    @patch('spotify_client.os.getenv')
    @patch('spotify_client.spotipy.Spotify')
    def test_get_playlist_tracks_success(self, mock_spotify, mock_getenv):
        """Test successful retrieval of tracks from a playlist."""
        mock_getenv.side_effect = lambda key, default=None: {
            'SPOTIFY_CLIENT_ID': 'test_client_id',
            'SPOTIFY_CLIENT_SECRET': 'test_client_secret'
        }.get(key, default)
        
        mock_sp_instance = Mock()
        mock_spotify.return_value = mock_sp_instance
        
        mock_sp_instance.playlist_tracks.side_effect = [
            {
                'items': [
                    {
                        'track': {
                            'id': 'track1',
                            'name': 'Playlist Song 1',
                            'artists': [{'name': 'Artist 1'}],
                            'album': {'name': 'Album 1'},
                            'duration_ms': 180000,
                            'popularity': 75,
                            'preview_url': 'http://preview1.mp3',
                            'external_urls': {'spotify': 'http://spotify.com/track1'}
                        },
                        'added_at': '2023-01-01T00:00:00Z'
                    }
                ]
            },
            {'items': []}  # End pagination
        ]
        
        client = SpotifyClient()
        tracks = client.get_playlist_tracks('test_playlist_id')
        
        assert len(tracks) == 1
        assert tracks[0]['id'] == 'track1'
        assert tracks[0]['name'] == 'Playlist Song 1'
        assert tracks[0]['playlist_id'] == 'test_playlist_id'

    @patch('spotify_client.os.getenv')
    @patch('spotify_client.spotipy.Spotify')
    def test_get_playlist_tracks_skip_null_tracks(self, mock_spotify, mock_getenv):
        """Test that null tracks (local files, removed tracks) are skipped."""
        mock_getenv.side_effect = lambda key, default=None: {
            'SPOTIFY_CLIENT_ID': 'test_client_id',
            'SPOTIFY_CLIENT_SECRET': 'test_client_secret'
        }.get(key, default)
        
        mock_sp_instance = Mock()
        mock_spotify.return_value = mock_sp_instance
        
        mock_sp_instance.playlist_tracks.side_effect = [
            {
                'items': [
                    {
                        'track': None,  # Null track (should be skipped)
                        'added_at': '2023-01-01T00:00:00Z'
                    },
                    {
                        'track': {
                            'id': None,  # Track with null ID (should be skipped)
                            'name': 'Local Track',
                            'artists': [{'name': 'Local Artist'}],
                            'album': {'name': 'Local Album'},
                            'duration_ms': 180000,
                            'popularity': 0,
                            'preview_url': None,
                            'external_urls': {}
                        },
                        'added_at': '2023-01-01T00:00:00Z'
                    },
                    {
                        'track': {
                            'id': 'track1',
                            'name': 'Valid Track',
                            'artists': [{'name': 'Valid Artist'}],
                            'album': {'name': 'Valid Album'},
                            'duration_ms': 180000,
                            'popularity': 75,
                            'preview_url': 'http://preview1.mp3',
                            'external_urls': {'spotify': 'http://spotify.com/track1'}
                        },
                        'added_at': '2023-01-01T00:00:00Z'
                    }
                ]
            },
            {'items': []}  # End pagination
        ]
        
        client = SpotifyClient()
        tracks = client.get_playlist_tracks('test_playlist_id')
        
        # Should only return the valid track
        assert len(tracks) == 1
        assert tracks[0]['id'] == 'track1'
        assert tracks[0]['name'] == 'Valid Track'

    @patch('spotify_client.os.getenv')
    @patch('spotify_client.spotipy.Spotify')
    def test_get_all_user_tracks_deduplication(self, mock_spotify, mock_getenv):
        """Test that duplicate tracks are removed from the combined results."""
        mock_getenv.side_effect = lambda key, default=None: {
            'SPOTIFY_CLIENT_ID': 'test_client_id',
            'SPOTIFY_CLIENT_SECRET': 'test_client_secret'
        }.get(key, default)
        
        mock_sp_instance = Mock()
        mock_spotify.return_value = mock_sp_instance
        
        # Mock current user
        mock_sp_instance.current_user.return_value = {'id': 'test_user'}
        
        # Mock liked songs
        mock_sp_instance.current_user_saved_tracks.side_effect = [
            {
                'items': [
                    {
                        'track': {
                            'id': 'track1',  # This track will appear in both liked and playlist
                            'name': 'Duplicate Song',
                            'artists': [{'name': 'Artist 1'}],
                            'album': {'name': 'Album 1'},
                            'duration_ms': 180000,
                            'popularity': 75,
                            'preview_url': 'http://preview1.mp3',
                            'external_urls': {'spotify': 'http://spotify.com/track1'}
                        },
                        'added_at': '2023-01-01T00:00:00Z'
                    }
                ]
            },
            {'items': []}
        ]
        
        # Mock playlists
        mock_sp_instance.current_user_playlists.side_effect = [
            {
                'items': [
                    {
                        'id': 'playlist1',
                        'name': 'My Playlist',
                        'description': 'Test playlist',
                        'tracks': {'total': 1},
                        'public': True,
                        'collaborative': False,
                        'external_urls': {'spotify': 'http://spotify.com/playlist1'},
                        'owner': {'id': 'test_user'}
                    }
                ]
            },
            {'items': []}
        ]
        
        # Mock playlist tracks (same track as in liked songs)
        mock_sp_instance.playlist_tracks.side_effect = [
            {
                'items': [
                    {
                        'track': {
                            'id': 'track1',  # Same track ID as in liked songs
                            'name': 'Duplicate Song',
                            'artists': [{'name': 'Artist 1'}],
                            'album': {'name': 'Album 1'},
                            'duration_ms': 180000,
                            'popularity': 75,
                            'preview_url': 'http://preview1.mp3',
                            'external_urls': {'spotify': 'http://spotify.com/track1'}
                        },
                        'added_at': '2023-01-01T00:00:00Z'
                    }
                ]
            },
            {'items': []}
        ]
        
        client = SpotifyClient()
        all_tracks = client.get_all_user_tracks()
        
        # Should only have one copy of the track despite it being in both liked and playlist
        assert len(all_tracks) == 1
        assert all_tracks[0]['id'] == 'track1'
        assert all_tracks[0]['source'] == 'liked_songs'  # Should be marked as from liked songs since it was processed first