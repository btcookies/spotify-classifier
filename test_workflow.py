import pytest
import json
import os
from unittest.mock import Mock, patch, mock_open
from spotify_classifier import SpotifyClassificationWorkflow


class TestSpotifyClassificationWorkflow:
    
    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    def test_init(self, mock_spotify_client, mock_classifier):
        """Test workflow initialization."""
        workflow = SpotifyClassificationWorkflow(llm_provider='openai', batch_size=30)
        
        mock_spotify_client.assert_called_once()
        mock_classifier.assert_called_once_with(provider='openai', batch_size=30)

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    def test_fetch_and_enrich_tracks(self, mock_spotify_client, mock_classifier):
        """Test track fetching and enrichment."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_spotify_client.return_value = mock_client_instance
        
        mock_tracks = [
            {'id': 'track1', 'name': 'Song 1'},
            {'id': 'track2', 'name': 'Song 2'}
        ]
        
        enriched_tracks = [
            {
                'id': 'track1',
                'name': 'Song 1',
                'audio_features': {'tempo': 120},
                'genres': ['pop']
            },
            {
                'id': 'track2',
                'name': 'Song 2',
                'audio_features': {'tempo': 125},
                'genres': ['house']
            }
        ]
        
        mock_client_instance.get_all_user_tracks.return_value = mock_tracks
        mock_client_instance.enrich_tracks_with_features.return_value = enriched_tracks
        
        workflow = SpotifyClassificationWorkflow()
        result = workflow.fetch_and_enrich_tracks()
        
        assert result == enriched_tracks
        mock_client_instance.get_all_user_tracks.assert_called_once()
        mock_client_instance.enrich_tracks_with_features.assert_called_once_with(mock_tracks)

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    def test_fetch_and_enrich_tracks_empty(self, mock_spotify_client, mock_classifier):
        """Test track fetching when no tracks found."""
        mock_client_instance = Mock()
        mock_spotify_client.return_value = mock_client_instance
        mock_client_instance.get_all_user_tracks.return_value = []
        
        workflow = SpotifyClassificationWorkflow()
        result = workflow.fetch_and_enrich_tracks()
        
        assert result == []
        mock_client_instance.enrich_tracks_with_features.assert_not_called()

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    def test_classify_all_tracks(self, mock_spotify_client, mock_classifier):
        """Test track classification."""
        mock_classifier_instance = Mock()
        mock_classifier.return_value = mock_classifier_instance
        
        tracks = [
            {'id': 'track1', 'name': 'Song 1'},
            {'id': 'track2', 'name': 'Song 2'}
        ]
        
        classified_tracks = [
            {'id': 'track1', 'name': 'Song 1', 'classification': 'Dance Pop'},
            {'id': 'track2', 'name': 'Song 2', 'classification': 'House'}
        ]
        
        summary = {
            'total_tracks': 2,
            'categories': {'Dance Pop': 1, 'House': 1, 'Bass': 0},
            'unclassified': 0,
            'success_rate': 1.0
        }
        
        mock_classifier_instance.classify_tracks.return_value = classified_tracks
        mock_classifier_instance.get_classification_summary.return_value = summary
        
        workflow = SpotifyClassificationWorkflow()
        result = workflow.classify_all_tracks(tracks)
        
        assert result == classified_tracks
        mock_classifier_instance.classify_tracks.assert_called_once_with(tracks)

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    def test_classify_all_tracks_empty(self, mock_spotify_client, mock_classifier):
        """Test classification with empty track list."""
        workflow = SpotifyClassificationWorkflow()
        result = workflow.classify_all_tracks([])
        
        assert result == []

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    @patch('spotify_classifier.datetime')
    def test_save_results(self, mock_datetime, mock_spotify_client, mock_classifier):
        """Test saving classification results."""
        # Mock datetime
        mock_datetime.now.return_value.isoformat.return_value = '2023-01-01T12:00:00'
        mock_datetime.now.return_value.strftime.return_value = '20230101_120000'
        
        # Mock classifier
        mock_classifier_instance = Mock()
        mock_classifier.return_value = mock_classifier_instance
        mock_classifier_instance.provider = 'openai'
        mock_classifier_instance.batch_size = 25
        
        summary = {
            'total_tracks': 2,
            'categories': {'Dance Pop': 1, 'House': 1, 'Bass': 0},
            'unclassified': 0,
            'success_rate': 1.0
        }
        mock_classifier_instance.get_classification_summary.return_value = summary
        
        classified_tracks = [
            {'id': 'track1', 'name': 'Song 1', 'classification': 'Dance Pop'},
            {'id': 'track2', 'name': 'Song 2', 'classification': 'House'}
        ]
        
        with patch('builtins.open', mock_open()) as mock_file:
            workflow = SpotifyClassificationWorkflow()
            result = workflow.save_results(classified_tracks)
            
            assert result == 'spotify_classifications_20230101_120000.json'
            mock_file.assert_called_once()

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    def test_create_categorized_playlists(self, mock_spotify_client, mock_classifier):
        """Test creation of categorized playlists."""
        classified_tracks = [
            {'id': 'track1', 'name': 'Song 1', 'classification': 'Dance Pop'},
            {'id': 'track2', 'name': 'Song 2', 'classification': 'House'},
            {'id': 'track3', 'name': 'Song 3', 'classification': 'Dance Pop'},
            {'id': 'track4', 'name': 'Song 4', 'classification': None}  # Unclassified
        ]
        
        workflow = SpotifyClassificationWorkflow()
        result = workflow.create_categorized_playlists(classified_tracks)
        
        assert len(result['Dance Pop']) == 2
        assert len(result['House']) == 1
        assert len(result['Bass']) == 0
        assert len(result['Unclassified']) == 1

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    @patch('spotify_classifier.os.makedirs')
    @patch('spotify_classifier.datetime')
    def test_export_playlist_files(self, mock_datetime, mock_makedirs, mock_spotify_client, mock_classifier):
        """Test exporting playlist files."""
        mock_datetime.now.return_value.strftime.return_value = '2023-01-01 12:00:00'
        
        categorized_tracks = {
            'Dance Pop': [
                {
                    'name': 'Pop Song',
                    'artists': ['Pop Artist'],
                    'external_urls': {'spotify': 'https://open.spotify.com/track/123'}
                }
            ],
            'House': [
                {
                    'name': 'House Song',
                    'artists': ['House Artist'],
                    'external_urls': {'spotify': 'https://open.spotify.com/track/456'}
                }
            ],
            'Bass': [],  # Empty category should be skipped
            'Unclassified': []
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            workflow = SpotifyClassificationWorkflow()
            result = workflow.export_playlist_files(categorized_tracks)
            
            # Should create files for non-empty categories only
            assert len(result) == 2
            assert 'playlists/dance_pop_playlist.txt' in result
            assert 'playlists/house_playlist.txt' in result
            
            # Check that directories are created
            mock_makedirs.assert_called_once_with('playlists', exist_ok=True)
            
            # Check that files are opened for writing
            assert mock_file.call_count == 2

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    def test_run_full_workflow_no_tracks(self, mock_spotify_client, mock_classifier):
        """Test full workflow when no tracks are found."""
        mock_client_instance = Mock()
        mock_spotify_client.return_value = mock_client_instance
        mock_client_instance.get_all_user_tracks.return_value = []
        
        workflow = SpotifyClassificationWorkflow()
        result = workflow.run_full_workflow()
        
        assert 'error' in result
        assert result['error'] == 'No tracks found'

    @patch('spotify_classifier.MusicClassifier')
    @patch('spotify_classifier.SpotifyClient')
    @patch('spotify_classifier.datetime')
    @patch('spotify_classifier.os.makedirs')
    def test_run_full_workflow_success(self, mock_makedirs, mock_datetime, mock_spotify_client, mock_classifier):
        """Test successful full workflow execution."""
        # Setup datetime mock
        mock_datetime.now.return_value.isoformat.return_value = '2023-01-01T12:00:00'
        mock_datetime.now.return_value.strftime.return_value = '20230101_120000'
        
        # Setup Spotify client mock
        mock_client_instance = Mock()
        mock_spotify_client.return_value = mock_client_instance
        
        tracks = [{'id': 'track1', 'name': 'Song 1'}]
        enriched_tracks = [
            {
                'id': 'track1',
                'name': 'Song 1',
                'audio_features': {'tempo': 120},
                'genres': ['pop']
            }
        ]
        
        mock_client_instance.get_all_user_tracks.return_value = tracks
        mock_client_instance.enrich_tracks_with_features.return_value = enriched_tracks
        
        # Setup classifier mock
        mock_classifier_instance = Mock()
        mock_classifier.return_value = mock_classifier_instance
        mock_classifier_instance.provider = 'openai'
        mock_classifier_instance.batch_size = 25
        
        classified_tracks = [
            {
                'id': 'track1',
                'name': 'Song 1',
                'classification': 'Dance Pop',
                'audio_features': {'tempo': 120},
                'genres': ['pop']
            }
        ]
        
        summary = {
            'total_tracks': 1,
            'categories': {'Dance Pop': 1, 'House': 0, 'Bass': 0},
            'unclassified': 0,
            'success_rate': 1.0
        }
        
        mock_classifier_instance.classify_tracks.return_value = classified_tracks
        mock_classifier_instance.get_classification_summary.return_value = summary
        
        with patch('builtins.open', mock_open()) as mock_file:
            workflow = SpotifyClassificationWorkflow()
            result = workflow.run_full_workflow()
            
            assert 'error' not in result
            assert 'tracks' in result
            assert 'results_file' in result
            assert 'playlist_files' in result
            assert 'summary' in result
            
            assert len(result['tracks']) == 1
            assert result['tracks'][0]['classification'] == 'Dance Pop'