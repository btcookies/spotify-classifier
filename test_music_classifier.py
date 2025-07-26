import pytest
from unittest.mock import Mock, patch, MagicMock
from music_classifier import MusicClassifier


class TestMusicClassifier:
    
    @patch('music_classifier.os.getenv')
    def test_init_openai_success(self, mock_getenv):
        """Test successful initialization with OpenAI."""
        mock_getenv.side_effect = lambda key, default=None: {
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test_key',
            'BATCH_SIZE': '20',
            'MAX_RETRIES': '3'
        }.get(key, default)
        
        with patch('music_classifier.openai') as mock_openai:
            mock_openai.OpenAI.return_value = Mock()
            
            classifier = MusicClassifier()
            
            assert classifier.provider == 'openai'
            assert classifier.batch_size == 20
            assert classifier.max_retries == 3
            mock_openai.OpenAI.assert_called_once_with(api_key='test_key')

    @patch('music_classifier.os.getenv')
    def test_init_anthropic_success(self, mock_getenv):
        """Test successful initialization with Anthropic."""
        mock_getenv.side_effect = lambda key, default=None: {
            'LLM_PROVIDER': 'anthropic',
            'ANTHROPIC_API_KEY': 'test_key',
            'BATCH_SIZE': '30',
            'MAX_RETRIES': '2'
        }.get(key, default)
        
        with patch('music_classifier.anthropic') as mock_anthropic:
            mock_anthropic.Anthropic.return_value = Mock()
            
            classifier = MusicClassifier()
            
            assert classifier.provider == 'anthropic'
            assert classifier.batch_size == 30
            assert classifier.max_retries == 2
            mock_anthropic.Anthropic.assert_called_once_with(api_key='test_key')

    @patch('music_classifier.os.getenv')
    def test_init_missing_api_key(self, mock_getenv):
        """Test initialization fails when API key is missing."""
        mock_getenv.side_effect = lambda key, default=None: {
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': None
        }.get(key, default)
        
        with patch('music_classifier.openai'):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
                MusicClassifier()

    def test_format_track_for_classification(self):
        """Test track formatting for classification."""
        with patch('music_classifier.os.getenv'), \
             patch('music_classifier.openai') as mock_openai:
            
            mock_openai.OpenAI.return_value = Mock()
            classifier = MusicClassifier(provider='openai')
            
            track = {
                'name': 'Test Song',
                'artists': ['Artist 1', 'Artist 2'],
                'genres': ['electronic', 'dance'],
                'audio_features': {
                    'tempo': 128.5,
                    'energy': 0.85,
                    'danceability': 0.92
                }
            }
            
            formatted = classifier._format_track_for_classification(track, 0)
            
            assert 'Track 1' in formatted
            assert 'Test Song' in formatted
            assert 'Artist 1, Artist 2' in formatted
            assert 'electronic, dance' in formatted
            assert '129 BPM' in formatted
            assert '0.85' in formatted
            assert '0.92' in formatted

    def test_format_track_missing_data(self):
        """Test track formatting with missing data."""
        with patch('music_classifier.os.getenv'), \
             patch('music_classifier.openai') as mock_openai:
            
            mock_openai.OpenAI.return_value = Mock()
            classifier = MusicClassifier(provider='openai')
            
            track = {
                'name': 'Test Song',
                'artists': [],
                'genres': [],
                'audio_features': {}
            }
            
            formatted = classifier._format_track_for_classification(track, 0)
            
            assert 'Unknown' in formatted  # Should handle missing artists
            assert 'Unknown' in formatted  # Should handle missing genres

    def test_parse_classification_response_success(self):
        """Test successful parsing of classification response."""
        with patch('music_classifier.os.getenv'), \
             patch('music_classifier.openai') as mock_openai:
            
            mock_openai.OpenAI.return_value = Mock()
            classifier = MusicClassifier(provider='openai')
            
            response = """Track 1: **Dance Pop**
Track 2: **House**
Track 3: **Bass**"""
            
            classifications = classifier._parse_classification_response(response, 3)
            
            assert classifications == ['Dance Pop', 'House', 'Bass']

    def test_parse_classification_response_partial(self):
        """Test parsing with some failed classifications."""
        with patch('music_classifier.os.getenv'), \
             patch('music_classifier.openai') as mock_openai:
            
            mock_openai.OpenAI.return_value = Mock()
            classifier = MusicClassifier(provider='openai')
            
            response = """Track 1: **Dance Pop**
Track 2: **Invalid Category**
Track 3: **House**"""
            
            classifications = classifier._parse_classification_response(response, 3)
            
            assert classifications == ['Dance Pop', None, 'House']

    def test_parse_classification_response_fuzzy_matching(self):
        """Test fuzzy matching for category names."""
        with patch('music_classifier.os.getenv'), \
             patch('music_classifier.openai') as mock_openai:
            
            mock_openai.OpenAI.return_value = Mock()
            classifier = MusicClassifier(provider='openai')
            
            response = """Track 1: **dance pop**
Track 2: **HOUSE**
Track 3: **bass music**"""
            
            classifications = classifier._parse_classification_response(response, 3)
            
            assert classifications == ['Dance Pop', 'House', 'Bass']

    @patch('music_classifier.os.getenv')
    def test_classify_batch_openai_success(self, mock_getenv):
        """Test successful batch classification with OpenAI."""
        mock_getenv.side_effect = lambda key, default=None: {
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test_key'
        }.get(key, default)
        
        with patch('music_classifier.openai') as mock_openai:
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Track 1: **Dance Pop**\nTrack 2: **House**"
            mock_client.chat.completions.create.return_value = mock_response
            
            classifier = MusicClassifier()
            
            tracks = [
                {
                    'name': 'Song 1',
                    'artists': ['Artist 1'],
                    'genres': ['pop'],
                    'audio_features': {'tempo': 120, 'energy': 0.8, 'danceability': 0.9}
                },
                {
                    'name': 'Song 2',
                    'artists': ['Artist 2'],
                    'genres': ['house'],
                    'audio_features': {'tempo': 125, 'energy': 0.9, 'danceability': 0.8}
                }
            ]
            
            classifications = classifier.classify_batch(tracks)
            
            assert classifications == ['Dance Pop', 'House']
            mock_client.chat.completions.create.assert_called_once()

    @patch('music_classifier.os.getenv')
    def test_classify_batch_anthropic_success(self, mock_getenv):
        """Test successful batch classification with Anthropic."""
        mock_getenv.side_effect = lambda key, default=None: {
            'LLM_PROVIDER': 'anthropic',
            'ANTHROPIC_API_KEY': 'test_key'
        }.get(key, default)
        
        with patch('music_classifier.anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.Anthropic.return_value = mock_client
            
            # Mock Anthropic response
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = "Track 1: **Bass**"
            mock_client.messages.create.return_value = mock_response
            
            classifier = MusicClassifier()
            
            tracks = [
                {
                    'name': 'Bass Song',
                    'artists': ['Bass Artist'],
                    'genres': ['dubstep'],
                    'audio_features': {'tempo': 140, 'energy': 0.95, 'danceability': 0.7}
                }
            ]
            
            classifications = classifier.classify_batch(tracks)
            
            assert classifications == ['Bass']
            mock_client.messages.create.assert_called_once()

    @patch('music_classifier.os.getenv')
    @patch('music_classifier.time.sleep')  # Mock sleep to speed up test
    def test_classify_batch_retry_logic(self, mock_sleep, mock_getenv):
        """Test retry logic when classification fails."""
        mock_getenv.side_effect = lambda key, default=None: {
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test_key',
            'MAX_RETRIES': '2'
        }.get(key, default)
        
        with patch('music_classifier.openai') as mock_openai:
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            
            # First call fails, second succeeds
            mock_client.chat.completions.create.side_effect = [
                Exception("API Error"),
                Mock(choices=[Mock(message=Mock(content="Track 1: **Dance Pop**"))])
            ]
            
            classifier = MusicClassifier()
            
            tracks = [
                {
                    'name': 'Song 1',
                    'artists': ['Artist 1'],
                    'genres': ['pop'],
                    'audio_features': {'tempo': 120, 'energy': 0.8, 'danceability': 0.9}
                }
            ]
            
            classifications = classifier.classify_batch(tracks)
            
            assert classifications == ['Dance Pop']
            assert mock_client.chat.completions.create.call_count == 2
            mock_sleep.assert_called_once()  # Should have slept between retries

    def test_get_classification_summary(self):
        """Test generation of classification summary."""
        with patch('music_classifier.os.getenv'), \
             patch('music_classifier.openai') as mock_openai:
            
            mock_openai.OpenAI.return_value = Mock()
            classifier = MusicClassifier(provider='openai')
            
            classified_tracks = [
                {'classification': 'Dance Pop'},
                {'classification': 'Dance Pop'},
                {'classification': 'House'},
                {'classification': 'Bass'},
                {'classification': None}  # Unclassified
            ]
            
            summary = classifier.get_classification_summary(classified_tracks)
            
            assert summary['total_tracks'] == 5
            assert summary['categories']['Dance Pop'] == 2
            assert summary['categories']['House'] == 1
            assert summary['categories']['Bass'] == 1
            assert summary['unclassified'] == 1
            assert summary['success_rate'] == 0.8

    def test_get_classification_summary_empty(self):
        """Test summary generation with empty track list."""
        with patch('music_classifier.os.getenv'), \
             patch('music_classifier.openai') as mock_openai:
            
            mock_openai.OpenAI.return_value = Mock()
            classifier = MusicClassifier(provider='openai')
            
            summary = classifier.get_classification_summary([])
            
            assert summary['total_tracks'] == 0
            assert summary['unclassified'] == 0