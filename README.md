# 🎵 Spotify Music Classifier

Automatically categorize all your Spotify songs (liked tracks + your created playlists) into **Dance Pop**, **House**, and **Bass** genres using AI-powered classification with rich audio feature analysis.

## Features

- 🎯 **Intelligent Classification**: Uses LLM analysis with Spotify audio features (tempo, energy, danceability, genres)
- 🔄 **Dual LLM Support**: Works with OpenAI GPT-4 or Anthropic Claude
- ⚡ **Batch Processing**: Efficient classification of large music libraries
- 📊 **Rich Analytics**: Detailed summaries and classification statistics
- 📁 **Export Options**: JSON results + categorized playlist files
- 🛡️ **Robust**: Error handling, retries, and graceful degradation
- ✅ **Well Tested**: Comprehensive unit test coverage

## Quick Start

### 1. Prerequisites

- Python 3.8+
- Spotify Developer Account
- OpenAI API key OR Anthropic API key

### 2. Installation

```bash
git clone <your-repo-url>
cd spotify-classifier
pip install -r requirements.txt
```

### 3. Setup Spotify API

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Note your `Client ID` and `Client Secret`
4. Add `http://localhost:8080/callback` to your app's redirect URIs

### 4. Setup LLM API

**Option A: OpenAI**
1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)

**Option B: Anthropic**
1. Get API key from [Anthropic Console](https://console.anthropic.com/)

### 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Spotify API (required)
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8080/callback

# LLM API (choose one)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Classification settings (optional)
LLM_PROVIDER=openai  # or 'anthropic'
BATCH_SIZE=25
MAX_RETRIES=3
```

### 6. Run Classification

```bash
# Basic classification (uses your .env settings)
python spotify_classifier.py

# Specify LLM provider
python spotify_classifier.py --provider anthropic

# Custom batch size for faster/slower processing
python spotify_classifier.py --batch-size 15

# Test your setup (fetch tracks without classification)
python spotify_classifier.py --tracks-only
```

## Usage Examples

### Basic Classification
```bash
python spotify_classifier.py
```
This will:
1. Fetch all your liked songs and tracks from your created playlists
2. Enrich them with Spotify audio features and genre data
3. Classify them using your configured LLM
4. Save results to `spotify_classifications_YYYYMMDD_HHMMSS.json`
5. Create categorized playlist files in `playlists/` directory

### Advanced Options
```bash
# Use specific LLM provider
python spotify_classifier.py --provider openai
python spotify_classifier.py --provider anthropic

# Custom batch size (affects speed vs API costs)
python spotify_classifier.py --batch-size 20

# Custom output file
python spotify_classifier.py --output my_results.json

# Skip creating playlist files
python spotify_classifier.py --no-playlists

# Test track fetching only (no classification)
python spotify_classifier.py --tracks-only
```

## Output Files

### Classification Results (`spotify_classifications_*.json`)
```json
{
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "total_tracks": 1250,
    "llm_provider": "openai",
    "batch_size": 25
  },
  "summary": {
    "total_tracks": 1250,
    "categories": {
      "Dance Pop": 450,
      "House": 650,
      "Bass": 150
    },
    "unclassified": 0,
    "success_rate": 1.0
  },
  "tracks": [...]
}
```

### Playlist Files (`playlists/`)
- `dance_pop_playlist.txt` - All Dance Pop tracks
- `house_playlist.txt` - All House tracks  
- `bass_playlist.txt` - All Bass tracks
- `unclassified_playlist.txt` - Any unclassified tracks

Each playlist file includes track names, artists, and Spotify URLs.

## Classification Categories

- **Dance Pop**: Melodic, catchy, often vocal-heavy tracks for mainstream dance audiences (Dua Lipa, Calvin Harris)
- **House**: Rhythm-driven 4/4 beats, consistent grooves, minimal vocals, strong club energy (deep house, tech house)
- **Bass**: Heavy low-end focused genres like dubstep, trap, future bass with syncopated beats

## Development

### Running Tests
```bash
pytest
```

### Project Structure
```
spotify-classifier/
├── spotify_client.py          # Spotify API integration
├── music_classifier.py        # LLM classification service  
├── spotify_classifier.py      # Main workflow orchestrator
├── test_*.py                  # Unit tests
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## Troubleshooting

### Common Issues

**"Spotify credentials not found"**
- Check your `.env` file has valid `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`
- Ensure the `.env` file is in the project root directory

**"OPENAI_API_KEY environment variable not set"**
- Add your OpenAI API key to `.env` or use `--provider anthropic`

**"No tracks found"**
- Make sure you have liked songs or created playlists in Spotify
- Check that your Spotify app has the correct scopes enabled

**Classification failures**
- Verify your LLM API key is valid and has sufficient credits
- Try reducing `--batch-size` for more reliable processing
- Check your internet connection

### Performance Tips

- **Faster processing**: Increase `--batch-size` (costs more API tokens)
- **More reliable**: Decrease `--batch-size` (slower but more stable)
- **Cost optimization**: Use `anthropic` provider (often cheaper than OpenAI)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.