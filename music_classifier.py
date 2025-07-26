import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class MusicClassifier:
    """LLM-based music classification service supporting OpenAI and Anthropic."""
    
    CATEGORIES = ['Dance Pop', 'House', 'Bass']
    
    CLASSIFICATION_PROMPT = """You are an expert in electronic music categorization, helping DJs classify tracks into broad electronic genres. The available categories are:

- Dance Pop: melodic, catchy, often vocal-heavy tracks intended for mainstream dance audiences. Think Dua Lipa, Calvin Harris, or remixes of pop hits.
- House: rhythm-driven tracks with 4/4 beats, consistent grooves, minimal vocals, and strong club energy. Think deep house, tech house, or progressive house.
- Bass: includes genres like dubstep, trap, future bass, or other subgenres focused on heavy low-end, syncopated beats, or experimental production.

Categorize each song based on the metadata provided.

### Example 1
Track: "One Kiss"  
Artist: Calvin Harris, Dua Lipa  
Genres: dance pop, pop, EDM  
Tempo: 124 BPM  
Energy: 0.8  
Danceability: 0.85  
Prediction: **Dance Pop**

### Example 2
Track: "Losing It"  
Artist: Fisher  
Genres: tech house, house  
Tempo: 125 BPM  
Energy: 0.9  
Danceability: 0.82  
Prediction: **House**

### Example 3
Track: "Core"  
Artist: RL Grime  
Genres: trap, bass, electronic  
Tempo: 150 BPM  
Energy: 0.95  
Danceability: 0.6  
Prediction: **Bass**

{track_data}

Respond with ONLY the predictions in this exact format for each track:
Track X: **Category**

Do not include any other text, explanations, or formatting."""

    def __init__(self, provider: str = None, batch_size: int = None, max_retries: int = None):
        """
        Initialize the music classifier.
        
        Args:
            provider (str): LLM provider ('openai' or 'anthropic')
            batch_size (int): Number of tracks to classify per API call
            max_retries (int): Maximum retry attempts for failed requests
        """
        self.provider = provider or os.getenv('LLM_PROVIDER', 'openai')
        self.batch_size = batch_size or int(os.getenv('BATCH_SIZE', '25'))
        self.max_retries = max_retries or int(os.getenv('MAX_RETRIES', '3'))
        
        # Initialize the appropriate client
        if self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.client = openai.OpenAI(api_key=api_key)
            
        elif self.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
            self.client = anthropic.Anthropic(api_key=api_key)
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'openai' or 'anthropic'")

    def _format_track_for_classification(self, track: Dict[str, Any], index: int) -> str:
        """
        Format a single track for LLM classification.
        
        Args:
            track (Dict[str, Any]): Track data with audio features and genres
            index (int): Track index for identification
            
        Returns:
            str: Formatted track string
        """
        name = track.get('name', 'Unknown')
        artists = ', '.join(track.get('artists', ['Unknown']))
        genres = ', '.join(track.get('genres', ['Unknown']))
        
        audio_features = track.get('audio_features', {})
        tempo = audio_features.get('tempo', 'Unknown')
        energy = audio_features.get('energy', 'Unknown')
        danceability = audio_features.get('danceability', 'Unknown')
        
        # Format tempo
        tempo_str = f"{tempo:.0f} BPM" if isinstance(tempo, (int, float)) else str(tempo)
        
        # Format decimal values
        energy_str = f"{energy:.2f}" if isinstance(energy, (int, float)) else str(energy)
        danceability_str = f"{danceability:.2f}" if isinstance(danceability, (int, float)) else str(danceability)
        
        return f"""### Track {index + 1}
Track: "{name}"  
Artist: {artists}  
Genres: {genres}  
Tempo: {tempo_str}  
Energy: {energy_str}  
Danceability: {danceability_str}  
Prediction:"""

    def _format_batch_for_classification(self, tracks: List[Dict[str, Any]]) -> str:
        """
        Format a batch of tracks for LLM classification.
        
        Args:
            tracks (List[Dict[str, Any]]): List of track data
            
        Returns:
            str: Formatted prompt for the batch
        """
        track_data = '\n\n'.join([
            self._format_track_for_classification(track, i) 
            for i, track in enumerate(tracks)
        ])
        
        return self.CLASSIFICATION_PROMPT.format(track_data=track_data)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with the classification prompt."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API with the classification prompt."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _parse_classification_response(self, response: str, num_tracks: int) -> List[Optional[str]]:
        """
        Parse LLM response to extract classifications.
        
        Args:
            response (str): Raw LLM response
            num_tracks (int): Expected number of classifications
            
        Returns:
            List[Optional[str]]: List of classifications, None for failed parses
        """
        classifications = []
        
        # Look for pattern: Track X: **Category**
        pattern = r'Track\s+(\d+):\s*\*\*([^*]+)\*\*'
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        # Create a mapping from track number to classification
        track_classifications = {}
        for track_num_str, category in matches:
            try:
                track_num = int(track_num_str)
                category_clean = category.strip()
                
                # Validate category
                if category_clean in self.CATEGORIES:
                    track_classifications[track_num] = category_clean
                else:
                    # Try to find closest match
                    category_lower = category_clean.lower()
                    for valid_category in self.CATEGORIES:
                        if valid_category.lower() in category_lower or category_lower in valid_category.lower():
                            track_classifications[track_num] = valid_category
                            break
            except ValueError:
                continue
        
        # Build result list in order
        for i in range(1, num_tracks + 1):
            classifications.append(track_classifications.get(i))
            
        return classifications

    def classify_batch(self, tracks: List[Dict[str, Any]]) -> List[Optional[str]]:
        """
        Classify a batch of tracks using the configured LLM.
        
        Args:
            tracks (List[Dict[str, Any]]): List of track data with audio features
            
        Returns:
            List[Optional[str]]: Classifications for each track (None if failed)
        """
        if not tracks:
            return []
            
        prompt = self._format_batch_for_classification(tracks)
        
        for attempt in range(self.max_retries):
            try:
                if self.provider == 'openai':
                    response = self._call_openai(prompt)
                else:  # anthropic
                    response = self._call_anthropic(prompt)
                
                classifications = self._parse_classification_response(response, len(tracks))
                
                # Check if we got reasonable results
                successful_classifications = sum(1 for c in classifications if c is not None)
                success_rate = successful_classifications / len(tracks)
                
                if success_rate >= 0.7:  # Accept if at least 70% were classified
                    return classifications
                else:
                    print(f"Low success rate ({success_rate:.2%}) on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"Classification attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                    
        # If all attempts failed, return None for all tracks
        return [None] * len(tracks)

    def classify_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify all tracks with progress tracking.
        
        Args:
            tracks (List[Dict[str, Any]]): List of enriched track data
            
        Returns:
            List[Dict[str, Any]]: Tracks with added 'classification' field
        """
        if not tracks:
            return []
            
        classified_tracks = []
        total_batches = (len(tracks) + self.batch_size - 1) // self.batch_size
        
        print(f"Classifying {len(tracks)} tracks in {total_batches} batches using {self.provider}")
        
        for batch_idx in range(0, len(tracks), self.batch_size):
            batch_tracks = tracks[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_tracks)} tracks)...")
            
            classifications = self.classify_batch(batch_tracks)
            
            # Add classifications to tracks
            for track, classification in zip(batch_tracks, classifications):
                track_with_classification = track.copy()
                track_with_classification['classification'] = classification
                classified_tracks.append(track_with_classification)
            
            # Progress update
            successful = sum(1 for c in classifications if c is not None)
            print(f"Batch {batch_num} complete: {successful}/{len(classifications)} classified successfully")
            
            # Rate limiting - small delay between batches
            if batch_idx + self.batch_size < len(tracks):
                time.sleep(1)
        
        # Summary
        total_classified = sum(1 for track in classified_tracks if track.get('classification') is not None)
        success_rate = total_classified / len(classified_tracks) if classified_tracks else 0
        
        print(f"\nClassification complete: {total_classified}/{len(classified_tracks)} tracks classified ({success_rate:.1%} success rate)")
        
        return classified_tracks

    def get_classification_summary(self, classified_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of classification results.
        
        Args:
            classified_tracks (List[Dict[str, Any]]): Tracks with classifications
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        total_tracks = len(classified_tracks)
        
        if total_tracks == 0:
            return {'total_tracks': 0, 'categories': {}, 'unclassified': 0}
        
        category_counts = {category: 0 for category in self.CATEGORIES}
        unclassified = 0
        
        for track in classified_tracks:
            classification = track.get('classification')
            if classification and classification in self.CATEGORIES:
                category_counts[classification] += 1
            else:
                unclassified += 1
        
        return {
            'total_tracks': total_tracks,
            'categories': category_counts,
            'unclassified': unclassified,
            'success_rate': (total_tracks - unclassified) / total_tracks
        }