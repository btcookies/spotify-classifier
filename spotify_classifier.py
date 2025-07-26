#!/usr/bin/env python3
"""
Spotify Music Classifier

Automatically categorizes Spotify songs into Dance Pop, House, or Bass genres
using LLM-based classification with rich audio feature analysis.
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from spotify_client import SpotifyClient
from music_classifier import MusicClassifier

load_dotenv()


class SpotifyClassificationWorkflow:
    """Main workflow orchestrator for Spotify music classification."""
    
    def __init__(self, llm_provider: str = None, batch_size: int = None):
        """
        Initialize the classification workflow.
        
        Args:
            llm_provider (str): LLM provider ('openai' or 'anthropic')
            batch_size (int): Batch size for classification
        """
        self.spotify_client = SpotifyClient()
        self.classifier = MusicClassifier(provider=llm_provider, batch_size=batch_size)
        
    def fetch_and_enrich_tracks(self) -> List[Dict[str, Any]]:
        """
        Fetch all user tracks and enrich with audio features and genres.
        
        Returns:
            List[Dict[str, Any]]: Enriched track data
        """
        print("Fetching user's liked songs and playlists...")
        tracks = self.spotify_client.get_all_user_tracks()
        
        if not tracks:
            print("No tracks found!")
            return []
            
        print(f"Found {len(tracks)} unique tracks")
        
        print("Enriching tracks with audio features and genres...")
        enriched_tracks = self.spotify_client.enrich_tracks_with_features(tracks)
        
        print(f"Successfully enriched {len(enriched_tracks)} tracks")
        return enriched_tracks
    
    def classify_all_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify all tracks using the LLM classifier.
        
        Args:
            tracks (List[Dict[str, Any]]): Enriched track data
            
        Returns:
            List[Dict[str, Any]]: Classified track data
        """
        if not tracks:
            return []
            
        print(f"\nStarting classification of {len(tracks)} tracks...")
        classified_tracks = self.classifier.classify_tracks(tracks)
        
        # Print summary
        summary = self.classifier.get_classification_summary(classified_tracks)
        print(f"\n📊 Classification Summary:")
        print(f"Total tracks: {summary['total_tracks']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Unclassified: {summary['unclassified']}")
        print("\nCategory breakdown:")
        for category, count in summary['categories'].items():
            percentage = (count / summary['total_tracks']) * 100 if summary['total_tracks'] > 0 else 0
            print(f"  {category}: {count} tracks ({percentage:.1f}%)")
            
        return classified_tracks
    
    def save_results(self, classified_tracks: List[Dict[str, Any]], output_file: str = None) -> str:
        """
        Save classification results to a JSON file.
        
        Args:
            classified_tracks (List[Dict[str, Any]]): Classified track data
            output_file (str): Output file path (optional)
            
        Returns:
            str: Path to the saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"spotify_classifications_{timestamp}.json"
        
        # Prepare output data
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_tracks': len(classified_tracks),
                'llm_provider': self.classifier.provider,
                'batch_size': self.classifier.batch_size
            },
            'summary': self.classifier.get_classification_summary(classified_tracks),
            'tracks': classified_tracks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        return output_file
    
    def create_categorized_playlists(self, classified_tracks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize tracks by classification category.
        
        Args:
            classified_tracks (List[Dict[str, Any]]): Classified track data
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Tracks organized by category
        """
        categorized = {category: [] for category in MusicClassifier.CATEGORIES}
        categorized['Unclassified'] = []
        
        for track in classified_tracks:
            classification = track.get('classification')
            if classification and classification in MusicClassifier.CATEGORIES:
                categorized[classification].append(track)
            else:
                categorized['Unclassified'].append(track)
        
        return categorized
    
    def export_playlist_files(self, categorized_tracks: Dict[str, List[Dict[str, Any]]], output_dir: str = "playlists") -> List[str]:
        """
        Export categorized tracks to separate playlist files.
        
        Args:
            categorized_tracks (Dict[str, List[Dict[str, Any]]]): Tracks by category
            output_dir (str): Output directory for playlist files
            
        Returns:
            List[str]: List of created file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        for category, tracks in categorized_tracks.items():
            if not tracks:
                continue
                
            filename = f"{category.lower().replace(' ', '_')}_playlist.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {category} Playlist\n")
                f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# {len(tracks)} tracks\n\n")
                
                for track in tracks:
                    artists = ', '.join(track.get('artists', ['Unknown']))
                    f.write(f"{track.get('name', 'Unknown')} - {artists}\n")
                    
                    # Add Spotify URL if available
                    spotify_url = track.get('external_urls', {}).get('spotify')
                    if spotify_url:
                        f.write(f"  🎵 {spotify_url}\n")
                    f.write("\n")
            
            created_files.append(filepath)
            print(f"📝 Created {category} playlist: {filepath} ({len(tracks)} tracks)")
        
        return created_files
    
    def run_full_workflow(self, output_file: str = None, export_playlists: bool = True) -> Dict[str, Any]:
        """
        Run the complete classification workflow.
        
        Args:
            output_file (str): Output file for results (optional)
            export_playlists (bool): Whether to export playlist files
            
        Returns:
            Dict[str, Any]: Workflow results and file paths
        """
        print("🎵 Starting Spotify Music Classification Workflow\n")
        
        # Step 1: Fetch and enrich tracks
        tracks = self.fetch_and_enrich_tracks()
        if not tracks:
            return {'error': 'No tracks found'}
        
        # Step 2: Classify tracks
        classified_tracks = self.classify_all_tracks(tracks)
        
        # Step 3: Save results
        results_file = self.save_results(classified_tracks, output_file)
        
        # Step 4: Export playlists (optional)
        playlist_files = []
        if export_playlists:
            print("\n📂 Creating categorized playlist files...")
            categorized = self.create_categorized_playlists(classified_tracks)
            playlist_files = self.export_playlist_files(categorized)
        
        print("\n✅ Workflow completed successfully!")
        
        return {
            'tracks': classified_tracks,
            'results_file': results_file,
            'playlist_files': playlist_files,
            'summary': self.classifier.get_classification_summary(classified_tracks)
        }


def main():
    """Command-line interface for the Spotify classifier."""
    parser = argparse.ArgumentParser(description='Classify Spotify songs into Dance Pop, House, or Bass genres')
    
    parser.add_argument('--provider', choices=['openai', 'anthropic'], 
                       help='LLM provider to use (default: from env or openai)')
    parser.add_argument('--batch-size', type=int, 
                       help='Number of tracks to classify per API call (default: from env or 25)')
    parser.add_argument('--output', '-o', 
                       help='Output file for classification results (default: auto-generated)')
    parser.add_argument('--no-playlists', action='store_true',
                       help='Skip creating playlist files')
    parser.add_argument('--tracks-only', action='store_true',
                       help='Only fetch tracks without classification (for testing)')
    
    args = parser.parse_args()
    
    try:
        workflow = SpotifyClassificationWorkflow(
            llm_provider=args.provider,
            batch_size=args.batch_size
        )
        
        if args.tracks_only:
            # Just fetch and show track info
            tracks = workflow.fetch_and_enrich_tracks()
            print(f"\nFound {len(tracks)} tracks")
            
            if tracks:
                print(f"\nSample track info:")
                sample = tracks[0]
                print(f"Name: {sample.get('name')}")
                print(f"Artists: {', '.join(sample.get('artists', []))}")
                print(f"Genres: {', '.join(sample.get('genres', []))}")
                audio_features = sample.get('audio_features', {})
                print(f"Tempo: {audio_features.get('tempo', 'N/A')}")
                print(f"Energy: {audio_features.get('energy', 'N/A')}")
                print(f"Danceability: {audio_features.get('danceability', 'N/A')}")
        else:
            # Run full workflow
            results = workflow.run_full_workflow(
                output_file=args.output,
                export_playlists=not args.no_playlists
            )
            
            if 'error' in results:
                print(f"❌ Error: {results['error']}")
                return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Classification interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())