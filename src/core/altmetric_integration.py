"""
Altmetric API Integration for Social Media Tracking
Free tier implementation for academic research
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings
import requests
import time
from datetime import datetime
import re
from urllib.parse import quote

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AltmetricAPIIntegrator:
    """
    Altmetric API integration for tracking social media mentions and attention scores
    Free tier for academic research - no API key required for basic access
    """
    
    def __init__(self, api_key=None, rate_limit_delay=1.0):
        """
        Initialize Altmetric API integrator
        
        Args:
            api_key (str, optional): API key for enhanced access (free for academic research)
            rate_limit_delay (float): Delay between API calls to respect rate limits
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://api.altmetric.com/v1"
        self.session = requests.Session()
        
        # Set headers
        headers = {
            'User-Agent': 'Academic-Research-Project/1.0 (Paper-Virality-Prediction)',
            'Accept': 'application/json'
        }
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        self.session.headers.update(headers)
        
        # Statistics tracking
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.cached_results = {}
        
    def extract_doi_from_paper(self, paper):
        """Extract DOI from paper data"""
        doi = paper.get('doi', '')
        if not doi:
            return None
            
        # Clean DOI - remove URL prefix if present
        if doi.startswith('http'):
            doi = doi.split('doi.org/')[-1]
        
        # Remove any trailing slashes or spaces
        doi = doi.strip().rstrip('/')
        
        return doi if doi else None
    
    def get_altmetric_data_by_doi(self, doi, max_retries=3):
        """
        Get Altmetric data for a paper by DOI
        
        Args:
            doi (str): DOI of the paper
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            dict: Altmetric data or None if not found
        """
        if not doi:
            return None
            
        # Check cache first
        if doi in self.cached_results:
            return self.cached_results[doi]
        
        # Clean and encode DOI
        clean_doi = quote(doi, safe='')
        url = f"{self.base_url}/doi/{clean_doi}"
        
        for attempt in range(max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                response = self.session.get(url, timeout=10)
                self.api_calls_made += 1
                
                if response.status_code == 200:
                    data = response.json()
                    self.successful_calls += 1
                    self.cached_results[doi] = data
                    logger.debug(f"Successfully retrieved Altmetric data for DOI: {doi}")
                    return data
                    
                elif response.status_code == 404:
                    # Paper not found in Altmetric database
                    logger.debug(f"Paper not found in Altmetric database: {doi}")
                    self.cached_results[doi] = None
                    return None
                    
                elif response.status_code == 429:
                    # Rate limit exceeded
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60 seconds
                    logger.warning(f"Rate limit exceeded for DOI {doi}, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.warning(f"Unexpected status code {response.status_code} for DOI: {doi}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for DOI {doi}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        self.failed_calls += 1
        return None
    
    def extract_altmetric_features(self, altmetric_data):
        """
        Extract features from Altmetric API response
        
        Args:
            altmetric_data (dict): Raw Altmetric API response
            
        Returns:
            dict: Extracted features
        """
        if not altmetric_data:
            return self.get_default_altmetric_features()
        
        try:
            features = {
                # Core Altmetric metrics
                'altmetric_score': altmetric_data.get('score', 0),
                'altmetric_id': altmetric_data.get('altmetric_id', 0),
                
                # Social media mentions
                'twitter_mentions': altmetric_data.get('cited_by_tweeters_count', 0),
                'facebook_mentions': altmetric_data.get('cited_by_fbwalls_count', 0),
                'linkedin_mentions': altmetric_data.get('cited_by_linkedin_count', 0),
                'reddit_mentions': altmetric_data.get('cited_by_rdts_count', 0),
                
                # Traditional media
                'news_mentions': altmetric_data.get('cited_by_msm_count', 0),
                'blog_mentions': altmetric_data.get('cited_by_feeds_count', 0),
                
                # Academic and policy mentions
                'wikipedia_mentions': altmetric_data.get('cited_by_wikipedia_count', 0),
                'policy_mentions': altmetric_data.get('cited_by_policies_count', 0),
                'patent_mentions': altmetric_data.get('cited_by_patents_count', 0),
                
                # Video and multimedia
                'video_mentions': altmetric_data.get('cited_by_videos_count', 0),
                'youtube_mentions': altmetric_data.get('cited_by_videos_count', 0),  # Often same as video
                
                # Academic platforms
                'mendeley_readers': altmetric_data.get('readers_count', 0),
                'connotea_bookmarks': altmetric_data.get('connotea', 0),
                'citeulike_bookmarks': altmetric_data.get('citeulike', 0),
                
                # Geographic and demographic data
                'countries_mentioning': len(altmetric_data.get('demographics', {}).get('geo', {}).get('twitter', {})),
                'twitter_user_types': len(altmetric_data.get('demographics', {}).get('users', {}).get('twitter', {})),
                
                # Temporal features
                'first_mention_date': altmetric_data.get('first_seen', 0),
                'last_mention_date': altmetric_data.get('last_updated', 0),
                
                # Attention sources
                'total_sources': len([k for k, v in altmetric_data.items() 
                                    if k.startswith('cited_by_') and v > 0]),
                
                # Derived metrics
                'social_media_total': (
                    altmetric_data.get('cited_by_tweeters_count', 0) +
                    altmetric_data.get('cited_by_fbwalls_count', 0) +
                    altmetric_data.get('cited_by_linkedin_count', 0) +
                    altmetric_data.get('cited_by_rdts_count', 0)
                ),
                
                'media_coverage_total': (
                    altmetric_data.get('cited_by_msm_count', 0) +
                    altmetric_data.get('cited_by_feeds_count', 0)
                ),
                
                'academic_attention_total': (
                    altmetric_data.get('cited_by_wikipedia_count', 0) +
                    altmetric_data.get('cited_by_policies_count', 0) +
                    altmetric_data.get('readers_count', 0)
                ),
                
                # Quality indicators
                'has_altmetric_data': 1,
                'altmetric_percentile': altmetric_data.get('context', {}).get('all', {}).get('pct', 0),
                'journal_percentile': altmetric_data.get('context', {}).get('journal', {}).get('pct', 0),
                
                # Influence metrics (if available)
                'twitter_influence_score': self.calculate_twitter_influence(altmetric_data),
                'media_influence_score': self.calculate_media_influence(altmetric_data),
            }
            
            # Normalize temporal features
            features = self.normalize_temporal_features(features)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting Altmetric features: {e}")
            return self.get_default_altmetric_features()
    
    def get_default_altmetric_features(self):
        """Return default features when no Altmetric data is available"""
        return {
            'altmetric_score': 0,
            'altmetric_id': 0,
            'twitter_mentions': 0,
            'facebook_mentions': 0,
            'linkedin_mentions': 0,
            'reddit_mentions': 0,
            'news_mentions': 0,
            'blog_mentions': 0,
            'wikipedia_mentions': 0,
            'policy_mentions': 0,
            'patent_mentions': 0,
            'video_mentions': 0,
            'youtube_mentions': 0,
            'mendeley_readers': 0,
            'connotea_bookmarks': 0,
            'citeulike_bookmarks': 0,
            'countries_mentioning': 0,
            'twitter_user_types': 0,
            'first_mention_date': 0,
            'last_mention_date': 0,
            'total_sources': 0,
            'social_media_total': 0,
            'media_coverage_total': 0,
            'academic_attention_total': 0,
            'has_altmetric_data': 0,
            'altmetric_percentile': 0,
            'journal_percentile': 0,
            'twitter_influence_score': 0,
            'media_influence_score': 0,
        }
    
    def calculate_twitter_influence(self, altmetric_data):
        """Calculate Twitter influence score based on user demographics"""
        try:
            twitter_demo = altmetric_data.get('demographics', {}).get('users', {}).get('twitter', {})
            if not twitter_demo:
                return 0
            
            # Weight different user types
            influence_weights = {
                'public': 1.0,
                'practitioners': 2.0,
                'researchers': 3.0,
                'science_communicators': 2.5
            }
            
            total_influence = 0
            total_users = 0
            
            for user_type, count in twitter_demo.items():
                weight = influence_weights.get(user_type, 1.0)
                total_influence += count * weight
                total_users += count
            
            return total_influence / max(total_users, 1)
            
        except:
            return 0
    
    def calculate_media_influence(self, altmetric_data):
        """Calculate media influence score"""
        try:
            news_count = altmetric_data.get('cited_by_msm_count', 0)
            blog_count = altmetric_data.get('cited_by_feeds_count', 0)
            
            # News outlets typically have higher influence than blogs
            return news_count * 3.0 + blog_count * 1.5
            
        except:
            return 0
    
    def normalize_temporal_features(self, features):
        """Normalize temporal features to useful metrics"""
        try:
            current_timestamp = int(time.time())
            
            if features['first_mention_date'] > 0:
                # Days since first mention
                features['days_since_first_mention'] = (
                    current_timestamp - features['first_mention_date']
                ) / (24 * 60 * 60)
            else:
                features['days_since_first_mention'] = 0
            
            if features['last_mention_date'] > 0:
                # Days since last mention
                features['days_since_last_mention'] = (
                    current_timestamp - features['last_mention_date']
                ) / (24 * 60 * 60)
                
                # Mention duration (how long the paper stayed in social attention)
                if features['first_mention_date'] > 0:
                    features['mention_duration_days'] = (
                        features['last_mention_date'] - features['first_mention_date']
                    ) / (24 * 60 * 60)
                else:
                    features['mention_duration_days'] = 0
            else:
                features['days_since_last_mention'] = 0
                features['mention_duration_days'] = 0
            
            # Mention velocity (mentions per day)
            if features['mention_duration_days'] > 0:
                total_mentions = (features['social_media_total'] + 
                                features['media_coverage_total'] + 
                                features['academic_attention_total'])
                features['mention_velocity'] = total_mentions / features['mention_duration_days']
            else:
                features['mention_velocity'] = 0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error normalizing temporal features: {e}")
            return features
    
    def process_paper_batch(self, papers_df, batch_size=50):
        """
        Process a batch of papers to extract Altmetric data
        
        Args:
            papers_df (pd.DataFrame): DataFrame containing paper data
            batch_size (int): Number of papers to process in each batch
            
        Returns:
            pd.DataFrame: DataFrame with Altmetric features
        """
        logger.info(f"Processing {len(papers_df)} papers for Altmetric data...")
        
        altmetric_features = []
        total_papers = len(papers_df)
        
        for idx, (_, paper) in enumerate(papers_df.iterrows()):
            # Progress reporting
            if idx % 10 == 0:
                logger.info(f"Processing paper {idx + 1}/{total_papers} ({(idx + 1) / total_papers * 100:.1f}%)")
            
            # Extract DOI
            doi = self.extract_doi_from_paper(paper)
            
            if doi:
                # Get Altmetric data
                altmetric_data = self.get_altmetric_data_by_doi(doi)
                
                # Extract features
                features = self.extract_altmetric_features(altmetric_data)
                
                # Add paper identifier
                features['paper_index'] = idx
                features['doi'] = doi
                
            else:
                # No DOI available
                features = self.get_default_altmetric_features()
                features['paper_index'] = idx
                features['doi'] = ''
                logger.debug(f"No DOI found for paper at index {idx}")
            
            altmetric_features.append(features)
            
            # Batch progress save (optional)
            if idx > 0 and idx % batch_size == 0:
                self.save_progress(altmetric_features, idx)
        
        # Convert to DataFrame
        altmetric_df = pd.DataFrame(altmetric_features)
        
        # Generate summary statistics
        self.print_processing_summary(altmetric_df)
        
        return altmetric_df
    
    def save_progress(self, features_list, current_index):
        """Save progress periodically"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_file = f"results/altmetric_progress_{current_index}_{timestamp}.json"
            
            Path("results").mkdir(exist_ok=True)
            
            with open(progress_file, 'w') as f:
                json.dump(features_list, f, indent=2)
                
            logger.info(f"Progress saved to {progress_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def print_processing_summary(self, altmetric_df):
        """Print summary of Altmetric data processing"""
        logger.info("\n" + "="*60)
        logger.info("ALTMETRIC DATA PROCESSING SUMMARY")
        logger.info("="*60)
        
        total_papers = len(altmetric_df)
        papers_with_data = altmetric_df['has_altmetric_data'].sum()
        
        logger.info(f"Total papers processed: {total_papers}")
        logger.info(f"Papers with Altmetric data: {papers_with_data} ({papers_with_data/total_papers*100:.1f}%)")
        logger.info(f"API calls made: {self.api_calls_made}")
        logger.info(f"Successful calls: {self.successful_calls}")
        logger.info(f"Failed calls: {self.failed_calls}")
        
        if papers_with_data > 0:
            logger.info("\nAltmetric Score Statistics:")
            scores = altmetric_df[altmetric_df['has_altmetric_data'] == 1]['altmetric_score']
            logger.info(f"  Mean score: {scores.mean():.2f}")
            logger.info(f"  Median score: {scores.median():.2f}")
            logger.info(f"  Max score: {scores.max():.2f}")
            
            logger.info("\nSocial Media Mention Statistics:")
            twitter = altmetric_df[altmetric_df['has_altmetric_data'] == 1]['twitter_mentions']
            logger.info(f"  Papers with Twitter mentions: {(twitter > 0).sum()}")
            logger.info(f"  Mean Twitter mentions: {twitter.mean():.2f}")
            
            news = altmetric_df[altmetric_df['has_altmetric_data'] == 1]['news_mentions']
            logger.info(f"  Papers with news mentions: {(news > 0).sum()}")
            logger.info(f"  Mean news mentions: {news.mean():.2f}")
        
        logger.info("="*60)
    
    def save_altmetric_features(self, altmetric_df, output_file=None):
        """Save Altmetric features to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/altmetric_features_{timestamp}.json"
        
        # Create results directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        # Convert DataFrame to JSON-serializable format
        features_records = []
        for record in altmetric_df.to_dict('records'):
            features_records.append(convert_types(record))
        
        # Convert to JSON-serializable format
        altmetric_data = {
            'timestamp': datetime.now().isoformat(),
            'total_papers': int(len(altmetric_df)),
            'papers_with_data': int(altmetric_df['has_altmetric_data'].sum()),
            'api_calls_made': int(self.api_calls_made),
            'successful_calls': int(self.successful_calls),
            'failed_calls': int(self.failed_calls),
            'features': features_records
        }
        
        with open(output_file, 'w') as f:
            json.dump(altmetric_data, f, indent=2)
        
        logger.info(f"Altmetric features saved to: {output_file}")
        return output_file

def main():
    """Main execution for Altmetric API integration"""
    logger.info("Starting Altmetric API Integration...")
    
    try:
        # Initialize Altmetric API integrator
        # Note: For enhanced access, register for free academic API key at altmetric.com
        integrator = AltmetricAPIIntegrator(
            api_key=None,  # Free tier - no API key required for basic access
            rate_limit_delay=1.0  # Be respectful to the API
        )
        
        # Load our dataset
        data_file = "data/datasets/openalex_5000_papers.json"
        if not Path(data_file).exists():
            logger.error(f"Dataset not found: {data_file}")
            return
        
        logger.info(f"Loading dataset from: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', [])
        df = pd.json_normalize(papers)
        
        # Apply quality filtering (similar to our previous implementations)
        df = df.dropna(subset=['title', 'abstract'])
        df = df[df['title'].str.len() >= 5]
        df = df[df['abstract'].str.len() >= 50]
        df = df[df['year'] >= 2015]
        df = df[df['year'] <= 2023]
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Sample for testing (limit to first 100 papers for initial test)
        test_df = df.head(100).copy()
        logger.info(f"Testing with {len(test_df)} papers")
        
        # Process papers for Altmetric data
        altmetric_features_df = integrator.process_paper_batch(test_df, batch_size=20)
        
        # Save results
        output_file = integrator.save_altmetric_features(altmetric_features_df)
        
        # Print final summary
        print("\n" + "="*80)
        print("🌐 ALTMETRIC INTEGRATION COMPLETED")
        print("="*80)
        print(f"✅ Processed: {len(test_df)} papers")
        print(f"✅ Papers with Altmetric data: {altmetric_features_df['has_altmetric_data'].sum()}")
        print(f"✅ Social media features extracted: {len([c for c in altmetric_features_df.columns if c not in ['paper_index', 'doi']])}")
        print(f"✅ API calls made: {integrator.api_calls_made}")
        print(f"✅ Success rate: {integrator.successful_calls / integrator.api_calls_made * 100 if integrator.api_calls_made > 0 else 0:.1f}%")
        print(f"✅ Results saved to: {output_file}")
        
        # Show sample of extracted features
        if altmetric_features_df['has_altmetric_data'].sum() > 0:
            print("\n📊 Sample Altmetric Features (papers with data):")
            sample_data = altmetric_features_df[altmetric_features_df['has_altmetric_data'] == 1].head(3)
            key_features = ['altmetric_score', 'twitter_mentions', 'news_mentions', 'social_media_total']
            for feature in key_features:
                if feature in sample_data.columns:
                    values = sample_data[feature].values
                    print(f"  {feature}: {values}")
        
        print(f"\n🎯 Altmetric integration adds social media signals for early virality prediction!")
        print(f"   • Twitter mentions, news coverage, blog posts tracked")
        print(f"   • Academic attention via Wikipedia, policies, patents")
        print(f"   • Temporal dynamics and influence scoring")
        print(f"   • Ready for integration with existing ML models")
        
    except Exception as e:
        logger.error(f"Error in Altmetric integration: {e}")
        raise

if __name__ == "__main__":
    main()