"""
Improved Altmetric API Integration for Social Media Tracking
Enhanced identifier extraction, caching, and feature coverage
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
import hashlib
import pickle

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AltmetricAPIIntegrator:
    """
    Altmetric API integration for tracking social media mentions and attention scores
    Free tier for academic research - no API key required for basic access
    """
    
    def __init__(self, api_key=None, rate_limit_delay=1.0, cache_file="altmetric_cache.pkl"):
        """
        Initialize Altmetric API integrator with enhanced caching
        
        Args:
            api_key (str, optional): API key for enhanced access (free for academic research)
            rate_limit_delay (float): Delay between API calls to respect rate limits
            cache_file (str): File to store persistent cache
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://api.altmetric.com/v1"
        self.session = requests.Session()
        self.cache_file = cache_file
        
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
        
        # Load persistent cache
        self.load_cache()
        
        # Enhanced fallback strategies
        self.fallback_methods = [
            'doi', 'arxiv_id', 'pmid', 'title_search'
        ]
        
    def load_cache(self):
        """Load persistent cache from file"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'rb') as f:
                    self.cached_results = pickle.load(f)
                logger.info(f"Loaded {len(self.cached_results)} cached results")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self.cached_results = {}
    
    def save_cache(self):
        """Save cache to persistent storage"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cached_results, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def extract_identifiers_from_paper(self, paper):
        """Extract multiple identifiers from paper data with improved methods"""
        identifiers = {}
        
        # DOI extraction
        doi = paper.get('doi', '')
        if doi:
            # Clean DOI - remove URL prefix if present
            if doi.startswith('http'):
                doi = doi.split('doi.org/')[-1]
            doi = doi.strip().rstrip('/')
            if doi:
                identifiers['doi'] = doi
        
        # ArXiv ID extraction
        arxiv_id = paper.get('arxiv_id', '')
        if not arxiv_id:
            # Try to extract from title or abstract
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            arxiv_match = re.search(r'arxiv:?(\d{4}\.\d{4,5})', text.lower())
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
        
        if arxiv_id:
            identifiers['arxiv'] = arxiv_id
        
        # PubMed ID extraction
        pmid = paper.get('pmid', '') or paper.get('pubmed_id', '')
        if pmid:
            identifiers['pmid'] = str(pmid)
        
        # Title for fallback search
        title = paper.get('title', '').strip()
        if title and len(title) > 10:  # Minimum meaningful title length
            identifiers['title'] = title
        
        return identifiers
    
    def get_altmetric_data_by_identifier(self, identifier_type, identifier_value, max_retries=3):
        """
        Get Altmetric data for a paper by any supported identifier
        
        Args:
            identifier_type (str): Type of identifier ('doi', 'arxiv', 'pmid')
            identifier_value (str): Value of the identifier
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            dict: Altmetric data or None if not found
        """
        if not identifier_value:
            return None
        
        cache_key = f"{identifier_type}:{identifier_value}"
        
        # Check cache first
        if cache_key in self.cached_results:
            return self.cached_results[cache_key]
        
        # Build URL based on identifier type
        if identifier_type == 'doi':
            clean_id = quote(identifier_value, safe='')
            url = f"{self.base_url}/doi/{clean_id}"
        elif identifier_type == 'arxiv':
            url = f"{self.base_url}/arxiv/{identifier_value}"
        elif identifier_type == 'pmid':
            url = f"{self.base_url}/pmid/{identifier_value}"
        else:
            logger.warning(f"Unsupported identifier type: {identifier_type}")
            return None
        
        for attempt in range(max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                response = self.session.get(url, timeout=15)
                self.api_calls_made += 1
                
                if response.status_code == 200:
                    data = response.json()
                    self.successful_calls += 1
                    self.cached_results[cache_key] = data
                    logger.debug(f"Successfully retrieved Altmetric data for {identifier_type}: {identifier_value}")
                    return data
                    
                elif response.status_code == 404:
                    # Paper not found in Altmetric database
                    logger.debug(f"Paper not found in Altmetric database: {identifier_type}:{identifier_value}")
                    self.cached_results[cache_key] = None
                    return None
                    
                elif response.status_code == 429:
                    # Rate limit exceeded
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60 seconds
                    logger.warning(f"Rate limit exceeded for {identifier_type} {identifier_value}, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.warning(f"Unexpected status code {response.status_code} for {identifier_type}: {identifier_value}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {identifier_type} {identifier_value}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        self.failed_calls += 1
        return None
    
    def get_altmetric_data_with_fallback(self, paper):
        """
        Try multiple methods to get Altmetric data for a paper
        
        Args:
            paper (dict): Paper data with multiple possible identifiers
            
        Returns:
            dict: Altmetric data or None if not found with any method
        """
        identifiers = self.extract_identifiers_from_paper(paper)
        
        # Try each identifier type in order of preference
        for id_type in self.fallback_methods:
            if id_type == 'title_search':
                continue  # Skip title search for now (requires different API)
            
            if id_type == 'arxiv_id':
                id_type = 'arxiv'  # API uses 'arxiv' not 'arxiv_id'
            
            if id_type in identifiers:
                data = self.get_altmetric_data_by_identifier(id_type, identifiers[id_type])
                if data:
                    logger.debug(f"Found Altmetric data using {id_type}: {identifiers[id_type]}")
                    return data
        
        logger.debug(f"No Altmetric data found for paper with identifiers: {list(identifiers.keys())}")
        return None
    
    def extract_altmetric_features(self, altmetric_data):
        """
        Extract enhanced features from Altmetric API response
        
        Args:
            altmetric_data (dict): Raw Altmetric API response
            
        Returns:
            dict: Extracted features with enhanced social metrics
        """
        if not altmetric_data:
            return self.get_default_altmetric_features()
        
        try:
            # Basic metrics
            features = {
                # Core Altmetric metrics
                'altmetric_score': altmetric_data.get('score', 0),
                'altmetric_id': altmetric_data.get('altmetric_id', 0),
                
                # Social media mentions (enhanced)
                'twitter_mentions': altmetric_data.get('cited_by_tweeters_count', 0),
                'facebook_mentions': altmetric_data.get('cited_by_fbwalls_count', 0),
                'linkedin_mentions': altmetric_data.get('cited_by_linkedin_count', 0),
                'reddit_mentions': altmetric_data.get('cited_by_rdts_count', 0),
                'googleplus_mentions': altmetric_data.get('cited_by_gplus_count', 0),
                'pinterest_mentions': altmetric_data.get('cited_by_pinners_count', 0),
                
                # Traditional and new media
                'news_mentions': altmetric_data.get('cited_by_msm_count', 0),
                'blog_mentions': altmetric_data.get('cited_by_feeds_count', 0),
                'forum_mentions': altmetric_data.get('cited_by_forum_count', 0),
                'qa_mentions': altmetric_data.get('cited_by_qna_count', 0),
                
                # Academic and policy mentions
                'wikipedia_mentions': altmetric_data.get('cited_by_wikipedia_count', 0),
                'policy_mentions': altmetric_data.get('cited_by_policies_count', 0),
                'patent_mentions': altmetric_data.get('cited_by_patents_count', 0),
                'peer_review_mentions': altmetric_data.get('cited_by_peer_review_sites_count', 0),
                
                # Video and multimedia
                'video_mentions': altmetric_data.get('cited_by_videos_count', 0),
                'youtube_mentions': altmetric_data.get('cited_by_videos_count', 0),
                
                # Academic platforms and bookmarking
                'mendeley_readers': altmetric_data.get('readers_count', 0),
                'connotea_bookmarks': altmetric_data.get('connotea', 0),
                'citeulike_bookmarks': altmetric_data.get('citeulike', 0),
                'delicious_bookmarks': altmetric_data.get('cited_by_delicious_count', 0),
                
                # Geographic and demographic data (enhanced)
                'countries_mentioning': len(altmetric_data.get('demographics', {}).get('geo', {}).get('twitter', {})),
                'twitter_user_types': len(altmetric_data.get('demographics', {}).get('users', {}).get('twitter', {})),
                'geographic_diversity': self.calculate_geographic_diversity(altmetric_data),
                'demographic_diversity': self.calculate_demographic_diversity(altmetric_data),
                
                # Temporal features
                'first_mention_date': altmetric_data.get('first_seen', 0),
                'last_mention_date': altmetric_data.get('last_updated', 0),
                
                # Enhanced source analysis
                'total_sources': len([k for k, v in altmetric_data.items() 
                                    if k.startswith('cited_by_') and v > 0]),
                'source_diversity_score': self.calculate_source_diversity(altmetric_data),
                
                # Enhanced aggregated metrics
                'social_media_total': (
                    altmetric_data.get('cited_by_tweeters_count', 0) +
                    altmetric_data.get('cited_by_fbwalls_count', 0) +
                    altmetric_data.get('cited_by_linkedin_count', 0) +
                    altmetric_data.get('cited_by_rdts_count', 0) +
                    altmetric_data.get('cited_by_gplus_count', 0) +
                    altmetric_data.get('cited_by_pinners_count', 0)
                ),
                
                'media_coverage_total': (
                    altmetric_data.get('cited_by_msm_count', 0) +
                    altmetric_data.get('cited_by_feeds_count', 0)
                ),
                
                'academic_attention_total': (
                    altmetric_data.get('cited_by_wikipedia_count', 0) +
                    altmetric_data.get('cited_by_policies_count', 0) +
                    altmetric_data.get('readers_count', 0) +
                    altmetric_data.get('cited_by_peer_review_sites_count', 0)
                ),
                
                'public_engagement_total': (
                    altmetric_data.get('cited_by_forum_count', 0) +
                    altmetric_data.get('cited_by_qna_count', 0) +
                    altmetric_data.get('cited_by_videos_count', 0)
                ),
                
                # Quality and impact indicators
                'has_altmetric_data': 1,
                'altmetric_percentile': altmetric_data.get('context', {}).get('all', {}).get('pct', 0),
                'journal_percentile': altmetric_data.get('context', {}).get('journal', {}).get('pct', 0),
                'similar_age_percentile': altmetric_data.get('context', {}).get('similar_age_3m', {}).get('pct', 0),
                
                # Enhanced influence metrics
                'twitter_influence_score': self.calculate_twitter_influence(altmetric_data),
                'media_influence_score': self.calculate_media_influence(altmetric_data),
                'academic_influence_score': self.calculate_academic_influence(altmetric_data),
                'overall_influence_score': self.calculate_overall_influence(altmetric_data),
                
                # Virality indicators
                'viral_potential_score': self.calculate_viral_potential(altmetric_data),
                'cross_platform_engagement': self.calculate_cross_platform_engagement(altmetric_data),
            }
            
            # Normalize temporal features
            features = self.normalize_temporal_features(features)
            
            # Add ratio-based features
            features.update(self.calculate_ratio_features(features))
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting Altmetric features: {e}")
            return self.get_default_altmetric_features()
    
    def calculate_geographic_diversity(self, altmetric_data):
        """Calculate geographic diversity of mentions"""
        try:
            geo_data = altmetric_data.get('demographics', {}).get('geo', {}).get('twitter', {})
            if not geo_data:
                return 0
            
            # Shannon diversity index for geographic distribution
            total_mentions = sum(geo_data.values())
            if total_mentions == 0:
                return 0
            
            diversity = 0
            for count in geo_data.values():
                if count > 0:
                    p = count / total_mentions
                    diversity -= p * np.log(p)
            
            return diversity
        except:
            return 0
    
    def calculate_demographic_diversity(self, altmetric_data):
        """Calculate demographic diversity of Twitter users"""
        try:
            demo_data = altmetric_data.get('demographics', {}).get('users', {}).get('twitter', {})
            if not demo_data:
                return 0
            
            # Count different user types
            user_types = len([k for k, v in demo_data.items() if v > 0])
            return user_types
        except:
            return 0
    
    def calculate_source_diversity(self, altmetric_data):
        """Calculate diversity across different attention sources"""
        try:
            source_counts = []
            sources = [
                'cited_by_tweeters_count', 'cited_by_fbwalls_count', 'cited_by_linkedin_count',
                'cited_by_rdts_count', 'cited_by_msm_count', 'cited_by_feeds_count',
                'cited_by_wikipedia_count', 'cited_by_policies_count', 'cited_by_videos_count'
            ]
            
            for source in sources:
                count = altmetric_data.get(source, 0)
                if count > 0:
                    source_counts.append(count)
            
            if not source_counts:
                return 0
            
            # Coefficient of variation as diversity measure
            mean_count = np.mean(source_counts)
            std_count = np.std(source_counts)
            
            return std_count / mean_count if mean_count > 0 else 0
        except:
            return 0
    
    def calculate_academic_influence(self, altmetric_data):
        """Calculate academic influence score"""
        try:
            academic_score = (
                altmetric_data.get('cited_by_wikipedia_count', 0) * 3.0 +
                altmetric_data.get('cited_by_policies_count', 0) * 5.0 +
                altmetric_data.get('cited_by_peer_review_sites_count', 0) * 4.0 +
                altmetric_data.get('readers_count', 0) * 0.1
            )
            return academic_score
        except:
            return 0
    
    def calculate_overall_influence(self, altmetric_data):
        """Calculate overall influence score combining all metrics"""
        try:
            twitter_score = self.calculate_twitter_influence(altmetric_data)
            media_score = self.calculate_media_influence(altmetric_data)
            academic_score = self.calculate_academic_influence(altmetric_data)
            
            # Weighted combination
            overall_score = (
                twitter_score * 0.4 +
                media_score * 0.3 +
                academic_score * 0.3
            )
            return overall_score
        except:
            return 0
    
    def calculate_viral_potential(self, altmetric_data):
        """Calculate viral potential based on social media engagement patterns"""
        try:
            # High viral potential = rapid spread across multiple platforms
            social_platforms = [
                altmetric_data.get('cited_by_tweeters_count', 0),
                altmetric_data.get('cited_by_fbwalls_count', 0),
                altmetric_data.get('cited_by_linkedin_count', 0),
                altmetric_data.get('cited_by_rdts_count', 0)
            ]
            
            active_platforms = len([p for p in social_platforms if p > 0])
            total_social = sum(social_platforms)
            
            # Viral potential based on platform diversity and volume
            if active_platforms == 0:
                return 0
            
            diversity_factor = active_platforms / 4  # Normalize by max platforms
            volume_factor = min(total_social / 100, 1.0)  # Cap at reasonable level
            
            return diversity_factor * volume_factor * 100
        except:
            return 0
    
    def calculate_cross_platform_engagement(self, altmetric_data):
        """Calculate engagement across different platform types"""
        try:
            platform_types = {
                'social': (
                    altmetric_data.get('cited_by_tweeters_count', 0) +
                    altmetric_data.get('cited_by_fbwalls_count', 0) +
                    altmetric_data.get('cited_by_linkedin_count', 0) +
                    altmetric_data.get('cited_by_rdts_count', 0)
                ),
                'media': (
                    altmetric_data.get('cited_by_msm_count', 0) +
                    altmetric_data.get('cited_by_feeds_count', 0)
                ),
                'academic': (
                    altmetric_data.get('cited_by_wikipedia_count', 0) +
                    altmetric_data.get('cited_by_policies_count', 0) +
                    altmetric_data.get('readers_count', 0)
                ),
                'multimedia': altmetric_data.get('cited_by_videos_count', 0)
            }
            
            active_types = len([v for v in platform_types.values() if v > 0])
            return active_types
        except:
            return 0
    
    def calculate_ratio_features(self, features):
        """Calculate ratio-based features for enhanced insights"""
        try:
            ratios = {}
            
            # Social media ratios
            total_social = features.get('social_media_total', 0)
            if total_social > 0:
                ratios['twitter_dominance'] = features.get('twitter_mentions', 0) / total_social
                ratios['facebook_share'] = features.get('facebook_mentions', 0) / total_social
                ratios['professional_share'] = features.get('linkedin_mentions', 0) / total_social
            else:
                ratios.update({
                    'twitter_dominance': 0,
                    'facebook_share': 0,
                    'professional_share': 0
                })
            
            # Academic vs public engagement ratio
            total_academic = features.get('academic_attention_total', 0)
            total_public = features.get('social_media_total', 0) + features.get('public_engagement_total', 0)
            
            if total_public > 0:
                ratios['academic_to_public_ratio'] = total_academic / total_public
            else:
                ratios['academic_to_public_ratio'] = 0
            
            # Media coverage efficiency
            total_attention = (total_social + total_academic + 
                             features.get('media_coverage_total', 0))
            if total_attention > 0:
                ratios['media_coverage_efficiency'] = (
                    features.get('media_coverage_total', 0) / total_attention
                )
            else:
                ratios['media_coverage_efficiency'] = 0
            
            return ratios
        except:
            return {}
    
    def get_default_altmetric_features(self):
        """Return default features when no Altmetric data is available (enhanced)"""
        return {
            # Core metrics
            'altmetric_score': 0,
            'altmetric_id': 0,
            
            # Social media mentions (enhanced)
            'twitter_mentions': 0,
            'facebook_mentions': 0,
            'linkedin_mentions': 0,
            'reddit_mentions': 0,
            'googleplus_mentions': 0,
            'pinterest_mentions': 0,
            
            # Media mentions
            'news_mentions': 0,
            'blog_mentions': 0,
            'forum_mentions': 0,
            'qa_mentions': 0,
            
            # Academic mentions
            'wikipedia_mentions': 0,
            'policy_mentions': 0,
            'patent_mentions': 0,
            'peer_review_mentions': 0,
            
            # Multimedia
            'video_mentions': 0,
            'youtube_mentions': 0,
            
            # Academic platforms
            'mendeley_readers': 0,
            'connotea_bookmarks': 0,
            'citeulike_bookmarks': 0,
            'delicious_bookmarks': 0,
            
            # Demographics and geography
            'countries_mentioning': 0,
            'twitter_user_types': 0,
            'geographic_diversity': 0,
            'demographic_diversity': 0,
            
            # Temporal features
            'first_mention_date': 0,
            'last_mention_date': 0,
            'days_since_first_mention': 0,
            'days_since_last_mention': 0,
            'mention_duration_days': 0,
            'mention_velocity': 0,
            
            # Source analysis
            'total_sources': 0,
            'source_diversity_score': 0,
            
            # Aggregated metrics
            'social_media_total': 0,
            'media_coverage_total': 0,
            'academic_attention_total': 0,
            'public_engagement_total': 0,
            
            # Quality indicators
            'has_altmetric_data': 0,
            'altmetric_percentile': 0,
            'journal_percentile': 0,
            'similar_age_percentile': 0,
            
            # Influence metrics
            'twitter_influence_score': 0,
            'media_influence_score': 0,
            'academic_influence_score': 0,
            'overall_influence_score': 0,
            
            # Virality indicators
            'viral_potential_score': 0,
            'cross_platform_engagement': 0,
            
            # Ratio features
            'twitter_dominance': 0,
            'facebook_share': 0,
            'professional_share': 0,
            'academic_to_public_ratio': 0,
            'media_coverage_efficiency': 0,
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
        Process a batch of papers to extract Altmetric data with enhanced fallback
        
        Args:
            papers_df (pd.DataFrame): DataFrame containing paper data
            batch_size (int): Number of papers to process in each batch
            
        Returns:
            pd.DataFrame: DataFrame with Altmetric features
        """
        logger.info(f"Processing {len(papers_df)} papers for enhanced Altmetric data...")
        
        altmetric_features = []
        total_papers = len(papers_df)
        successful_retrievals = 0
        
        for idx, (_, paper) in enumerate(papers_df.iterrows()):
            # Progress reporting
            if idx % 10 == 0:
                logger.info(f"Processing paper {idx + 1}/{total_papers} ({(idx + 1) / total_papers * 100:.1f}%)")
            
            # Try multiple identifiers with fallback
            altmetric_data = self.get_altmetric_data_with_fallback(paper)
            
            # Extract features (enhanced version)
            features = self.extract_altmetric_features(altmetric_data)
            
            # Add paper metadata
            features['paper_index'] = idx
            identifiers = self.extract_identifiers_from_paper(paper)
            features['doi'] = identifiers.get('doi', '')
            features['arxiv_id'] = identifiers.get('arxiv', '')
            features['pmid'] = identifiers.get('pmid', '')
            
            if altmetric_data:
                successful_retrievals += 1
                logger.debug(f"Successfully retrieved enhanced Altmetric data for paper {idx}")
            
            altmetric_features.append(features)
            
            # Batch progress save and cache persistence
            if idx > 0 and idx % batch_size == 0:
                self.save_progress(altmetric_features, idx)
                self.save_cache()  # Persist cache regularly
        
        # Final cache save
        self.save_cache()
        
        # Convert to DataFrame
        altmetric_df = pd.DataFrame(altmetric_features)
        
        # Generate enhanced summary statistics
        self.print_enhanced_processing_summary(altmetric_df, successful_retrievals)
        
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
    
    def print_enhanced_processing_summary(self, altmetric_df, successful_retrievals):
        """Print enhanced summary of Altmetric data processing"""
        logger.info("\n" + "="*70)
        logger.info("ENHANCED ALTMETRIC DATA PROCESSING SUMMARY")
        logger.info("="*70)
        
        total_papers = len(altmetric_df)
        papers_with_data = altmetric_df['has_altmetric_data'].sum()
        
        logger.info(f"Total papers processed: {total_papers}")
        logger.info(f"Papers with Altmetric data: {papers_with_data} ({papers_with_data/total_papers*100:.1f}%)")
        logger.info(f"Successful API retrievals: {successful_retrievals}")
        logger.info(f"API calls made: {self.api_calls_made}")
        logger.info(f"Success rate: {self.successful_calls/self.api_calls_made*100 if self.api_calls_made > 0 else 0:.1f}%")
        logger.info(f"Cache size: {len(self.cached_results)} entries")
        
        if papers_with_data > 0:
            papers_with_data_df = altmetric_df[altmetric_df['has_altmetric_data'] == 1]
            
            logger.info("\nEnhanced Social Media Statistics:")
            
            # Core metrics
            scores = papers_with_data_df['altmetric_score']
            logger.info(f"  Altmetric scores - Mean: {scores.mean():.2f}, Max: {scores.max():.2f}")
            
            # Social media breakdown
            twitter = papers_with_data_df['twitter_mentions']
            facebook = papers_with_data_df['facebook_mentions']
            reddit = papers_with_data_df['reddit_mentions']
            
            logger.info(f"  Twitter mentions - Papers: {(twitter > 0).sum()}, Mean: {twitter.mean():.2f}")
            logger.info(f"  Facebook mentions - Papers: {(facebook > 0).sum()}, Mean: {facebook.mean():.2f}")
            logger.info(f"  Reddit mentions - Papers: {(reddit > 0).sum()}, Mean: {reddit.mean():.2f}")
            
            # Media coverage
            news = papers_with_data_df['news_mentions']
            blogs = papers_with_data_df['blog_mentions']
            
            logger.info(f"  News coverage - Papers: {(news > 0).sum()}, Mean: {news.mean():.2f}")
            logger.info(f"  Blog mentions - Papers: {(blogs > 0).sum()}, Mean: {blogs.mean():.2f}")
            
            # Academic attention
            wikipedia = papers_with_data_df['wikipedia_mentions']
            policies = papers_with_data_df['policy_mentions']
            
            logger.info(f"  Wikipedia mentions - Papers: {(wikipedia > 0).sum()}, Mean: {wikipedia.mean():.2f}")
            logger.info(f"  Policy mentions - Papers: {(policies > 0).sum()}, Mean: {policies.mean():.2f}")
            
            # Enhanced metrics
            viral_potential = papers_with_data_df['viral_potential_score']
            cross_platform = papers_with_data_df['cross_platform_engagement']
            
            logger.info(f"\nEnhanced Engagement Metrics:")
            logger.info(f"  Viral potential - Mean: {viral_potential.mean():.2f}, Max: {viral_potential.max():.2f}")
            logger.info(f"  Cross-platform engagement - Mean: {cross_platform.mean():.2f}")
            
            # Geographic diversity
            geo_diversity = papers_with_data_df['geographic_diversity']
            if geo_diversity.max() > 0:
                logger.info(f"  Geographic diversity - Mean: {geo_diversity.mean():.3f}, Max: {geo_diversity.max():.3f}")
        
        logger.info("="*70)
    
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
        
        # Load dataset with fallback (prioritize ArXiv)
        data_files = [
            "data/datasets/cs_papers_arxiv_50k.json",  # ArXiv dataset (50K papers)
            "data/datasets/openalex_5000_papers.json"  # Fallback OpenAlex dataset
        ]
        
        data_file = None
        for file_path in data_files:
            if Path(file_path).exists():
                data_file = file_path
                break
        
        if not data_file:
            logger.error(f"No dataset found. Checked: {data_files}")
            return
        
        logger.info(f"Loading enhanced dataset from: {data_file}")
        
        # Handle different dataset formats
        if 'arxiv' in data_file.lower():
            # ArXiv dataset is in JSON Lines format
            df = pd.read_json(data_file, lines=True)
            logger.info(f"Loaded ArXiv JSON Lines dataset")
        else:
            # OpenAlex format (legacy)
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            papers = data.get('papers', [])
            df = pd.json_normalize(papers)
            logger.info(f"Loaded OpenAlex JSON dataset")
        
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
        
        # Print final enhanced summary
        print("\n" + "="*80)
        print("🌐 ENHANCED ALTMETRIC INTEGRATION COMPLETED")
        print("="*80)
        print(f"✅ Processed: {len(test_df)} papers")
        print(f"✅ Papers with Altmetric data: {altmetric_features_df['has_altmetric_data'].sum()}")
        print(f"✅ Enhanced social features extracted: {len([c for c in altmetric_features_df.columns if c not in ['paper_index', 'doi', 'arxiv_id', 'pmid']])}")
        print(f"✅ API calls made: {integrator.api_calls_made}")
        print(f"✅ Success rate: {integrator.successful_calls / integrator.api_calls_made * 100 if integrator.api_calls_made > 0 else 0:.1f}%")
        print(f"✅ Cache entries: {len(integrator.cached_results)}")
        print(f"✅ Results saved to: {output_file}")
        
        # Show sample of enhanced features
        if altmetric_features_df['has_altmetric_data'].sum() > 0:
            print("\n📊 Sample Enhanced Altmetric Features (papers with data):")
            sample_data = altmetric_features_df[altmetric_features_df['has_altmetric_data'] == 1].head(2)
            enhanced_features = [
                'altmetric_score', 'twitter_mentions', 'news_mentions', 'social_media_total',
                'viral_potential_score', 'cross_platform_engagement', 'geographic_diversity'
            ]
            for feature in enhanced_features:
                if feature in sample_data.columns:
                    values = sample_data[feature].values
                    print(f"  {feature}: {values}")
        
        print(f"\n🎯 ENHANCED Altmetric integration with improved coverage and features!")
        print(f"   • Multiple identifier fallback (DOI, ArXiv, PubMed)")
        print(f"   • Enhanced social media tracking (6 platforms)")
        print(f"   • Viral potential and cross-platform engagement scoring")
        print(f"   • Geographic and demographic diversity analysis")
        print(f"   • Persistent caching for improved efficiency")
        print(f"   • Academic influence and ratio-based features")
        print(f"   • Ready for enhanced ML model integration")
        
    except Exception as e:
        logger.error(f"Error in Altmetric integration: {e}")
        raise

if __name__ == "__main__":
    main()