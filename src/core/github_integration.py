"""
GitHub API Integration for Code Repository Metrics
Free tier implementation for academic research - no authentication required for public repos
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

class GitHubAPIIntegrator:
    """
    GitHub API integration for tracking code repository metrics
    Free tier for academic research - no authentication required for public repositories
    """
    
    def __init__(self, rate_limit_delay=1.0):
        """
        Initialize GitHub API integrator
        
        Args:
            rate_limit_delay (float): Delay between API calls to respect rate limits
        """
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://api.github.com"
        self.session = requests.Session()
        
        # Set headers for academic research
        headers = {
            'User-Agent': 'Academic-Research-Project/1.0 (Paper-Code-Analysis)',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.session.headers.update(headers)
        
        # Statistics tracking
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.cached_results = {}
        
    def extract_github_urls_from_paper(self, paper):
        """Extract GitHub URLs from paper data"""
        github_urls = []
        
        # Check common fields for GitHub URLs
        fields_to_check = ['abstract', 'title', 'full_text', 'urls', 'links']
        
        github_pattern = r'https?://github\.com/([^/\s]+)/([^/\s]+)'
        
        for field in fields_to_check:
            content = paper.get(field, '')
            if isinstance(content, str):
                matches = re.findall(github_pattern, content)
                for owner, repo in matches:
                    # Clean repo name (remove common suffixes)
                    repo = repo.rstrip('.,;)')
                    github_urls.append(f"https://github.com/{owner}/{repo}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        matches = re.findall(github_pattern, item)
                        for owner, repo in matches:
                            repo = repo.rstrip('.,;)')
                            github_urls.append(f"https://github.com/{owner}/{repo}")
        
        return list(set(github_urls))  # Remove duplicates
    
    def parse_github_url(self, github_url):
        """Parse GitHub URL to extract owner and repository name"""
        try:
            # Remove trailing slashes and extract parts
            clean_url = github_url.rstrip('/')
            parts = clean_url.split('/')
            
            if len(parts) >= 5 and 'github.com' in clean_url:
                owner = parts[-2]
                repo = parts[-1]
                
                # Remove common file extensions or query parameters
                repo = repo.split('?')[0].split('#')[0]
                if repo.endswith('.git'):
                    repo = repo[:-4]
                
                return owner, repo
            
        except Exception as e:
            logger.warning(f"Error parsing GitHub URL {github_url}: {e}")
        
        return None, None
    
    def get_repository_metrics(self, owner, repo, max_retries=3):
        """
        Get GitHub repository metrics without authentication
        
        Args:
            owner (str): Repository owner
            repo (str): Repository name
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            dict: Repository metrics or None if not found
        """
        cache_key = f"{owner}/{repo}"
        
        # Check cache first
        if cache_key in self.cached_results:
            return self.cached_results[cache_key]
        
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        for attempt in range(max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                self.api_calls_made += 1
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    self.successful_calls += 1
                    repo_data = response.json()
                    
                    # Extract key metrics
                    metrics = self.extract_repository_features(repo_data)
                    
                    # Cache result
                    self.cached_results[cache_key] = metrics
                    
                    logger.debug(f"✓ Found GitHub repo: {owner}/{repo} - Stars: {metrics.get('stars', 0)}")
                    return metrics
                
                elif response.status_code == 404:
                    # Repository not found or private
                    logger.debug(f"Repository not found: {owner}/{repo}")
                    self.cached_results[cache_key] = None
                    return None
                
                elif response.status_code == 403:
                    # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded, waiting longer...")
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                else:
                    logger.warning(f"API error {response.status_code} for {owner}/{repo}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {owner}/{repo}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        self.failed_calls += 1
        return None
    
    def extract_repository_features(self, repo_data):
        """Extract meaningful features from GitHub repository data"""
        try:
            created_at = repo_data.get('created_at', '')
            updated_at = repo_data.get('updated_at', '')
            pushed_at = repo_data.get('pushed_at', '')
            
            # Calculate repository age in days
            repo_age = 0
            if created_at:
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                repo_age = (datetime.now(created_date.tzinfo) - created_date).days
            
            # Calculate days since last update
            days_since_update = 0
            if updated_at:
                updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                days_since_update = (datetime.now(updated_date.tzinfo) - updated_date).days
            
            # Calculate days since last push
            days_since_push = 0
            if pushed_at:
                pushed_date = datetime.fromisoformat(pushed_at.replace('Z', '+00:00'))
                days_since_push = (datetime.now(pushed_date.tzinfo) - pushed_date).days
            
            features = {
                # Basic metrics
                'github_stars': repo_data.get('stargazers_count', 0),
                'github_forks': repo_data.get('forks_count', 0),
                'github_watchers': repo_data.get('watchers_count', 0),
                'github_open_issues': repo_data.get('open_issues_count', 0),
                'github_size_kb': repo_data.get('size', 0),  # Size in KB
                'github_subscribers': repo_data.get('subscribers_count', 0),
                
                # Repository characteristics
                'github_is_fork': 1 if repo_data.get('fork', False) else 0,
                'github_has_wiki': 1 if repo_data.get('has_wiki', False) else 0,
                'github_has_pages': 1 if repo_data.get('has_pages', False) else 0,
                'github_has_downloads': 1 if repo_data.get('has_downloads', False) else 0,
                'github_has_issues': 1 if repo_data.get('has_issues', False) else 0,
                'github_has_projects': 1 if repo_data.get('has_projects', False) else 0,
                
                # Language and topics
                'github_language': repo_data.get('language', ''),
                'github_topics_count': len(repo_data.get('topics', [])),
                
                # Temporal features
                'github_repo_age_days': repo_age,
                'github_days_since_update': days_since_update,
                'github_days_since_push': days_since_push,
                
                # Activity indicators
                'github_is_archived': 1 if repo_data.get('archived', False) else 0,
                'github_is_disabled': 1 if repo_data.get('disabled', False) else 0,
                'github_default_branch': repo_data.get('default_branch', 'main'),
                
                # Calculated metrics
                'github_stars_per_day': repo_data.get('stargazers_count', 0) / max(repo_age, 1),
                'github_forks_per_star': repo_data.get('forks_count', 0) / max(repo_data.get('stargazers_count', 1), 1),
                'github_issues_per_star': repo_data.get('open_issues_count', 0) / max(repo_data.get('stargazers_count', 1), 1),
                
                # Repository metadata
                'github_repo_owner': repo_data.get('owner', {}).get('login', ''),
                'github_repo_name': repo_data.get('name', ''),
                'github_full_name': repo_data.get('full_name', ''),
                'github_description_length': len(repo_data.get('description', '') or ''),
                
                # Additional derived features
                'github_has_description': 1 if repo_data.get('description') else 0,
                'github_has_homepage': 1 if repo_data.get('homepage') else 0,
                'github_activity_score': self.calculate_activity_score(repo_data),
                'github_popularity_score': self.calculate_popularity_score(repo_data),
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting repository features: {e}")
            return {}
    
    def calculate_activity_score(self, repo_data):
        """Calculate repository activity score"""
        try:
            stars = repo_data.get('stargazers_count', 0)
            forks = repo_data.get('forks_count', 0)
            watchers = repo_data.get('watchers_count', 0)
            open_issues = repo_data.get('open_issues_count', 0)
            
            # Simple activity score calculation
            activity_score = (stars * 1.0 + forks * 2.0 + watchers * 0.5 + 
                            min(open_issues, 50) * 0.1)  # Cap issues at 50
            
            return activity_score
        except:
            return 0
    
    def calculate_popularity_score(self, repo_data):
        """Calculate repository popularity score"""
        try:
            stars = repo_data.get('stargazers_count', 0)
            forks = repo_data.get('forks_count', 0)
            size = repo_data.get('size', 0)
            
            # Logarithmic popularity score to handle wide ranges
            popularity_score = np.log1p(stars) + np.log1p(forks) + np.log1p(size / 1000)
            
            return popularity_score
        except:
            return 0
    
    def process_paper_batch(self, papers_df, batch_size=50):
        """
        Process a batch of papers to extract GitHub repository metrics
        
        Args:
            papers_df (pd.DataFrame): DataFrame containing paper information
            batch_size (int): Number of papers to process before saving progress
            
        Returns:
            pd.DataFrame: DataFrame with GitHub features for each paper
        """
        logger.info(f"Processing {len(papers_df)} papers for GitHub repository data...")
        
        github_features_list = []
        
        for idx, (_, paper) in enumerate(papers_df.iterrows()):
            if idx % 10 == 0:
                logger.info(f"Processing paper {idx + 1}/{len(papers_df)} ({((idx + 1)/len(papers_df)*100):.1f}%)")
            
            # Extract GitHub URLs from paper
            github_urls = self.extract_github_urls_from_paper(paper)
            
            if github_urls:
                # Process first GitHub URL found (can be extended to handle multiple)
                github_url = github_urls[0]
                owner, repo = self.parse_github_url(github_url)
                
                if owner and repo:
                    metrics = self.get_repository_metrics(owner, repo)
                    
                    if metrics:
                        features = metrics.copy()
                        features['paper_index'] = idx
                        features['github_url'] = github_url
                        features['has_github_repo'] = 1
                    else:
                        # Repository not found or private
                        features = self.create_empty_github_features()
                        features['paper_index'] = idx
                        features['github_url'] = github_url
                        features['has_github_repo'] = 0
                else:
                    # Invalid GitHub URL
                    features = self.create_empty_github_features()
                    features['paper_index'] = idx
                    features['has_github_repo'] = 0
            else:
                # No GitHub URLs found
                features = self.create_empty_github_features()
                features['paper_index'] = idx
                features['has_github_repo'] = 0
            
            github_features_list.append(features)
            
            # Save progress periodically
            if (idx + 1) % batch_size == 0:
                self.save_progress(github_features_list, idx + 1)
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"GITHUB DATA PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total papers processed: {len(papers_df)}")
        logger.info(f"Papers with GitHub repositories: {sum(1 for f in github_features_list if f['has_github_repo'] == 1)}")
        logger.info(f"API calls made: {self.api_calls_made}")
        logger.info(f"Successful calls: {self.successful_calls}")
        logger.info(f"Failed calls: {self.failed_calls}")
        
        if self.successful_calls > 0:
            successful_repos = [f for f in github_features_list if f['has_github_repo'] == 1]
            if successful_repos:
                avg_stars = np.mean([f['github_stars'] for f in successful_repos])
                avg_forks = np.mean([f['github_forks'] for f in successful_repos])
                logger.info(f"\nGitHub Repository Statistics:")
                logger.info(f"  Average stars: {avg_stars:.1f}")
                logger.info(f"  Average forks: {avg_forks:.1f}")
        logger.info(f"{'='*60}")
        
        # Convert to DataFrame
        github_df = pd.DataFrame(github_features_list)
        
        # Save final results
        self.save_final_results(github_df)
        
        return github_df
    
    def create_empty_github_features(self):
        """Create empty GitHub features for papers without repositories"""
        return {
            'github_stars': 0,
            'github_forks': 0,
            'github_watchers': 0,
            'github_open_issues': 0,
            'github_size_kb': 0,
            'github_subscribers': 0,
            'github_is_fork': 0,
            'github_has_wiki': 0,
            'github_has_pages': 0,
            'github_has_downloads': 0,
            'github_has_issues': 0,
            'github_has_projects': 0,
            'github_language': '',
            'github_topics_count': 0,
            'github_repo_age_days': 0,
            'github_days_since_update': 0,
            'github_days_since_push': 0,
            'github_is_archived': 0,
            'github_is_disabled': 0,
            'github_default_branch': '',
            'github_stars_per_day': 0,
            'github_forks_per_star': 0,
            'github_issues_per_star': 0,
            'github_repo_owner': '',
            'github_repo_name': '',
            'github_full_name': '',
            'github_description_length': 0,
            'github_has_description': 0,
            'github_has_homepage': 0,
            'github_activity_score': 0,
            'github_popularity_score': 0,
            'has_github_repo': 0
        }
    
    def save_progress(self, features_list, current_count):
        """Save progress to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_file = Path(f"results/github_progress_{current_count}_{timestamp}.json")
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to Python types for JSON serialization
            clean_features = []
            for features in features_list:
                clean_feature_dict = {}
                for key, value in features.items():
                    if isinstance(value, (np.integer, np.floating)):
                        clean_feature_dict[key] = value.item()
                    elif isinstance(value, np.ndarray):
                        clean_feature_dict[key] = value.tolist()
                    else:
                        clean_feature_dict[key] = value
                clean_features.append(clean_feature_dict)
            
            with open(progress_file, 'w') as f:
                json.dump(clean_features, f, indent=2)
            
            logger.info(f"Progress saved to {progress_file}")
            
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def save_final_results(self, github_df):
        """Save final GitHub features results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Save GitHub features summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_papers': len(github_df),
                'papers_with_repos': github_df['has_github_repo'].sum(),
                'api_calls_made': self.api_calls_made,
                'successful_calls': self.successful_calls,
                'failed_calls': self.failed_calls,
                'features': []
            }
            
            # Add feature data for papers with repositories
            repo_papers = github_df[github_df['has_github_repo'] == 1]
            for _, row in repo_papers.iterrows():
                feature_dict = {}
                for col in github_df.columns:
                    value = row[col]
                    if isinstance(value, (np.integer, np.floating)):
                        feature_dict[col] = value.item()
                    elif pd.isna(value):
                        feature_dict[col] = None
                    else:
                        feature_dict[col] = value
                summary['features'].append(feature_dict)
            
            results_file = results_dir / f"github_features_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"GitHub features saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")

def main():
    """Test GitHub API integration"""
    logger.info("Testing GitHub API integration...")
    
    integrator = GitHubAPIIntegrator(rate_limit_delay=1.0)
    
    # Test with a sample repository
    test_metrics = integrator.get_repository_metrics("microsoft", "vscode")
    
    if test_metrics:
        logger.info("✓ GitHub API integration working!")
        logger.info(f"Sample repo metrics: Stars={test_metrics['github_stars']}, "
                   f"Forks={test_metrics['github_forks']}")
    else:
        logger.error("✗ GitHub API integration failed")

if __name__ == "__main__":
    main()