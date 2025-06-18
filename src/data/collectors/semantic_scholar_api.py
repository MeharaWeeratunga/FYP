"""
Semantic Scholar API Client for Research Paper Data Collection

This module provides functionality to collect academic papers from the Semantic Scholar
Academic Graph API, focusing on computer science publications for virality prediction.

Key Features:
- Rate-limited API calls with exponential backoff
- Bulk data collection with pagination
- CS-specific paper filtering
- Early citation signal extraction
- SPECTER embeddings integration
"""

import requests
import time
import json
import logging
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PaperData:
    """Data structure for paper information"""
    paper_id: str
    title: str
    abstract: Optional[str]
    authors: List[Dict[str, Any]]
    venue: Optional[str]
    year: Optional[int]
    publication_date: Optional[str]
    citation_count: int
    reference_count: int
    fields_of_study: List[str]
    is_cs_paper: bool
    early_citations: List[Dict[str, Any]]  # Citations within first 90 days
    specter_embedding: Optional[List[float]]
    url: Optional[str]
    pdf_url: Optional[str]
    github_urls: List[str]
    
class SemanticScholarAPI:
    """
    Semantic Scholar API client optimized for early virality prediction research
    
    Features:
    - Handles rate limiting and API quotas
    - Filters for computer science papers
    - Extracts early citation signals (30-90 day window)
    - Collects reproducibility signals (GitHub links, code availability)
    """
    
    BASE_URL = "https://api.semanticscholar.org"
    API_VERSION = "graph/v1"
    
    # CS-specific venues and fields
    CS_VENUES = {
        'ICML', 'NeurIPS', 'ICLR', 'AAAI', 'IJCAI', 'KDD', 'WWW', 'SIGIR',
        'CVPR', 'ICCV', 'ECCV', 'ACL', 'EMNLP', 'NAACL', 'ICCL', 'ICSE',
        'FSE', 'ASE', 'PLDI', 'POPL', 'OSDI', 'SOSP', 'NSDI', 'SIGCOMM',
        'STOC', 'FOCS', 'SODA', 'ICALP', 'CRYPTO', 'EUROCRYPT', 'CCS',
        'OAKLAND', 'USENIX Security', 'NDSS'
    }
    
    CS_FIELDS = {
        'Computer Science', 'Machine Learning', 'Artificial Intelligence',
        'Computer Vision', 'Natural Language Processing', 'Data Mining',
        'Human-Computer Interaction', 'Software Engineering', 'Programming Languages',
        'Computer Systems', 'Computer Networks', 'Computer Security',
        'Theory of Computation', 'Algorithms and Data Structures'
    }
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.0):
        """
        Initialize Semantic Scholar API client
        
        Args:
            api_key: Optional API key for higher rate limits
            rate_limit: Minimum time between API calls (seconds)
        """
        self.api_key = api_key or os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        
        # Set headers
        headers = {
            'User-Agent': 'FYP-Virality-Prediction/1.0 (himasha626@gmail.com)'
        }
        if self.api_key:
            headers['x-api-key'] = self.api_key
        
        self.session.headers.update(headers)
        
        # Create data directory
        self.data_dir = Path("data/raw/semantic_scholar")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _rate_limit_wait(self):
        """Enforce rate limiting between API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
        
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make rate-limited API request with error handling
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response data
        """
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/{self.API_VERSION}/{endpoint}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return {}
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
                
        return {}
    
    def is_cs_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Determine if a paper is from computer science field
        
        Args:
            paper: Paper data from API
            
        Returns:
            True if paper is CS-related
        """
        # Check fields of study
        fields = paper.get('fieldsOfStudy', []) or []
        if fields and any(field in self.CS_FIELDS for field in fields):
            return True
            
        # Check venue
        venue = paper.get('venue', '') or ''
        if venue in self.CS_VENUES:
            return True
            
        # Check if venue contains CS-related keywords
        venue_lower = venue.lower()
        cs_keywords = ['computer', 'computing', 'artificial intelligence', 'machine learning', 
                      'software', 'programming', 'algorithm', 'data mining']
        if venue_lower and any(keyword in venue_lower for keyword in cs_keywords):
            return True
            
        return False
    
    def get_early_citations(self, paper_id: str, publication_date: str, 
                          days_window: int = 90) -> List[Dict[str, Any]]:
        """
        Get citations within the early window (default 90 days) of publication
        
        Args:
            paper_id: Semantic Scholar paper ID
            publication_date: Publication date string
            days_window: Number of days to consider as "early"
            
        Returns:
            List of early citations with metadata
        """
        if not publication_date:
            return []
            
        try:
            pub_date = datetime.strptime(publication_date, '%Y-%m-%d')
            early_cutoff = pub_date + timedelta(days=days_window)
        except ValueError:
            logger.warning(f"Invalid publication date format: {publication_date}")
            return []
        
        # Get citations
        citations_data = self._make_request(f"paper/{paper_id}/citations", 
                                          params={'limit': 1000})
        
        early_citations = []
        for citation in citations_data.get('data', []):
            citing_paper = citation.get('citingPaper', {})
            cite_date = citing_paper.get('publicationDate')
            
            if cite_date:
                try:
                    cite_datetime = datetime.strptime(cite_date, '%Y-%m-%d')
                    if cite_datetime <= early_cutoff:
                        early_citations.append({
                            'citing_paper_id': citing_paper.get('paperId'),
                            'title': citing_paper.get('title'),
                            'citation_date': cite_date,
                            'days_after_publication': (cite_datetime - pub_date).days,
                            'citing_paper_venue': citing_paper.get('venue'),
                            'citing_paper_authors': citing_paper.get('authors', [])
                        })
                except ValueError:
                    continue
                    
        return early_citations
    
    def extract_github_urls(self, paper: Dict[str, Any]) -> List[str]:
        """
        Extract GitHub URLs from paper metadata and content
        
        Args:
            paper: Paper data from API
            
        Returns:
            List of GitHub URLs
        """
        github_urls = []
        
        # Check external IDs for GitHub
        external_ids = paper.get('externalIds', {})
        if 'ArXiv' in external_ids:
            # Could implement ArXiv paper text parsing for GitHub links
            pass
            
        # Check paper URL and references
        url = paper.get('url', '')
        if 'github.com' in url.lower():
            github_urls.append(url)
            
        return github_urls
    
    def get_paper_details(self, paper_id: str) -> Optional[PaperData]:
        """
        Get comprehensive paper details including early signals
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            PaperData object or None if failed
        """
        # Get basic paper data
        paper_data = self._make_request(f"paper/{paper_id}", params={
            'fields': 'paperId,title,abstract,authors,venue,year,publicationDate,'
                     'citationCount,referenceCount,fieldsOfStudy,url,externalIds,'
                     'embedding.specter_v2'
        })
        
        if not paper_data:
            return None
            
        # Check if CS paper
        is_cs = self.is_cs_paper(paper_data)
        
        # Get early citations
        publication_date = paper_data.get('publicationDate')
        early_citations = self.get_early_citations(paper_id, publication_date) if publication_date else []
        
        # Extract GitHub URLs
        github_urls = self.extract_github_urls(paper_data)
        
        # Get SPECTER embedding
        embedding = paper_data.get('embedding', {})
        specter_embedding = embedding.get('specter_v2') if embedding else None
        
        return PaperData(
            paper_id=paper_data.get('paperId', ''),
            title=paper_data.get('title', ''),
            abstract=paper_data.get('abstract'),
            authors=paper_data.get('authors', []),
            venue=paper_data.get('venue'),
            year=paper_data.get('year'),
            publication_date=publication_date,
            citation_count=paper_data.get('citationCount', 0),
            reference_count=paper_data.get('referenceCount', 0),
            fields_of_study=paper_data.get('fieldsOfStudy', []),
            is_cs_paper=is_cs,
            early_citations=early_citations,
            specter_embedding=specter_embedding,
            url=paper_data.get('url'),
            pdf_url=paper_data.get('externalIds', {}).get('ArXiv'),
            github_urls=github_urls
        )
    
    def search_cs_papers(self, query: str = None, year_from: int = 2020, 
                        year_to: int = 2024, limit: int = 1000) -> Generator[PaperData, None, None]:
        """
        Search for computer science papers within specified timeframe
        
        Args:
            query: Search query (default: broad CS search)
            year_from: Start year for papers
            year_to: End year for papers
            limit: Maximum number of papers to retrieve
            
        Yields:
            PaperData objects for CS papers
        """
        if query is None:
            # Default query for CS papers
            query = ('fieldsOfStudy:Computer Science OR fieldsOfStudy:Machine Learning OR '
                    'fieldsOfStudy:Artificial Intelligence OR venue:ICML OR venue:NeurIPS OR '
                    'venue:ICLR OR venue:AAAI OR venue:CVPR')
        
        offset = 0
        batch_size = 100
        collected = 0
        
        while collected < limit:
            current_limit = min(batch_size, limit - collected)
            
            search_params = {
                'query': query,
                'offset': offset,
                'limit': current_limit,
                'fields': 'paperId,title,abstract,authors,venue,year,publicationDate,'
                         'citationCount,referenceCount,fieldsOfStudy,url,externalIds',
                'year': f"{year_from}-{year_to}"
            }
            
            results = self._make_request("paper/search", params=search_params)
            
            if not results or 'data' not in results:
                break
                
            papers = results['data']
            if not papers:
                break
                
            for paper in papers:
                # Get detailed paper data
                paper_details = self.get_paper_details(paper['paperId'])
                
                if paper_details and paper_details.is_cs_paper:
                    yield paper_details
                    collected += 1
                    
                    if collected >= limit:
                        break
                        
            offset += batch_size
            
            # Progress logging
            if collected % 100 == 0:
                logger.info(f"Collected {collected} CS papers...")
    
    def save_papers_to_json(self, papers: List[PaperData], filename: str):
        """
        Save collected papers to JSON file
        
        Args:
            papers: List of PaperData objects
            filename: Output filename
        """
        filepath = self.data_dir / filename
        
        papers_dict = []
        for paper in papers:
            paper_dict = {
                'paper_id': paper.paper_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'authors': paper.authors,
                'venue': paper.venue,
                'year': paper.year,
                'publication_date': paper.publication_date,
                'citation_count': paper.citation_count,
                'reference_count': paper.reference_count,
                'fields_of_study': paper.fields_of_study,
                'is_cs_paper': paper.is_cs_paper,
                'early_citations': paper.early_citations,
                'early_citation_count': len(paper.early_citations),
                'specter_embedding': paper.specter_embedding,
                'url': paper.url,
                'pdf_url': paper.pdf_url,
                'github_urls': paper.github_urls,
                'has_github': len(paper.github_urls) > 0
            }
            papers_dict.append(paper_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(papers_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(papers)} papers to {filepath}")

def main():
    """Main function for testing and data collection"""
    # Initialize API client
    api = SemanticScholarAPI(rate_limit=1.0)  # 1 second between requests
    
    # Collect recent CS papers for early prediction research
    papers = []
    
    logger.info("Starting CS paper collection for virality prediction...")
    
    # Focus on recent papers (2022-2024) for early prediction analysis
    for paper in api.search_cs_papers(year_from=2022, year_to=2024, limit=100):
        papers.append(paper)
        
        if len(papers) % 10 == 0:
            logger.info(f"Collected {len(papers)} papers...")
    
    # Save collected data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cs_papers_early_prediction_{timestamp}.json"
    api.save_papers_to_json(papers, filename)
    
    # Print statistics
    cs_papers = sum(1 for p in papers if p.is_cs_paper)
    papers_with_early_citations = sum(1 for p in papers if len(p.early_citations) > 0)
    papers_with_github = sum(1 for p in papers if len(p.github_urls) > 0)
    
    logger.info(f"Collection complete!")
    logger.info(f"Total papers: {len(papers)}")
    logger.info(f"CS papers: {cs_papers}")
    logger.info(f"Papers with early citations (90 days): {papers_with_early_citations}")
    logger.info(f"Papers with GitHub links: {papers_with_github}")

if __name__ == "__main__":
    main()