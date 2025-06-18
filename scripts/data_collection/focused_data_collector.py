"""
Focused Data Collection Strategy for Early Prediction Research

Based on API testing, this implements a practical data collection approach:
1. Collect recent CS papers (2022-2024) for early prediction analysis
2. Handle rate limiting gracefully
3. Focus on papers with sufficient early citation data
4. Extract key features for virality prediction
"""

import requests
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocusedDataCollector:
    """
    Practical data collector optimized for our research constraints
    
    Key Features:
    - Rate-limited API calls (respects 429 errors)
    - Focuses on recent papers for early prediction research
    - Extracts essential features only
    - Handles API limitations gracefully
    """
    
    def __init__(self, rate_limit_seconds: float = 2.0):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.rate_limit = rate_limit_seconds
        self.last_request_time = 0
        
        # Create output directory
        os.makedirs('data/raw/focused_collection', exist_ok=True)
        
        # CS venues and conferences for targeted search
        self.cs_venues = [
            'ICML', 'NeurIPS', 'ICLR', 'AAAI', 'IJCAI', 'CVPR', 'ICCV', 'ECCV',
            'ACL', 'EMNLP', 'NAACL', 'SIGIR', 'KDD', 'WWW', 'CHI', 'ICSE'
        ]
        
        # CS-related keywords for broader search
        self.cs_keywords = [
            'machine learning', 'deep learning', 'neural networks', 'computer vision',
            'natural language processing', 'artificial intelligence', 'data mining',
            'software engineering', 'human computer interaction', 'algorithms'
        ]
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_safe_request(self, url: str, params: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Make API request with error handling and rate limiting"""
        self._wait_for_rate_limit()
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = (2 ** attempt) * 5  # Exponential backoff starting at 5 seconds
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                else:
                    logger.error(f"API Error {response.status_code}: {response.text}")
                    return None
                    
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                
        return None
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific paper"""
        url = f"{self.base_url}/paper/{paper_id}"
        params = {
            'fields': 'paperId,title,abstract,authors,venue,year,publicationDate,'
                     'citationCount,referenceCount,fieldsOfStudy,url,externalIds'
        }
        
        return self._make_safe_request(url, params)
    
    def get_paper_citations(self, paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get citations for a paper (for early citation analysis)"""
        url = f"{self.base_url}/paper/{paper_id}/citations"
        params = {
            'fields': 'citingPaper.paperId,citingPaper.title,citingPaper.publicationDate,'
                     'citingPaper.venue,citingPaper.authors',
            'limit': limit
        }
        
        result = self._make_safe_request(url, params)
        if result and 'data' in result:
            return result['data']
        return []
    
    def search_papers_by_venue(self, venue: str, year: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for papers from specific venue and year"""
        url = f"{self.base_url}/paper/search"
        params = {
            'query': f'venue:{venue}',
            'year': str(year),
            'limit': limit,
            'fields': 'paperId,title,venue,year,citationCount,fieldsOfStudy,publicationDate'
        }
        
        result = self._make_safe_request(url, params)
        if result and 'data' in result:
            return result['data']
        return []
    
    def search_papers_by_keyword(self, keyword: str, year: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for papers by keyword and year"""
        url = f"{self.base_url}/paper/search"
        params = {
            'query': keyword,
            'year': str(year),
            'limit': limit,
            'fields': 'paperId,title,venue,year,citationCount,fieldsOfStudy,publicationDate'
        }
        
        result = self._make_safe_request(url, params)
        if result and 'data' in result:
            return result['data']
        return []
    
    def is_cs_paper(self, paper: Dict[str, Any]) -> bool:
        """Determine if paper is CS-related"""
        # Check fields of study
        fields = paper.get('fieldsOfStudy', []) or []
        cs_fields = {'Computer Science', 'Machine Learning', 'Artificial Intelligence'}
        if any(field in cs_fields for field in fields):
            return True
        
        # Check venue
        venue = paper.get('venue', '') or ''
        if any(cs_venue.lower() in venue.lower() for cs_venue in self.cs_venues):
            return True
        
        return False
    
    def calculate_early_citations(self, paper: Dict[str, Any], citation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate early citation metrics (30, 60, 90 days)"""
        pub_date = paper.get('publicationDate')
        if not pub_date:
            return {'early_30': 0, 'early_60': 0, 'early_90': 0, 'total_early': 0}
        
        try:
            pub_datetime = datetime.strptime(pub_date, '%Y-%m-%d')
        except ValueError:
            return {'early_30': 0, 'early_60': 0, 'early_90': 0, 'total_early': 0}
        
        early_30 = 0
        early_60 = 0
        early_90 = 0
        
        for citation in citation_data:
            citing_paper = citation.get('citingPaper', {})
            cite_date = citing_paper.get('publicationDate')
            
            if cite_date:
                try:
                    cite_datetime = datetime.strptime(cite_date, '%Y-%m-%d')
                    days_diff = (cite_datetime - pub_datetime).days
                    
                    if 0 <= days_diff <= 30:
                        early_30 += 1
                    if 0 <= days_diff <= 60:
                        early_60 += 1
                    if 0 <= days_diff <= 90:
                        early_90 += 1
                        
                except ValueError:
                    continue
        
        return {
            'early_30': early_30,
            'early_60': early_60, 
            'early_90': early_90,
            'total_early': early_90
        }
    
    def collect_venue_papers(self, year: int, max_papers_per_venue: int = 20) -> List[Dict[str, Any]]:
        """Collect papers from major CS venues for a specific year"""
        all_papers = []
        
        logger.info(f"Collecting papers from CS venues for {year}...")
        
        for venue in self.cs_venues:
            logger.info(f"Searching {venue} {year}...")
            
            papers = self.search_papers_by_venue(venue, year, max_papers_per_venue)
            
            for paper in papers:
                if self.is_cs_paper(paper):
                    # Get detailed information
                    detailed_paper = self.get_paper_details(paper['paperId'])
                    if detailed_paper:
                        # Get citation information
                        citations = self.get_paper_citations(paper['paperId'])
                        early_metrics = self.calculate_early_citations(detailed_paper, citations)
                        
                        # Combine all information
                        combined_paper = {
                            **detailed_paper,
                            'early_citation_metrics': early_metrics,
                            'total_citations_at_collection': detailed_paper.get('citationCount', 0),
                            'collection_date': datetime.now().isoformat(),
                            'data_source': 'semantic_scholar'
                        }
                        
                        all_papers.append(combined_paper)
                        
                        logger.info(f"  Added: {detailed_paper.get('title', 'Unknown')[:50]}... "
                                   f"(Citations: {detailed_paper.get('citationCount', 0)}, "
                                   f"Early 90d: {early_metrics['early_90']})")
            
            # Progress update
            logger.info(f"Collected {len(all_papers)} papers so far...")
            
            # Save intermediate results
            if len(all_papers) % 10 == 0:
                self.save_papers(all_papers, f'intermediate_{year}_{len(all_papers)}.json')
        
        return all_papers
    
    def collect_keyword_papers(self, year: int, max_papers_per_keyword: int = 10) -> List[Dict[str, Any]]:
        """Collect papers by CS keywords"""
        all_papers = []
        
        logger.info(f"Collecting papers by keywords for {year}...")
        
        for keyword in self.cs_keywords:
            logger.info(f"Searching '{keyword}' {year}...")
            
            papers = self.search_papers_by_keyword(keyword, year, max_papers_per_keyword)
            
            for paper in papers:
                if self.is_cs_paper(paper):
                    # Avoid duplicates
                    if not any(p['paperId'] == paper['paperId'] for p in all_papers):
                        # Get detailed information
                        detailed_paper = self.get_paper_details(paper['paperId'])
                        if detailed_paper:
                            # Get citation information
                            citations = self.get_paper_citations(paper['paperId'])
                            early_metrics = self.calculate_early_citations(detailed_paper, citations)
                            
                            # Combine all information
                            combined_paper = {
                                **detailed_paper,
                                'early_citation_metrics': early_metrics,
                                'search_keyword': keyword,
                                'collection_date': datetime.now().isoformat(),
                                'data_source': 'semantic_scholar'
                            }
                            
                            all_papers.append(combined_paper)
        
        return all_papers
    
    def save_papers(self, papers: List[Dict[str, Any]], filename: str):
        """Save collected papers to JSON file"""
        filepath = f'data/raw/focused_collection/{filename}'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(papers)} papers to {filepath}")
    
    def generate_collection_report(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a report of the collected data"""
        if not papers:
            return {}
        
        total_papers = len(papers)
        cs_papers = sum(1 for p in papers if self.is_cs_paper(p))
        
        # Citation statistics
        citations = [p.get('citationCount', 0) for p in papers]
        early_90_citations = [p.get('early_citation_metrics', {}).get('early_90', 0) for p in papers]
        
        # Venue distribution
        venues = {}
        for paper in papers:
            venue = paper.get('venue', 'Unknown')
            venues[venue] = venues.get(venue, 0) + 1
        
        # Year distribution
        years = {}
        for paper in papers:
            year = paper.get('year', 'Unknown')
            years[year] = years.get(year, 0) + 1
        
        report = {
            'collection_summary': {
                'total_papers': total_papers,
                'cs_papers': cs_papers,
                'cs_percentage': (cs_papers / total_papers * 100) if total_papers > 0 else 0
            },
            'citation_statistics': {
                'avg_citations': sum(citations) / len(citations) if citations else 0,
                'max_citations': max(citations) if citations else 0,
                'min_citations': min(citations) if citations else 0,
                'avg_early_90d': sum(early_90_citations) / len(early_90_citations) if early_90_citations else 0
            },
            'venue_distribution': dict(sorted(venues.items(), key=lambda x: x[1], reverse=True)[:10]),
            'year_distribution': dict(sorted(years.items())),
            'collection_date': datetime.now().isoformat()
        }
        
        return report

def main():
    """Main collection function"""
    collector = FocusedDataCollector(rate_limit_seconds=3.0)  # Conservative rate limiting
    
    # Collect papers for recent years (focus on 2022-2023 for good early citation data)
    all_papers = []
    
    for year in [2022, 2023]:
        logger.info(f"\n=== Collecting papers for {year} ===")
        
        # Collect from major venues
        venue_papers = collector.collect_venue_papers(year, max_papers_per_venue=5)  # Small batch for testing
        all_papers.extend(venue_papers)
        
        # Save intermediate results
        collector.save_papers(all_papers, f'collected_papers_{year}.json')
    
    # Generate and save final report
    report = collector.generate_collection_report(all_papers)
    with open('data/raw/focused_collection/collection_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n=== Collection Complete ===")
    print(f"Total papers collected: {len(all_papers)}")
    print(f"CS papers: {report['collection_summary']['cs_papers']}")
    print(f"Average citations: {report['citation_statistics']['avg_citations']:.1f}")
    print(f"Average early citations (90d): {report['citation_statistics']['avg_early_90d']:.1f}")
    print(f"\nTop venues:")
    for venue, count in list(report['venue_distribution'].items())[:5]:
        print(f"  - {venue}: {count}")

if __name__ == "__main__":
    main()