"""
Production Data Collection Pipeline for Early Virality Prediction Research

This module implements a robust, production-ready data collection system
based on lessons learned from initial testing and API exploration.

Key Features:
- Robust error handling and retry logic
- Rate limiting compliance
- Data validation and quality checks
- Incremental collection with progress tracking
- Focus on CS papers with early prediction potential
"""

import requests
import time
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CollectionConfig:
    """Configuration for data collection"""
    rate_limit_seconds: float = 3.0
    max_retries: int = 3
    papers_per_venue: int = 50
    papers_per_keyword: int = 30
    min_citation_count: int = 5  # Filter very low-impact papers
    target_years: List[int] = None
    output_dir: str = "data/raw/production"
    
    def __post_init__(self):
        if self.target_years is None:
            self.target_years = [2022, 2023, 2024]

@dataclass
class PaperRecord:
    """Structured paper data for consistent processing"""
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
    url: Optional[str]
    open_access_pdf: Optional[Dict[str, Any]]
    
    # Derived fields
    is_cs_paper: bool = False
    collection_date: str = ""
    data_source: str = "semantic_scholar"
    age_days: Optional[int] = None
    citation_velocity: Optional[float] = None
    
    def __post_init__(self):
        if not self.collection_date:
            self.collection_date = datetime.now().isoformat()
        
        # Calculate age if publication date available
        if self.publication_date:
            try:
                pub_date = datetime.strptime(self.publication_date, '%Y-%m-%d')
                self.age_days = (datetime.now() - pub_date).days
                
                # Calculate citation velocity (citations per month)
                if self.age_days > 0:
                    age_months = self.age_days / 30.44  # Average days per month
                    self.citation_velocity = self.citation_count / age_months if age_months > 0 else 0
            except ValueError:
                pass

class ProductionCollector:
    """
    Production-ready data collector for CS papers
    
    Features:
    - Robust API handling with retry logic
    - CS paper classification
    - Quality filtering
    - Progress tracking and resumption
    - Data validation
    """
    
    # High-impact CS venues for targeted collection
    CS_VENUES = [
        # ML/AI Top Venues
        'ICML', 'NeurIPS', 'ICLR', 'AAAI', 'IJCAI',
        # Computer Vision
        'CVPR', 'ICCV', 'ECCV',
        # NLP
        'ACL', 'EMNLP', 'NAACL', 'COLING',
        # Data Mining/Web
        'KDD', 'WWW', 'SIGIR', 'WSDM',
        # HCI
        'CHI', 'UIST', 'UbiComp',
        # Systems
        'OSDI', 'SOSP', 'NSDI', 'SIGCOMM',
        # Software Engineering
        'ICSE', 'FSE', 'ASE',
        # Security
        'CCS', 'USENIX Security', 'NDSS', 'Oakland'
    ]
    
    CS_KEYWORDS = [
        'machine learning', 'deep learning', 'neural networks',
        'computer vision', 'natural language processing',
        'artificial intelligence', 'data mining', 'algorithms',
        'software engineering', 'human computer interaction',
        'computer graphics', 'robotics', 'computer security'
    ]
    
    CS_FIELDS = {
        'Computer Science', 'Machine Learning', 'Artificial Intelligence',
        'Computer Vision', 'Natural Language Processing', 'Data Mining',
        'Human-Computer Interaction', 'Software Engineering',
        'Computer Graphics', 'Robotics', 'Computer Security'
    }
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.last_request_time = 0
        self.session = requests.Session()
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': 'Academic-Research-Virality-Prediction/1.0'
        })
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.output_dir / "collection_progress.json"
        self.collected_paper_ids = set()
        self.load_progress()
    
    def load_progress(self):
        """Load previous collection progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.collected_paper_ids = set(progress.get('collected_paper_ids', []))
                    logger.info(f"Loaded progress: {len(self.collected_paper_ids)} papers already collected")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
    
    def save_progress(self):
        """Save collection progress"""
        progress = {
            'collected_paper_ids': list(self.collected_paper_ids),
            'last_update': datetime.now().isoformat(),
            'total_collected': len(self.collected_paper_ids)
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _wait_for_rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.rate_limit_seconds:
            time.sleep(self.config.rate_limit_seconds - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with retry logic and error handling"""
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2 ** attempt) * 5
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    logger.debug(f"Resource not found: {url}")
                    return None
                else:
                    logger.error(f"API Error {response.status_code}: {response.text}")
                    return None
                    
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def is_cs_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Determine if paper is CS-related"""
        # Check fields of study
        fields = paper_data.get('fieldsOfStudy', []) or []
        if any(field in self.CS_FIELDS for field in fields):
            return True
        
        # Check venue
        venue = paper_data.get('venue', '') or ''
        if any(cs_venue.lower() in venue.lower() for cs_venue in self.CS_VENUES):
            return True
        
        # Check title for CS keywords (as fallback)
        title = paper_data.get('title', '').lower()
        if any(keyword in title for keyword in ['neural', 'algorithm', 'machine learning', 'deep learning']):
            return True
        
        return False
    
    def meets_quality_criteria(self, paper_data: Dict[str, Any]) -> bool:
        """Check if paper meets quality criteria for inclusion"""
        # Must have minimum citation count
        if paper_data.get('citationCount', 0) < self.config.min_citation_count:
            return False
        
        # Must have abstract
        if not paper_data.get('abstract'):
            return False
        
        # Must have publication date
        if not paper_data.get('publicationDate'):
            return False
        
        # Must be from target years
        year = paper_data.get('year')
        if year not in self.config.target_years:
            return False
        
        return True
    
    def create_paper_record(self, paper_data: Dict[str, Any]) -> PaperRecord:
        """Convert API response to structured PaperRecord"""
        record = PaperRecord(
            paper_id=paper_data.get('paperId', ''),
            title=paper_data.get('title', ''),
            abstract=paper_data.get('abstract'),
            authors=paper_data.get('authors', []),
            venue=paper_data.get('venue'),
            year=paper_data.get('year'),
            publication_date=paper_data.get('publicationDate'),
            citation_count=paper_data.get('citationCount', 0),
            reference_count=paper_data.get('referenceCount', 0),
            fields_of_study=paper_data.get('fieldsOfStudy', []),
            url=paper_data.get('url'),
            open_access_pdf=paper_data.get('openAccessPdf'),
            is_cs_paper=self.is_cs_paper(paper_data)
        )
        
        return record
    
    def collect_papers_by_venue(self, venue: str, year: int) -> List[PaperRecord]:
        """Collect papers from specific venue and year"""
        papers = []
        offset = 0
        batch_size = 20
        
        while len(papers) < self.config.papers_per_venue:
            params = {
                'query': f'venue:{venue}',
                'year': str(year),
                'offset': offset,
                'limit': min(batch_size, self.config.papers_per_venue - len(papers)),
                'fields': 'paperId,title,abstract,authors,venue,year,publicationDate,'
                         'citationCount,referenceCount,fieldsOfStudy,url,openAccessPdf'
            }
            
            result = self._make_request('paper/search', params)
            
            if not result or 'data' not in result:
                break
            
            batch_papers = result['data']
            if not batch_papers:
                break
            
            for paper_data in batch_papers:
                paper_id = paper_data.get('paperId')
                
                # Skip if already collected
                if paper_id in self.collected_paper_ids:
                    continue
                
                # Check quality criteria
                if not self.meets_quality_criteria(paper_data):
                    continue
                
                # Create record
                record = self.create_paper_record(paper_data)
                if record.is_cs_paper:
                    papers.append(record)
                    self.collected_paper_ids.add(paper_id)
            
            offset += batch_size
            
            # Log progress
            if len(papers) % 10 == 0:
                logger.info(f"  {venue} {year}: collected {len(papers)} papers")
        
        return papers
    
    def collect_papers_by_keyword(self, keyword: str, year: int) -> List[PaperRecord]:
        """Collect papers by keyword search"""
        papers = []
        offset = 0
        batch_size = 20
        
        while len(papers) < self.config.papers_per_keyword:
            params = {
                'query': keyword,
                'year': str(year),
                'offset': offset,
                'limit': min(batch_size, self.config.papers_per_keyword - len(papers)),
                'fields': 'paperId,title,abstract,authors,venue,year,publicationDate,'
                         'citationCount,referenceCount,fieldsOfStudy,url,openAccessPdf'
            }
            
            result = self._make_request('paper/search', params)
            
            if not result or 'data' not in result:
                break
            
            batch_papers = result['data']
            if not batch_papers:
                break
            
            for paper_data in batch_papers:
                paper_id = paper_data.get('paperId')
                
                # Skip if already collected
                if paper_id in self.collected_paper_ids:
                    continue
                
                # Check quality criteria
                if not self.meets_quality_criteria(paper_data):
                    continue
                
                # Create record
                record = self.create_paper_record(paper_data)
                if record.is_cs_paper:
                    papers.append(record)
                    self.collected_paper_ids.add(paper_id)
            
            offset += batch_size
        
        return papers
    
    def run_collection(self) -> List[PaperRecord]:
        """Run the complete data collection process"""
        logger.info("Starting production data collection...")
        logger.info(f"Target years: {self.config.target_years}")
        logger.info(f"Rate limit: {self.config.rate_limit_seconds}s")
        
        all_papers = []
        
        # Collection strategy 1: High-impact venues
        logger.info("\n=== Collecting from CS venues ===")
        for year in self.config.target_years:
            for venue in self.CS_VENUES:
                logger.info(f"Collecting {venue} {year}...")
                
                try:
                    venue_papers = self.collect_papers_by_venue(venue, year)
                    all_papers.extend(venue_papers)
                    
                    logger.info(f"  Added {len(venue_papers)} papers from {venue} {year}")
                    
                    # Save progress periodically
                    if len(all_papers) % 50 == 0:
                        self.save_progress()
                        self._save_intermediate_results(all_papers)
                        
                except Exception as e:
                    logger.error(f"Error collecting from {venue} {year}: {e}")
                    continue
        
        # Collection strategy 2: Keyword-based search
        logger.info("\n=== Collecting by keywords ===")
        for year in self.config.target_years:
            for keyword in self.CS_KEYWORDS:
                logger.info(f"Collecting '{keyword}' {year}...")
                
                try:
                    keyword_papers = self.collect_papers_by_keyword(keyword, year)
                    # Filter out duplicates
                    new_papers = [p for p in keyword_papers if p.paper_id not in {existing.paper_id for existing in all_papers}]
                    all_papers.extend(new_papers)
                    
                    logger.info(f"  Added {len(new_papers)} new papers for '{keyword}' {year}")
                    
                    # Save progress periodically
                    if len(all_papers) % 50 == 0:
                        self.save_progress()
                        self._save_intermediate_results(all_papers)
                        
                except Exception as e:
                    logger.error(f"Error collecting for '{keyword}' {year}: {e}")
                    continue
        
        # Final save
        self.save_progress()
        
        logger.info(f"\n=== Collection Complete ===")
        logger.info(f"Total papers collected: {len(all_papers)}")
        
        return all_papers
    
    def _save_intermediate_results(self, papers: List[PaperRecord]):
        """Save intermediate results during collection"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"intermediate_{len(papers)}papers_{timestamp}.json"
        
        papers_data = [asdict(paper) for paper in papers]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved intermediate results: {filename}")
    
    def save_final_results(self, papers: List[PaperRecord]) -> str:
        """Save final collection results with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save papers data
        papers_filename = self.output_dir / f"cs_papers_production_{timestamp}.json"
        papers_data = [asdict(paper) for paper in papers]
        
        with open(papers_filename, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)
        
        # Generate and save collection metadata
        metadata = self._generate_collection_metadata(papers)
        metadata_filename = self.output_dir / f"collection_metadata_{timestamp}.json"
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Final results saved:")
        logger.info(f"  Papers: {papers_filename}")
        logger.info(f"  Metadata: {metadata_filename}")
        
        return str(papers_filename)
    
    def _generate_collection_metadata(self, papers: List[PaperRecord]) -> Dict[str, Any]:
        """Generate comprehensive metadata about the collection"""
        if not papers:
            return {}
        
        # Basic statistics
        total_papers = len(papers)
        total_citations = sum(p.citation_count for p in papers)
        citation_velocities = [p.citation_velocity for p in papers if p.citation_velocity is not None]
        
        # Venue distribution
        venue_counts = {}
        for paper in papers:
            venue = paper.venue or 'Unknown'
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        # Year distribution
        year_counts = {}
        for paper in papers:
            year = paper.year or 'Unknown'
            year_counts[year] = year_counts.get(year, 0) + 1
        
        # Impact categories
        high_impact = sum(1 for p in papers if p.citation_count > 100)
        medium_impact = sum(1 for p in papers if 20 <= p.citation_count <= 100)
        low_impact = sum(1 for p in papers if p.citation_count < 20)
        
        metadata = {
            'collection_info': {
                'total_papers': total_papers,
                'collection_date': datetime.now().isoformat(),
                'config_used': asdict(self.config),
                'data_source': 'semantic_scholar'
            },
            'citation_statistics': {
                'total_citations': total_citations,
                'average_citations': total_citations / total_papers if total_papers > 0 else 0,
                'citation_velocity_avg': sum(citation_velocities) / len(citation_velocities) if citation_velocities else 0,
                'citation_velocity_range': [min(citation_velocities), max(citation_velocities)] if citation_velocities else [0, 0]
            },
            'impact_distribution': {
                'high_impact_100plus': high_impact,
                'medium_impact_20to100': medium_impact,
                'low_impact_under20': low_impact
            },
            'venue_distribution': dict(sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'year_distribution': dict(sorted(year_counts.items())),
            'quality_metrics': {
                'papers_with_abstract': sum(1 for p in papers if p.abstract),
                'papers_with_pdf': sum(1 for p in papers if p.open_access_pdf),
                'average_author_count': sum(len(p.authors) for p in papers) / total_papers if total_papers > 0 else 0
            }
        }
        
        return metadata

def main():
    """Main collection function"""
    # Configure collection
    config = CollectionConfig(
        rate_limit_seconds=3.0,  # Conservative rate limiting
        papers_per_venue=30,     # Manageable batch size
        papers_per_keyword=20,
        min_citation_count=10,   # Focus on papers with some impact
        target_years=[2022, 2023, 2024]
    )
    
    # Run collection
    collector = ProductionCollector(config)
    
    try:
        papers = collector.run_collection()
        
        if papers:
            # Save final results
            output_file = collector.save_final_results(papers)
            
            # Print summary
            print(f"\n🎉 Collection Successful!")
            print(f"📊 Papers collected: {len(papers)}")
            print(f"💾 Data saved to: {output_file}")
            print(f"📈 Average citations: {sum(p.citation_count for p in papers) / len(papers):.1f}")
            
            # Top venues
            venue_counts = {}
            for paper in papers:
                venue = paper.venue or 'Unknown'
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
            
            print(f"\n🏆 Top venues:")
            for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {venue}: {count} papers")
        
        else:
            print("❌ No papers collected. Check configuration and API access.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Collection interrupted by user")
        collector.save_progress()
        print("💾 Progress saved. You can resume later.")
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        collector.save_progress()

if __name__ == "__main__":
    main()