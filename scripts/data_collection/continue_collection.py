"""
Continue systematic data collection from where we left off
This script allows resuming collection in manageable chunks
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.collectors.production_collector import ProductionCollector, CollectionConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def continue_collection_from_venue(venue: str, year: int, target: int = 30):
    """Continue collection from a specific venue"""
    logger.info(f"🎯 Continuing {venue} {year} collection (target: {target})")
    
    # Initialize collector
    config = CollectionConfig(
        rate_limit_seconds=3.0,
        papers_per_venue=target,
        min_citation_count=10,
        target_years=[year]
    )
    
    collector = ProductionCollector(config)
    
    try:
        # Collect papers
        papers = collector.collect_papers_by_venue(venue, year)
        
        if papers:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/systematic_collection/{venue}_{year}_{timestamp}.json"
            
            # Convert to dict format
            papers_data = [
                {
                    'paper_id': p.paper_id,
                    'title': p.title,
                    'abstract': p.abstract,
                    'venue': p.venue,
                    'year': p.year,
                    'citation_count': p.citation_count,
                    'is_cs_paper': p.is_cs_paper,
                    'collection_date': p.collection_date
                }
                for p in papers
            ]
            
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(papers_data, f, indent=2)
            
            logger.info(f"✅ Collected {len(papers)} papers from {venue} {year}")
            logger.info(f"💾 Saved to {filename}")
            
            # Print sample papers
            logger.info("📄 Sample papers collected:")
            for i, paper in enumerate(papers[:3]):
                logger.info(f"  {i+1}. {paper.title[:60]}... ({paper.citation_count} citations)")
            
            return papers
        else:
            logger.warning(f"❌ No papers collected from {venue} {year}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Error collecting {venue} {year}: {e}")
        return []

def quick_collection_batch():
    """Run a quick batch collection"""
    logger.info("🚀 Starting quick collection batch")
    
    # Priority venues to collect from
    venues_to_collect = [
        ("ICML", 2023, 25),
        ("NeurIPS", 2022, 30),
        ("CVPR", 2023, 25),
        ("ACL", 2023, 20)
    ]
    
    all_papers = []
    
    for venue, year, target in venues_to_collect:
        papers = continue_collection_from_venue(venue, year, target)
        all_papers.extend(papers)
        
        logger.info(f"📊 Total collected so far: {len(all_papers)} papers")
        
        # Break if we have enough for this session
        if len(all_papers) >= 50:
            logger.info("🛑 Reached session target of 50 papers")
            break
    
    # Summary
    if all_papers:
        logger.info(f"\n🎉 Batch collection complete!")
        logger.info(f"📊 Total papers this session: {len(all_papers)}")
        
        # Show venue distribution
        venue_counts = {}
        for paper in all_papers:
            venue = getattr(paper, 'venue', 'Unknown')
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        logger.info("🏆 Venue distribution:")
        for venue, count in venue_counts.items():
            logger.info(f"  - {venue}: {count} papers")
    
    return all_papers

if __name__ == "__main__":
    try:
        papers = quick_collection_batch()
        print(f"\n✅ Success! Collected {len(papers)} papers")
        print("📋 Check data/raw/systematic_collection/ for saved files")
        
    except KeyboardInterrupt:
        print("\n⏹️ Collection stopped by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        logging.exception("Collection error")