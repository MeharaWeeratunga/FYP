"""
Targeted collection focusing on clearly CS venues and papers
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

def collect_from_specific_venues():
    """Collect from clearly CS venues with good CS classification"""
    
    # Highly targeted CS venues and searches
    targets = [
        # Machine Learning - very specific searches
        {"venue": "ICML", "year": 2023, "target": 15, "search_modifier": " machine learning"},
        {"venue": "NeurIPS", "year": 2023, "target": 15, "search_modifier": " neural networks"},
        
        # Computer Vision - very CS specific
        {"venue": "CVPR", "year": 2023, "target": 15, "search_modifier": " computer vision"},
        {"venue": "ICCV", "year": 2021, "target": 10, "search_modifier": " computer vision"},
        
        # NLP - clearly CS
        {"venue": "ACL", "year": 2023, "target": 10, "search_modifier": " natural language processing"},
        {"venue": "EMNLP", "year": 2022, "target": 10, "search_modifier": " NLP"},
        
        # Systems - clearly CS
        {"venue": "OSDI", "year": 2022, "target": 8, "search_modifier": " operating systems"},
        {"venue": "SOSP", "year": 2021, "target": 8, "search_modifier": " systems"},
    ]
    
    config = CollectionConfig(
        rate_limit_seconds=4.0,  # Slightly more conservative
        papers_per_venue=25,
        min_citation_count=15,   # Higher threshold for quality
        target_years=[2021, 2022, 2023]
    )
    
    collector = ProductionCollector(config)
    all_collected = []
    
    for target in targets:
        venue = target["venue"]
        year = target["year"]
        count = target["target"]
        search_mod = target["search_modifier"]
        
        logger.info(f"🎯 Collecting {count} papers from {venue} {year} with '{search_mod}'")
        
        try:
            # Use custom search query for better CS targeting
            papers = []
            offset = 0
            
            while len(papers) < count:
                # Custom API call with more specific search
                search_params = {
                    'query': f'venue:{venue}{search_mod}',
                    'year': str(year),
                    'offset': offset,
                    'limit': min(10, count - len(papers)),
                    'fields': 'paperId,title,abstract,authors,venue,year,publicationDate,'
                             'citationCount,referenceCount,fieldsOfStudy,url'
                }
                
                result = collector._make_request('paper/search', search_params)
                
                if not result or 'data' not in result:
                    break
                    
                batch_papers = result['data']
                if not batch_papers:
                    break
                
                for paper_data in batch_papers:
                    if len(papers) >= count:
                        break
                    
                    # Enhanced CS filtering
                    if is_clearly_cs_paper(paper_data):
                        if collector.meets_quality_criteria(paper_data):
                            paper_record = collector.create_paper_record(paper_data)
                            papers.append(paper_record)
                            logger.info(f"  ✅ Added: {paper_record.title[:50]}... ({paper_record.citation_count} cites)")
                
                offset += 10
                
                if offset > 100:  # Don't search too deep
                    break
            
            # Save papers for this venue
            if papers:
                save_venue_papers(papers, venue, year)
                all_collected.extend(papers)
                logger.info(f"📊 {venue} {year}: Collected {len(papers)} papers")
            else:
                logger.warning(f"❌ No papers collected from {venue} {year}")
                
        except Exception as e:
            logger.error(f"Error collecting {venue} {year}: {e}")
            continue
    
    # Final summary
    logger.info(f"\n🎉 Targeted collection complete!")
    logger.info(f"📊 Total papers: {len(all_collected)}")
    
    if all_collected:
        # Quality check
        cs_count = sum(1 for p in all_collected if p.is_cs_paper)
        logger.info(f"🖥️ CS papers: {cs_count}/{len(all_collected)} ({cs_count/len(all_collected)*100:.1f}%)")
        
        # Citation stats
        citations = [p.citation_count for p in all_collected]
        logger.info(f"📈 Citations: avg={sum(citations)/len(citations):.1f}, range={min(citations)}-{max(citations)}")
    
    return all_collected

def is_clearly_cs_paper(paper_data):
    """Enhanced CS paper detection"""
    
    # Check fields of study first
    fields = paper_data.get('fieldsOfStudy', []) or []
    cs_fields = {
        'Computer Science', 'Machine Learning', 'Artificial Intelligence',
        'Computer Vision', 'Natural Language Processing', 'Data Mining',
        'Software Engineering', 'Human-Computer Interaction'
    }
    
    if any(field in cs_fields for field in fields):
        return True
    
    # Check venue
    venue = paper_data.get('venue', '').lower()
    cs_venues = {
        'icml', 'neurips', 'iclr', 'cvpr', 'iccv', 'eccv', 'acl', 'emnlp', 
        'naacl', 'osdi', 'sosp', 'sigcomm', 'chi', 'uist', 'kdd', 'www'
    }
    
    if any(cs_venue in venue for cs_venue in cs_venues):
        return True
    
    # Check title for CS keywords
    title = paper_data.get('title', '').lower()
    cs_keywords = {
        'neural', 'algorithm', 'machine learning', 'deep learning', 
        'computer vision', 'natural language', 'software', 'system',
        'network', 'data mining', 'artificial intelligence'
    }
    
    if any(keyword in title for keyword in cs_keywords):
        return True
    
    # Exclude obvious non-CS fields in title/abstract
    exclude_keywords = {
        'medical', 'medicine', 'clinical', 'patient', 'disease', 'therapy',
        'biological', 'biology', 'genetic', 'protein', 'cell',
        'chemical', 'chemistry', 'molecular', 'pharmaceutical',
        'tobacco', 'smoking', 'health care', 'hospital'
    }
    
    title_abstract = (title + ' ' + paper_data.get('abstract', '')).lower()
    if any(keyword in title_abstract for keyword in exclude_keywords):
        return False
    
    return False

def save_venue_papers(papers, venue, year):
    """Save papers from a specific venue"""
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"data/raw/systematic_collection/targeted_{venue}_{year}_{timestamp}.json"
    
    papers_data = [
        {
            'paper_id': p.paper_id,
            'title': p.title,
            'abstract': p.abstract,
            'venue': p.venue,
            'year': p.year,
            'citation_count': p.citation_count,
            'is_cs_paper': p.is_cs_paper,
            'fields_of_study': p.fields_of_study,
            'collection_date': p.collection_date
        }
        for p in papers
    ]
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(papers_data, f, indent=2)
    
    logger.info(f"💾 Saved to {filename}")

if __name__ == "__main__":
    try:
        papers = collect_from_specific_venues()
        print(f"\n✅ Targeted collection complete: {len(papers)} papers")
        
    except KeyboardInterrupt:
        print("\n⏹️ Collection interrupted")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        logging.exception("Collection error")