"""
Test script to collect sample data and analyze what we're getting from Semantic Scholar API
"""

import sys
import os
sys.path.append('src')

from src.data.collectors.semantic_scholar_api import SemanticScholarAPI
import json
from datetime import datetime

def test_semantic_scholar_api():
    """Test the Semantic Scholar API with a small sample"""
    print("=== Testing Semantic Scholar API ===")
    
    # Initialize API client (no key required for basic usage)
    api = SemanticScholarAPI(rate_limit=2.0)  # 2 seconds between requests to be safe
    
    # Test 1: Get a specific well-known CS paper
    print("\n1. Testing single paper retrieval...")
    
    # Let's try to get a well-known ML paper (example paper ID)
    test_paper_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"  # A famous BERT paper
    
    try:
        paper = api.get_paper_details(test_paper_id)
        if paper:
            print(f"✓ Successfully retrieved paper: {paper.title}")
            print(f"  - Year: {paper.year}")
            print(f"  - Venue: {paper.venue}")
            print(f"  - Citation count: {paper.citation_count}")
            print(f"  - Is CS paper: {paper.is_cs_paper}")
            print(f"  - Early citations (90 days): {len(paper.early_citations)}")
            print(f"  - Has SPECTER embedding: {paper.specter_embedding is not None}")
            print(f"  - GitHub URLs: {len(paper.github_urls)}")
        else:
            print("✗ Failed to retrieve test paper")
    except Exception as e:
        print(f"✗ Error retrieving test paper: {e}")
    
    # Test 2: Search for recent CS papers
    print("\n2. Testing paper search...")
    
    try:
        papers = []
        search_query = "machine learning ICML 2023"
        
        print(f"Searching for: {search_query}")
        
        # Collect just 5 papers for testing
        for i, paper in enumerate(api.search_cs_papers(query=search_query, 
                                                      year_from=2023, 
                                                      year_to=2023, 
                                                      limit=5)):
            papers.append(paper)
            print(f"  Paper {i+1}: {paper.title[:60]}...")
            print(f"    - Venue: {paper.venue}")
            print(f"    - Citations: {paper.citation_count}")
            print(f"    - Early citations: {len(paper.early_citations)}")
            
            if i >= 4:  # Stop at 5 papers
                break
        
        print(f"\n✓ Successfully collected {len(papers)} papers")
        
        # Save sample data
        if papers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sample_papers_{timestamp}.json"
            api.save_papers_to_json(papers, filename)
            print(f"✓ Saved sample data to: {filename}")
            
            # Analyze the sample data
            analyze_sample_data(papers)
            
    except Exception as e:
        print(f"✗ Error in paper search: {e}")
        import traceback
        traceback.print_exc()

def analyze_sample_data(papers):
    """Analyze the collected sample data"""
    print("\n=== Sample Data Analysis ===")
    
    if not papers:
        print("No papers to analyze")
        return
    
    # Basic statistics
    total_papers = len(papers)
    cs_papers = sum(1 for p in papers if p.is_cs_paper)
    papers_with_abstracts = sum(1 for p in papers if p.abstract)
    papers_with_early_citations = sum(1 for p in papers if len(p.early_citations) > 0)
    papers_with_specter = sum(1 for p in papers if p.specter_embedding is not None)
    papers_with_github = sum(1 for p in papers if len(p.github_urls) > 0)
    
    print(f"Total papers collected: {total_papers}")
    print(f"CS papers: {cs_papers} ({cs_papers/total_papers*100:.1f}%)")
    print(f"Papers with abstracts: {papers_with_abstracts} ({papers_with_abstracts/total_papers*100:.1f}%)")
    print(f"Papers with early citations: {papers_with_early_citations} ({papers_with_early_citations/total_papers*100:.1f}%)")
    print(f"Papers with SPECTER embeddings: {papers_with_specter} ({papers_with_specter/total_papers*100:.1f}%)")
    print(f"Papers with GitHub links: {papers_with_github} ({papers_with_github/total_papers*100:.1f}%)")
    
    # Citation statistics
    citation_counts = [p.citation_count for p in papers]
    early_citation_counts = [len(p.early_citations) for p in papers]
    
    if citation_counts:
        print(f"\nCitation Statistics:")
        print(f"  - Average citations: {sum(citation_counts)/len(citation_counts):.1f}")
        print(f"  - Max citations: {max(citation_counts)}")
        print(f"  - Min citations: {min(citation_counts)}")
        
    if any(early_citation_counts):
        print(f"Early Citation Statistics (90 days):")
        print(f"  - Average early citations: {sum(early_citation_counts)/len(early_citation_counts):.1f}")
        print(f"  - Max early citations: {max(early_citation_counts)}")
    
    # Venue distribution
    venues = [p.venue for p in papers if p.venue]
    if venues:
        venue_counts = {}
        for venue in venues:
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        print(f"\nVenue Distribution:")
        for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {venue}: {count}")
    
    # Year distribution
    years = [p.year for p in papers if p.year]
    if years:
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        print(f"\nYear Distribution:")
        for year, count in sorted(year_counts.items()):
            print(f"  - {year}: {count}")
    
    # Fields of study
    all_fields = []
    for paper in papers:
        all_fields.extend(paper.fields_of_study)
    
    if all_fields:
        field_counts = {}
        for field in all_fields:
            field_counts[field] = field_counts.get(field, 0) + 1
        
        print(f"\nTop Fields of Study:")
        for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {field}: {count}")

def test_api_without_key():
    """Test if we can access the API without a key"""
    print("\n=== Testing API Access ===")
    
    import requests
    
    # Test basic API access
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': 'machine learning',
        'limit': 1,
        'fields': 'paperId,title,year'
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"API Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ API accessible without key")
            print(f"Sample response: {data.get('data', [{}])[0] if data.get('data') else 'No data'}")
        else:
            print(f"✗ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"✗ API Connection Error: {e}")

if __name__ == "__main__":
    print("Starting data collection test...")
    
    # First test basic API access
    test_api_without_key()
    
    # Then test our implementation
    test_semantic_scholar_api()
    
    print("\nTest complete!")