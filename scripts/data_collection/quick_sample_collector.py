"""
Quick sample collection to get a small dataset for analysis
"""

import requests
import time
import json
import os
from datetime import datetime

def collect_sample_papers():
    """Collect a small sample of papers for analysis"""
    print("=== Quick Sample Collection ===")
    
    # Create directory
    os.makedirs('data/raw/sample', exist_ok=True)
    
    base_url = "https://api.semanticscholar.org/graph/v1"
    
    # Search for recent ML papers
    search_url = f"{base_url}/paper/search"
    
    searches = [
        {'query': 'machine learning', 'year': '2023'},
        {'query': 'deep learning', 'year': '2023'},
        {'query': 'computer vision', 'year': '2022'},
        {'query': 'natural language processing', 'year': '2022'}
    ]
    
    all_papers = []
    
    for i, search_params in enumerate(searches):
        print(f"\nSearch {i+1}: {search_params['query']} {search_params['year']}")
        
        params = {
            **search_params,
            'limit': 5,  # Just 5 papers per search
            'fields': 'paperId,title,abstract,venue,year,citationCount,fieldsOfStudy,publicationDate,authors'
        }
        
        try:
            time.sleep(2)  # Rate limiting
            response = requests.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                
                print(f"  Found {len(papers)} papers")
                
                for j, paper in enumerate(papers):
                    # Basic info
                    title = paper.get('title', 'No title')
                    citations = paper.get('citationCount', 0)
                    venue = paper.get('venue', 'No venue')
                    fields = paper.get('fieldsOfStudy', [])
                    
                    print(f"    {j+1}. {title[:50]}...")
                    print(f"       Venue: {venue}")
                    print(f"       Citations: {citations}")
                    print(f"       Fields: {fields}")
                    
                    # Check if CS paper
                    cs_fields = {'Computer Science', 'Machine Learning', 'Artificial Intelligence'}
                    is_cs = any(field in cs_fields for field in (fields or []))
                    
                    if is_cs:
                        # Add to our dataset
                        enhanced_paper = {
                            **paper,
                            'is_cs_paper': True,
                            'search_query': search_params['query'],
                            'collection_date': datetime.now().isoformat()
                        }
                        all_papers.append(enhanced_paper)
                        print(f"       ✓ Added to dataset")
                    else:
                        print(f"       - Not CS paper")
                
            elif response.status_code == 429:
                print(f"  Rate limited for search {i+1}")
                time.sleep(10)
            else:
                print(f"  Error {response.status_code} for search {i+1}")
                
        except Exception as e:
            print(f"  Exception in search {i+1}: {e}")
    
    # Save collected papers
    if all_papers:
        filename = f'data/raw/sample/sample_papers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_papers, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {len(all_papers)} papers to {filename}")
        
        # Generate basic statistics
        analyze_sample_data(all_papers)
        
    else:
        print("\n✗ No papers collected")

def analyze_sample_data(papers):
    """Quick analysis of collected papers"""
    print(f"\n=== Sample Data Analysis ===")
    
    total = len(papers)
    print(f"Total papers: {total}")
    
    # Citation statistics
    citations = [p.get('citationCount', 0) for p in papers]
    if citations:
        print(f"Citation stats:")
        print(f"  - Average: {sum(citations)/len(citations):.1f}")
        print(f"  - Range: {min(citations)} - {max(citations)}")
    
    # Venue distribution
    venues = {}
    for paper in papers:
        venue = paper.get('venue', 'Unknown')
        venues[venue] = venues.get(venue, 0) + 1
    
    print(f"\nVenues:")
    for venue, count in sorted(venues.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {venue}: {count}")
    
    # Year distribution
    years = {}
    for paper in papers:
        year = paper.get('year', 'Unknown')
        years[year] = years.get(year, 0) + 1
    
    print(f"\nYears:")
    for year, count in sorted(years.items()):
        print(f"  - {year}: {count}")
    
    # Abstract availability
    with_abstract = sum(1 for p in papers if p.get('abstract'))
    print(f"\nPapers with abstracts: {with_abstract}/{total} ({with_abstract/total*100:.1f}%)")
    
    # High-impact papers (for later virality analysis)
    high_impact = [p for p in papers if p.get('citationCount', 0) > 50]
    medium_impact = [p for p in papers if 10 < p.get('citationCount', 0) <= 50]
    low_impact = [p for p in papers if p.get('citationCount', 0) <= 10]
    
    print(f"\nImpact distribution:")
    print(f"  - High impact (>50 citations): {len(high_impact)}")
    print(f"  - Medium impact (10-50 citations): {len(medium_impact)}")
    print(f"  - Low impact (≤10 citations): {len(low_impact)}")
    
    if high_impact:
        print(f"\nHigh-impact papers:")
        for paper in high_impact:
            title = paper.get('title', 'Unknown')[:60]
            citations = paper.get('citationCount', 0)
            venue = paper.get('venue', 'Unknown')
            print(f"  - {title}... ({citations} citations, {venue})")

if __name__ == "__main__":
    collect_sample_papers()