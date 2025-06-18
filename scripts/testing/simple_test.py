"""
Simple test to collect basic paper data and understand the API
"""

import requests
import time
import json
from datetime import datetime

def test_basic_api():
    """Test basic Semantic Scholar API functionality"""
    print("=== Basic API Test ===")
    
    # Test 1: Get a specific paper
    print("\n1. Testing single paper retrieval...")
    
    base_url = "https://api.semanticscholar.org/graph/v1"
    
    # Test with a well-known paper (BERT paper)
    paper_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    
    url = f"{base_url}/paper/{paper_id}"
    params = {
        'fields': 'paperId,title,abstract,authors,venue,year,publicationDate,citationCount,fieldsOfStudy'
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            paper = response.json()
            print(f"✓ Paper: {paper.get('title', 'No title')}")
            print(f"  Year: {paper.get('year')}")
            print(f"  Venue: {paper.get('venue')}")
            print(f"  Citations: {paper.get('citationCount')}")
            print(f"  Fields: {paper.get('fieldsOfStudy', [])}")
            print(f"  Has abstract: {bool(paper.get('abstract'))}")
            
            # Save this example
            with open('data/raw/example_paper.json', 'w') as f:
                json.dump(paper, f, indent=2)
                
        else:
            print(f"✗ Error: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    # Test 2: Search for recent papers
    print("\n2. Testing paper search...")
    
    search_url = f"{base_url}/paper/search"
    search_params = {
        'query': 'machine learning',
        'year': '2023',
        'limit': 5,
        'fields': 'paperId,title,venue,year,citationCount,fieldsOfStudy'
    }
    
    try:
        time.sleep(1)  # Rate limiting
        response = requests.get(search_url, params=search_params)
        print(f"Search Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            papers = data.get('data', [])
            
            print(f"✓ Found {len(papers)} papers")
            
            cs_papers = []
            for i, paper in enumerate(papers):
                title = paper.get('title', 'No title')
                venue = paper.get('venue', 'No venue')
                citations = paper.get('citationCount', 0)
                fields = paper.get('fieldsOfStudy', [])
                
                print(f"\nPaper {i+1}:")
                print(f"  Title: {title[:60]}...")
                print(f"  Venue: {venue}")
                print(f"  Citations: {citations}")
                print(f"  Fields: {fields}")
                
                # Check if it's a CS paper
                cs_fields = {'Computer Science', 'Machine Learning', 'Artificial Intelligence'}
                is_cs = any(field in cs_fields for field in (fields or []))
                if is_cs:
                    cs_papers.append(paper)
                    print(f"  ✓ CS Paper")
                else:
                    print(f"  - Not clearly CS")
            
            print(f"\n✓ CS Papers found: {len(cs_papers)}/{len(papers)}")
            
            # Save search results
            with open('data/raw/search_results.json', 'w') as f:
                json.dump(papers, f, indent=2)
                
        else:
            print(f"✗ Search Error: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Search Exception: {e}")

def analyze_data_availability():
    """Analyze what data fields are available"""
    print("\n=== Data Availability Analysis ===")
    
    # Check what fields are available for papers
    base_url = "https://api.semanticscholar.org/graph/v1"
    
    # Test with multiple papers to see data consistency
    test_papers = [
        "204e3073870fae3d05bcbc2f6a8e263d9b72e776",  # BERT
        "0c28d3d86066ecc27f6a2df5a13d166e1dfc5b49",  # Transformer
        "f9c602cc436a9ea2f9e7db48c77d924e09ce3c32"   # Fashion-MNIST
    ]
    
    all_fields = {
        'paperId', 'title', 'abstract', 'authors', 'venue', 'year', 
        'publicationDate', 'citationCount', 'referenceCount', 'fieldsOfStudy',
        'url', 'externalIds', 'embedding.specter_v2'
    }
    
    field_availability = {}
    
    for paper_id in test_papers:
        time.sleep(1)  # Rate limiting
        
        url = f"{base_url}/paper/{paper_id}"
        params = {'fields': ','.join(all_fields)}
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                paper = response.json()
                
                print(f"\nPaper: {paper.get('title', 'Unknown')[:50]}...")
                
                for field in all_fields:
                    if '.' in field:  # Handle nested fields like embedding.specter_v2
                        parts = field.split('.')
                        value = paper.get(parts[0], {})
                        if isinstance(value, dict):
                            value = value.get(parts[1])
                    else:
                        value = paper.get(field)
                    
                    has_value = value is not None and value != [] and value != ''
                    field_availability[field] = field_availability.get(field, 0) + (1 if has_value else 0)
                    
                    print(f"  {field}: {'✓' if has_value else '✗'}")
                    
        except Exception as e:
            print(f"Error with paper {paper_id}: {e}")
    
    print(f"\n=== Field Availability Summary ===")
    total_papers = len(test_papers)
    for field, count in sorted(field_availability.items()):
        percentage = (count / total_papers) * 100
        print(f"{field}: {count}/{total_papers} ({percentage:.1f}%)")

if __name__ == "__main__":
    # Create data directory
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    print("Starting simple API test...")
    test_basic_api()
    analyze_data_availability()
    print("\nTest complete!")