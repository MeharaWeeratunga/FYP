"""
Quick summary of data collection progress and quality
"""

import json
import glob
from pathlib import Path
from datetime import datetime

def analyze_collection_progress():
    """Analyze current collection progress"""
    print("📊 DATA COLLECTION PROGRESS SUMMARY")
    print("=" * 50)
    
    # Find all collection files
    collection_files = glob.glob("data/raw/systematic_collection/*.json")
    
    if not collection_files:
        print("❌ No collection files found")
        return
    
    all_papers = []
    venue_counts = {}
    
    # Load and analyze each file
    for file_path in collection_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            filename = Path(file_path).name
            venue_year = filename.replace('.json', '').replace('_20250619_', ' ')
            
            print(f"\n📄 {venue_year}: {len(papers)} papers")
            
            # Add to totals
            all_papers.extend(papers)
            
            # Count by venue
            for paper in papers:
                venue = paper.get('venue', 'Unknown')
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
            
            # Show sample papers
            if papers:
                print("   Sample papers:")
                for i, paper in enumerate(papers[:2]):
                    title = paper.get('title', 'No title')[:60]
                    citations = paper.get('citation_count', 0)
                    is_cs = paper.get('is_cs_paper', False)
                    cs_indicator = "✅" if is_cs else "❌"
                    print(f"   {i+1}. {title}... ({citations} cites) {cs_indicator}")
                    
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    
    # Overall summary
    print(f"\n🎯 OVERALL SUMMARY")
    print(f"Total papers collected: {len(all_papers)}")
    
    # CS classification accuracy
    cs_papers = sum(1 for p in all_papers if p.get('is_cs_paper', False))
    print(f"Papers marked as CS: {cs_papers}/{len(all_papers)} ({cs_papers/len(all_papers)*100:.1f}%)")
    
    # Citation statistics
    citations = [p.get('citation_count', 0) for p in all_papers]
    if citations:
        print(f"Citation stats: avg={sum(citations)/len(citations):.1f}, range={min(citations)}-{max(citations)}")
    
    # Top venues
    print(f"\n🏆 VENUE DISTRIBUTION:")
    sorted_venues = sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)
    for venue, count in sorted_venues[:10]:
        print(f"  - {venue}: {count} papers")
    
    # Check for quality issues
    print(f"\n🔍 QUALITY CHECK:")
    
    # Abstract completeness
    with_abstract = sum(1 for p in all_papers if p.get('abstract'))
    print(f"Papers with abstracts: {with_abstract}/{len(all_papers)} ({with_abstract/len(all_papers)*100:.1f}%)")
    
    # Year distribution
    years = {}
    for paper in all_papers:
        year = paper.get('year', 'Unknown')
        years[year] = years.get(year, 0) + 1
    
    print(f"Year distribution:")
    for year, count in sorted(years.items()):
        print(f"  - {year}: {count} papers")
    
    # Identify potential misclassifications
    print(f"\n⚠️  POTENTIAL CLASSIFICATION ISSUES:")
    non_cs_indicators = ['tobacco', 'medical', 'health', 'biology', 'chemistry', 'physics']
    
    suspicious_papers = []
    for paper in all_papers:
        if paper.get('is_cs_paper', False):
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            
            if any(indicator in title or indicator in abstract for indicator in non_cs_indicators):
                suspicious_papers.append(paper)
    
    if suspicious_papers:
        print(f"Found {len(suspicious_papers)} potentially misclassified papers:")
        for paper in suspicious_papers[:3]:
            title = paper.get('title', 'No title')[:60]
            venue = paper.get('venue', 'No venue')
            print(f"  - {title}... (from {venue})")
    else:
        print("No obvious misclassifications detected")
    
    # Progress toward target
    target_total = 400
    progress_pct = (len(all_papers) / target_total) * 100
    print(f"\n📈 PROGRESS TO TARGET:")
    print(f"{len(all_papers)}/{target_total} papers ({progress_pct:.1f}%)")
    
    remaining = target_total - len(all_papers)
    print(f"Remaining to collect: {remaining} papers")
    
    if len(all_papers) >= 50:
        print("✅ Good foundation dataset collected!")
        print("💡 Consider proceeding with feature engineering while continuing collection")
    elif len(all_papers) >= 20:
        print("⚠️  Small but usable dataset for initial development")
    else:
        print("📈 Continue collection to build substantial dataset")

if __name__ == "__main__":
    analyze_collection_progress()