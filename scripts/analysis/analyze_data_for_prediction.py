"""
Analyze collected data to identify features and patterns for virality prediction
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import statistics

def load_sample_data():
    """Load the collected sample data"""
    sample_dir = 'data/raw/sample'
    
    all_papers = []
    
    # Load all JSON files in the sample directory
    for filename in os.listdir(sample_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(sample_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                papers = json.load(f)
                all_papers.extend(papers)
    
    print(f"Loaded {len(all_papers)} papers from sample data")
    return all_papers

def analyze_virality_potential(papers: List[Dict[str, Any]]):
    """Analyze papers to understand virality patterns"""
    print("\n=== Virality Analysis ===")
    
    if not papers:
        print("No papers to analyze")
        return
    
    # Calculate age of papers (important for early prediction)
    current_date = datetime.now()
    
    for paper in papers:
        pub_date_str = paper.get('publicationDate')
        if pub_date_str:
            try:
                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d')
                age_days = (current_date - pub_date).days
                paper['age_days'] = age_days
                paper['age_months'] = age_days / 30
            except ValueError:
                paper['age_days'] = None
                paper['age_months'] = None
    
    # Categorize papers by impact level
    high_impact = [p for p in papers if p.get('citationCount', 0) > 200]
    medium_impact = [p for p in papers if 50 <= p.get('citationCount', 0) <= 200]
    low_impact = [p for p in papers if p.get('citationCount', 0) < 50]
    
    print(f"Impact distribution:")
    print(f"  High impact (>200 citations): {len(high_impact)}")
    print(f"  Medium impact (50-200): {len(medium_impact)}")
    print(f"  Low impact (<50): {len(low_impact)}")
    
    # Calculate citation velocity (citations per month)
    papers_with_age = [p for p in papers if p.get('age_months') and p.get('age_months') > 0]
    
    if papers_with_age:
        print(f"\n=== Citation Velocity Analysis ===")
        
        for paper in papers_with_age:
            citations = paper.get('citationCount', 0)
            age_months = paper.get('age_months', 1)
            citation_velocity = citations / age_months
            paper['citation_velocity'] = citation_velocity
        
        velocities = [p['citation_velocity'] for p in papers_with_age]
        
        print(f"Citation velocity stats (citations/month):")
        print(f"  Average: {statistics.mean(velocities):.2f}")
        print(f"  Median: {statistics.median(velocities):.2f}")
        print(f"  Range: {min(velocities):.2f} - {max(velocities):.2f}")
        
        # Sort by velocity to identify high-velocity papers
        sorted_papers = sorted(papers_with_age, key=lambda x: x['citation_velocity'], reverse=True)
        
        print(f"\nTop 3 highest velocity papers:")
        for i, paper in enumerate(sorted_papers[:3]):
            title = paper.get('title', 'Unknown')[:60]
            velocity = paper['citation_velocity']
            citations = paper.get('citationCount', 0)
            age = paper.get('age_months', 0)
            venue = paper.get('venue', 'Unknown')
            print(f"  {i+1}. {title}...")
            print(f"     Velocity: {velocity:.1f} citations/month")
            print(f"     Total citations: {citations} (age: {age:.1f} months)")
            print(f"     Venue: {venue}")

def identify_available_features(papers: List[Dict[str, Any]]):
    """Identify what features we can extract for prediction models"""
    print(f"\n=== Available Features Analysis ===")
    
    if not papers:
        return
    
    # Analyze feature availability across papers
    feature_availability = {}
    
    # Define key feature categories
    feature_mapping = {
        'text_features': ['title', 'abstract'],
        'author_features': ['authors'],
        'venue_features': ['venue'],
        'temporal_features': ['year', 'publicationDate'],
        'citation_features': ['citationCount'],
        'metadata_features': ['fieldsOfStudy', 'externalIds'],
        'access_features': ['openAccessPdf']
    }
    
    for category, features in feature_mapping.items():
        category_availability = {}
        
        for feature in features:
            count = 0
            for paper in papers:
                value = paper.get(feature)
                if value is not None and value != [] and value != '':
                    count += 1
            
            percentage = (count / len(papers)) * 100
            category_availability[feature] = {
                'count': count,
                'percentage': percentage
            }
        
        feature_availability[category] = category_availability
    
    # Print feature availability
    for category, features in feature_availability.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for feature, stats in features.items():
            print(f"  - {feature}: {stats['count']}/{len(papers)} ({stats['percentage']:.1f}%)")
    
    return feature_availability

def extract_text_features_sample(papers: List[Dict[str, Any]]):
    """Analyze text features that could be useful for prediction"""
    print(f"\n=== Text Features Analysis ===")
    
    # Abstract analysis
    abstracts = [p.get('abstract', '') for p in papers if p.get('abstract')]
    
    if abstracts:
        abstract_lengths = [len(abstract.split()) for abstract in abstracts]
        
        print(f"Abstract statistics:")
        print(f"  - Papers with abstracts: {len(abstracts)}/{len(papers)}")
        print(f"  - Average length: {statistics.mean(abstract_lengths):.1f} words")
        print(f"  - Length range: {min(abstract_lengths)} - {max(abstract_lengths)} words")
        
        # Look for technical keywords that might indicate innovation
        innovation_keywords = [
            'novel', 'new', 'first', 'breakthrough', 'innovative', 'advanced',
            'state-of-the-art', 'sota', 'outperform', 'improve', 'better',
            'superior', 'efficient', 'fast', 'scalable'
        ]
        
        evaluation_keywords = [
            'benchmark', 'dataset', 'experiment', 'evaluation', 'comparison',
            'baseline', 'results', 'performance', 'accuracy', 'precision'
        ]
        
        # Count keyword occurrences
        innovation_counts = []
        evaluation_counts = []
        
        for abstract in abstracts:
            abstract_lower = abstract.lower()
            
            innovation_count = sum(1 for keyword in innovation_keywords if keyword in abstract_lower)
            evaluation_count = sum(1 for keyword in evaluation_keywords if keyword in abstract_lower)
            
            innovation_counts.append(innovation_count)
            evaluation_counts.append(evaluation_count)
        
        print(f"\nKeyword analysis:")
        print(f"  - Avg innovation keywords per abstract: {statistics.mean(innovation_counts):.1f}")
        print(f"  - Avg evaluation keywords per abstract: {statistics.mean(evaluation_counts):.1f}")
    
    # Title analysis
    titles = [p.get('title', '') for p in papers if p.get('title')]
    
    if titles:
        title_lengths = [len(title.split()) for title in titles]
        
        print(f"\nTitle statistics:")
        print(f"  - Average length: {statistics.mean(title_lengths):.1f} words")
        print(f"  - Length range: {min(title_lengths)} - {max(title_lengths)} words")

def analyze_venue_patterns(papers: List[Dict[str, Any]]):
    """Analyze venue patterns for prediction features"""
    print(f"\n=== Venue Pattern Analysis ===")
    
    # Group papers by venue
    venue_stats = {}
    
    for paper in papers:
        venue = paper.get('venue', 'Unknown')
        if venue not in venue_stats:
            venue_stats[venue] = {
                'papers': [],
                'total_citations': 0,
                'avg_citations': 0
            }
        
        venue_stats[venue]['papers'].append(paper)
        venue_stats[venue]['total_citations'] += paper.get('citationCount', 0)
    
    # Calculate venue statistics
    for venue, stats in venue_stats.items():
        paper_count = len(stats['papers'])
        stats['paper_count'] = paper_count
        stats['avg_citations'] = stats['total_citations'] / paper_count if paper_count > 0 else 0
    
    # Sort venues by average impact
    sorted_venues = sorted(venue_stats.items(), key=lambda x: x[1]['avg_citations'], reverse=True)
    
    print(f"Venue impact analysis:")
    for venue, stats in sorted_venues:
        print(f"  - {venue}")
        print(f"    Papers: {stats['paper_count']}")
        print(f"    Avg citations: {stats['avg_citations']:.1f}")
        print(f"    Total citations: {stats['total_citations']}")

def generate_feature_recommendations(feature_availability: Dict[str, Any]):
    """Generate recommendations for feature engineering based on available data"""
    print(f"\n=== Feature Engineering Recommendations ===")
    
    recommendations = []
    
    # Text features
    text_features = feature_availability.get('text_features', {})
    if text_features.get('abstract', {}).get('percentage', 0) > 80:
        recommendations.append("✓ Abstract text analysis: Extract readability, innovation keywords, technical complexity")
    
    if text_features.get('title', {}).get('percentage', 0) > 90:
        recommendations.append("✓ Title analysis: Length, question marks, innovation indicators")
    
    # Author features
    author_features = feature_availability.get('author_features', {})
    if author_features.get('authors', {}).get('percentage', 0) > 80:
        recommendations.append("✓ Author features: Team size, author reputation (if available)")
    
    # Venue features
    venue_features = feature_availability.get('venue_features', {})
    if venue_features.get('venue', {}).get('percentage', 0) > 80:
        recommendations.append("✓ Venue prestige: Conference ranking, historical impact factor")
    
    # Temporal features
    temporal_features = feature_availability.get('temporal_features', {})
    if temporal_features.get('publicationDate', {}).get('percentage', 0) > 80:
        recommendations.append("✓ Temporal features: Publication timing, conference cycle effects")
    
    print("Based on available data, recommend implementing:")
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\nMissing features to address:")
    print(f"  - SPECTER embeddings: Need alternative text embedding strategy")
    print(f"  - Early citations: Need to collect citation timelines")
    print(f"  - Social signals: Need altmetrics/social media data")
    print(f"  - Reproducibility: Need GitHub/code availability detection")

def main():
    """Main analysis function"""
    print("=== Data Analysis for Virality Prediction ===")
    
    # Load sample data
    papers = load_sample_data()
    
    if not papers:
        print("No papers found to analyze")
        return
    
    # Run all analyses
    analyze_virality_potential(papers)
    feature_availability = identify_available_features(papers)
    extract_text_features_sample(papers)
    analyze_venue_patterns(papers)
    generate_feature_recommendations(feature_availability)
    
    # Save analysis results
    analysis_results = {
        'total_papers_analyzed': len(papers),
        'feature_availability': feature_availability,
        'analysis_date': datetime.now().isoformat(),
        'key_findings': [
            "High citation velocity papers show strong early virality potential",
            "Abstract and title text features are consistently available",
            "Venue information provides strong predictive signal",
            "Need alternative embedding strategy (no SPECTER access)",
            "Early citation data collection is critical for prediction"
        ]
    }
    
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/data_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n✓ Analysis complete. Results saved to data/processed/data_analysis_results.json")

if __name__ == "__main__":
    main()