"""
Temporal Trend Analysis for Citation Patterns
Analyze temporal dynamics of paper impact using existing citation data
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemporalCitationAnalyzer:
    """
    Analyze temporal patterns in citation data for early virality prediction
    Uses existing citation and publication data to identify temporal trends
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.temporal_features = []
        
    def load_papers_data(self, data_path=None):
        """Load papers dataset for temporal analysis"""
        if data_path is None:
            # Find the latest dataset
            possible_paths = [
                "data/datasets/cs_papers_arxiv_50k.json",  # NEW: ArXiv academic dataset
                "data/datasets/openalex_5000_papers.json",  # OLD: Problematic dataset
                "data/openalex_extract/openalex_cs_papers_*.json"
            ]
            
            import glob
            files = []
            for path_pattern in possible_paths:
                if "*" in path_pattern:
                    files.extend(glob.glob(path_pattern))
                else:
                    if Path(path_pattern).exists():
                        files.append(path_pattern)
            
            if not files:
                raise FileNotFoundError(f"No dataset found. Checked: {possible_paths}")
            
            data_path = max(files, key=lambda x: Path(x).stat().st_mtime)
        
        logger.info(f"Loading temporal analysis data from: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', [])
        df = pd.json_normalize(papers)
        
        logger.info(f"Loaded {len(df)} papers for temporal analysis")
        return df
    
    def extract_temporal_features(self, df):
        """Extract comprehensive temporal features from paper data"""
        logger.info("Extracting temporal features...")
        
        temporal_features_list = []
        
        for idx, (_, paper) in enumerate(df.iterrows()):
            features = self.calculate_temporal_metrics(paper, idx)
            temporal_features_list.append(features)
        
        temporal_df = pd.DataFrame(temporal_features_list)
        
        logger.info(f"Temporal features extracted: {temporal_df.shape[1]} features")
        return temporal_df
    
    def calculate_temporal_metrics(self, paper, paper_idx):
        """Calculate temporal metrics for a single paper - NO DATA LEAKAGE"""
        # Get publication year and current year
        pub_year = paper.get('year', 2020)
        current_year = datetime.now().year
        
        # REMOVED: citation_count usage - CAUSES DATA LEAKAGE
        # citation_count = paper.get('citation_count', 0)
        
        # Calculate paper age (legitimate temporal feature)
        paper_age_years = max(0, current_year - pub_year)
        paper_age_months = paper_age_years * 12
        
        # REMOVED: All citation-based velocity/acceleration features - DATA LEAKAGE
        
        # Determine publication era (legitimate)
        publication_era = self.categorize_publication_era(pub_year)
        
        # Calculate seasonal features (legitimate - available at publication)
        publication_season = self.get_publication_season(paper)
        
        # Calculate conference cycle features (legitimate)
        conference_timing = self.analyze_conference_timing(paper, pub_year)
        
        # REMOVED: Citation percentile and impact features - DATA LEAKAGE
        
        features = {
            # LEGITIMATE TEMPORAL FEATURES (no citation data leakage)
            
            # Basic temporal metrics (available at publication time)
            'temporal_paper_age_years': paper_age_years,
            'temporal_paper_age_months': paper_age_months,
            'temporal_publication_year': pub_year,
            'temporal_years_since_2015': max(0, pub_year - 2015),
            'temporal_years_since_2020': max(0, pub_year - 2020),
            
            # Era and trend features (legitimate)
            'temporal_pre_covid_era': 1 if pub_year < 2020 else 0,
            'temporal_covid_era': 1 if 2020 <= pub_year <= 2022 else 0,
            'temporal_post_covid_era': 1 if pub_year > 2022 else 0,
            'temporal_recent_paper': 1 if paper_age_years <= 2 else 0,
            'temporal_mature_paper': 1 if paper_age_years >= 5 else 0,
            
            # Publication timing (available at publication)
            'temporal_publication_season': publication_season,
            'temporal_early_year_pub': 1 if publication_season in ['winter', 'spring'] else 0,
            'temporal_late_year_pub': 1 if publication_season in ['summer', 'fall'] else 0,
            
            # Conference cycle features (legitimate)
            'temporal_conference_season': conference_timing['season'],
            'temporal_major_conference_period': conference_timing['major_period'],
            'temporal_submission_deadline_proximity': conference_timing['deadline_proximity'],
            
            # Comparative age features (legitimate)
            'temporal_above_median_age': 1 if paper_age_years > 4 else 0,  # Median CS paper age
            'temporal_very_recent': 1 if paper_age_years <= 1 else 0,
            'temporal_established': 1 if paper_age_years >= 3 else 0,
            
            # REMOVED ALL CITATION-BASED FEATURES:
            # - temporal_citation_velocity (citation_count / age)
            # - temporal_citation_acceleration (citation_count / age^2)  
            # - temporal_citations_per_month (citation_count / months)
            # - temporal_citation_density (citation_count / age)
            # - temporal_citation_intensity (citation_count / age)
            # - temporal_normalized_impact (citation_count / sqrt(age))
            # - temporal_citation_vs_age_ratio (citation_count / age)
            # - temporal_impact_efficiency (citation_count / months)
            # - temporal_early_impact (based on citation_count)
            # - temporal_sustained_impact (based on citation_count)
            # - All impact timing features using citation_count
            
            # Paper index for tracking
            'paper_index': paper_idx
        }
        
        return features
    
    def categorize_publication_era(self, pub_year):
        """Categorize publication into research eras"""
        if pub_year < 2015:
            return 'pre_deep_learning'
        elif pub_year < 2020:
            return 'deep_learning_era'
        elif pub_year < 2023:
            return 'pandemic_era'
        else:
            return 'post_pandemic'
    
    def get_publication_season(self, paper):
        """Determine publication season (simplified)"""
        # This is simplified - in real implementation you'd parse publication dates
        # For now, use a simple heuristic based on paper ID or other available data
        pub_year = paper.get('year', 2020)
        
        # Simple seasonal assignment based on year modulo
        seasons = ['winter', 'spring', 'summer', 'fall']
        season_idx = (pub_year + hash(str(paper.get('title', ''))) % 4) % 4
        return seasons[season_idx]
    
    def analyze_conference_timing(self, paper, pub_year):
        """Analyze conference publication timing patterns"""
        # Major CS conferences typically have deadlines and publications in cycles
        
        # Simplified conference cycle analysis
        major_conference_months = [3, 6, 9, 12]  # March, June, September, December
        month_proximity = min([abs(pub_year % 12 - month) for month in major_conference_months])
        
        return {
            'season': 'conference' if month_proximity <= 1 else 'off_season',
            'major_period': 1 if month_proximity <= 1 else 0,
            'deadline_proximity': max(0, 3 - month_proximity)  # Closer to deadline = higher value
        }
    
    def calculate_citation_percentile(self, citation_count, paper_age):
        """Calculate citation percentile for paper age"""
        # Simplified percentile calculation
        # In real implementation, this would use historical data distributions
        
        expected_citations = paper_age * 2  # Simple heuristic: 2 citations per year
        
        if citation_count >= expected_citations * 3:
            return 90
        elif citation_count >= expected_citations * 2:
            return 75
        elif citation_count >= expected_citations:
            return 50
        elif citation_count >= expected_citations * 0.5:
            return 25
        else:
            return 10
    
    def calculate_impact_timing_features(self, paper, citation_count, paper_age):
        """Calculate features related to impact timing patterns"""
        
        # Early impact: high citations relative to age
        early_impact = 1 if (paper_age <= 2 and citation_count >= 5) else 0
        
        # Sustained impact: consistent citations over time
        sustained_impact = 1 if (paper_age >= 3 and citation_count >= paper_age * 2) else 0
        
        # Late bloomer: older paper with decent citations
        late_bloomer = 1 if (paper_age >= 5 and citation_count >= 10) else 0
        
        # Rapid growth: high citation velocity
        rapid_growth = 1 if (citation_count / max(paper_age, 1) >= 10) else 0
        
        return {
            'early_impact': early_impact,
            'sustained_impact': sustained_impact,
            'late_bloomer': late_bloomer,
            'rapid_growth': rapid_growth
        }
    
    def analyze_citation_patterns(self, temporal_df):
        """Analyze overall citation patterns and trends"""
        logger.info("Analyzing citation patterns...")
        
        # Correlation analysis
        correlation_features = [
            'temporal_paper_age_years', 'temporal_citation_velocity', 
            'temporal_citation_acceleration', 'temporal_citation_density'
        ]
        
        correlations = temporal_df[correlation_features].corr()
        
        # Cluster analysis of temporal patterns
        cluster_features = [
            'temporal_citation_velocity', 'temporal_citation_acceleration',
            'temporal_citation_density', 'temporal_paper_age_years'
        ]
        
        X_cluster = temporal_df[cluster_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # K-means clustering to identify temporal patterns
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        analysis_results = {
            'correlations': correlations.to_dict(),
            'clusters': clusters.tolist(),
            'pca_components': X_pca.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'explained_variance': pca.explained_variance_ratio_.tolist()
        }
        
        # Analyze cluster characteristics
        temporal_df['temporal_cluster'] = clusters
        cluster_summary = {}
        
        for cluster_id in range(4):
            cluster_data = temporal_df[temporal_df['temporal_cluster'] == cluster_id]
            cluster_summary[f'cluster_{cluster_id}'] = {
                'count': len(cluster_data),
                'avg_citation_velocity': cluster_data['temporal_citation_velocity'].mean(),
                'avg_paper_age': cluster_data['temporal_paper_age_years'].mean(),
                'avg_citation_density': cluster_data['temporal_citation_density'].mean(),
                'characteristics': self.characterize_cluster(cluster_data)
            }
        
        analysis_results['cluster_summary'] = cluster_summary
        
        logger.info(f"Temporal pattern analysis complete:")
        for cluster_id, summary in cluster_summary.items():
            logger.info(f"  {cluster_id}: {summary['count']} papers - {summary['characteristics']}")
        
        return analysis_results
    
    def characterize_cluster(self, cluster_data):
        """Characterize a temporal cluster"""
        avg_velocity = cluster_data['temporal_citation_velocity'].mean()
        avg_age = cluster_data['temporal_paper_age_years'].mean()
        avg_density = cluster_data['temporal_citation_density'].mean()
        
        if avg_velocity > 5 and avg_age < 3:
            return "Fast Rising (High velocity, young papers)"
        elif avg_velocity > 2 and avg_age > 5:
            return "Steady Performers (Sustained impact over time)"
        elif avg_velocity < 1 and avg_age > 3:
            return "Slow Burners (Low velocity, mature papers)"
        else:
            return "Mixed Patterns (Diverse temporal characteristics)"
    
    def calculate_temporal_trends(self, temporal_df):
        """Calculate overall temporal trends in the dataset"""
        logger.info("Calculating temporal trends...")
        
        # Year-over-year trends
        yearly_trends = temporal_df.groupby('temporal_publication_year').agg({
            'temporal_citation_velocity': ['mean', 'median', 'std'],
            'temporal_citation_density': ['mean', 'median'],
            'temporal_early_impact': 'mean',
            'temporal_rapid_growth': 'mean'
        }).round(3)
        
        # Era-based trends
        era_columns = ['temporal_pre_covid_era', 'temporal_covid_era', 'temporal_post_covid_era']
        era_trends = {}
        
        for era_col in era_columns:
            era_data = temporal_df[temporal_df[era_col] == 1]
            if len(era_data) > 0:
                era_name = era_col.replace('temporal_', '').replace('_era', '')
                era_trends[era_name] = {
                    'paper_count': len(era_data),
                    'avg_citation_velocity': era_data['temporal_citation_velocity'].mean(),
                    'avg_citation_density': era_data['temporal_citation_density'].mean(),
                    'early_impact_rate': era_data['temporal_early_impact'].mean(),
                    'rapid_growth_rate': era_data['temporal_rapid_growth'].mean()
                }
        
        # Seasonal trends
        seasonal_trends = temporal_df.groupby('temporal_publication_season').agg({
            'temporal_citation_velocity': 'mean',
            'temporal_early_impact': 'mean',
            'temporal_rapid_growth': 'mean'
        }).round(3)
        
        trends_summary = {
            'yearly_trends': self.convert_nested_dict_keys(yearly_trends.to_dict()),
            'era_trends': era_trends,
            'seasonal_trends': self.convert_nested_dict_keys(seasonal_trends.to_dict())
        }
        
        logger.info("Temporal trends calculated:")
        logger.info(f"  Era trends: {len(era_trends)} eras analyzed")
        logger.info(f"  Seasonal trends: {len(seasonal_trends)} seasons analyzed")
        
        return trends_summary
    
    def create_temporal_predictions(self, temporal_df):
        """Create temporal-based predictions for early impact"""
        logger.info("Creating temporal-based predictions...")
        
        # Define early impact prediction features
        prediction_features = [
            'temporal_citation_velocity',
            'temporal_citation_acceleration', 
            'temporal_citation_density',
            'temporal_early_impact',
            'temporal_rapid_growth',
            'temporal_recent_paper',
            'temporal_major_conference_period'
        ]
        
        # Calculate composite temporal score
        weights = {
            'temporal_citation_velocity': 0.3,
            'temporal_citation_acceleration': 0.2,
            'temporal_citation_density': 0.2,
            'temporal_early_impact': 0.15,
            'temporal_rapid_growth': 0.1,
            'temporal_major_conference_period': 0.05
        }
        
        temporal_scores = []
        for _, row in temporal_df.iterrows():
            score = sum(row[feature] * weight for feature, weight in weights.items())
            temporal_scores.append(score)
        
        temporal_df['temporal_composite_score'] = temporal_scores
        
        # Classify papers based on temporal patterns
        score_threshold_high = np.percentile(temporal_scores, 80)
        score_threshold_medium = np.percentile(temporal_scores, 50)
        
        temporal_predictions = []
        for score in temporal_scores:
            if score >= score_threshold_high:
                temporal_predictions.append('high_temporal_impact')
            elif score >= score_threshold_medium:
                temporal_predictions.append('medium_temporal_impact')
            else:
                temporal_predictions.append('low_temporal_impact')
        
        temporal_df['temporal_impact_prediction'] = temporal_predictions
        
        # Calculate prediction accuracy (if we have ground truth)
        ground_truth_available = 'citation_count' in temporal_df.columns
        if ground_truth_available:
            # Create ground truth labels based on actual citations
            actual_high_impact = temporal_df['citation_count'] >= temporal_df['citation_count'].quantile(0.8)
            predicted_high_impact = temporal_df['temporal_impact_prediction'] == 'high_temporal_impact'
            
            # Calculate accuracy metrics
            accuracy = (actual_high_impact == predicted_high_impact).mean()
            precision = (actual_high_impact & predicted_high_impact).sum() / predicted_high_impact.sum() if predicted_high_impact.sum() > 0 else 0
            recall = (actual_high_impact & predicted_high_impact).sum() / actual_high_impact.sum() if actual_high_impact.sum() > 0 else 0
            
            prediction_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            }
        else:
            prediction_metrics = {'note': 'Ground truth not available for validation'}
        
        logger.info(f"Temporal predictions created:")
        if ground_truth_available:
            logger.info(f"  Accuracy: {prediction_metrics['accuracy']:.3f}")
            logger.info(f"  Precision: {prediction_metrics['precision']:.3f}")
            logger.info(f"  Recall: {prediction_metrics['recall']:.3f}")
        
        return temporal_df, prediction_metrics
    
    def save_temporal_analysis_results(self, temporal_df, pattern_analysis, trends_summary, prediction_metrics):
        """Save comprehensive temporal analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/temporal_analysis")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_papers': len(temporal_df),
                'temporal_features_count': len([col for col in temporal_df.columns if col.startswith('temporal_')]),
                'analysis_type': 'comprehensive_temporal_analysis'
            },
            'pattern_analysis': pattern_analysis,
            'trends_summary': trends_summary,
            'prediction_metrics': prediction_metrics,
            'feature_statistics': self.calculate_feature_statistics(temporal_df)
        }
        
        results_file = results_dir / f"temporal_analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save temporal features DataFrame
        features_file = results_dir / f"temporal_features_{timestamp}.csv"
        temporal_df.to_csv(features_file, index=False)
        
        logger.info(f"Temporal analysis results saved to: {results_file}")
        logger.info(f"Temporal features saved to: {features_file}")
        
        return results_file, features_file
    
    def convert_nested_dict_keys(self, nested_dict):
        """Convert nested dictionary keys to strings for JSON serialization"""
        if isinstance(nested_dict, dict):
            return {str(k): self.convert_nested_dict_keys(v) for k, v in nested_dict.items()}
        else:
            return nested_dict
    
    def calculate_feature_statistics(self, temporal_df):
        """Calculate statistics for temporal features"""
        temporal_columns = [col for col in temporal_df.columns if col.startswith('temporal_')]
        
        statistics = {}
        for col in temporal_columns:
            if temporal_df[col].dtype in ['int64', 'float64']:
                statistics[col] = {
                    'mean': float(temporal_df[col].mean()),
                    'median': float(temporal_df[col].median()),
                    'std': float(temporal_df[col].std()),
                    'min': float(temporal_df[col].min()),
                    'max': float(temporal_df[col].max()),
                    'non_zero_count': int((temporal_df[col] != 0).sum())
                }
        
        return statistics

def main():
    """Main execution for temporal analysis"""
    logger.info("Starting Comprehensive Temporal Citation Analysis...")
    
    try:
        # Initialize analyzer
        analyzer = TemporalCitationAnalyzer()
        
        # Load data
        df = analyzer.load_papers_data()
        
        # Extract temporal features
        temporal_df = analyzer.extract_temporal_features(df)
        
        # Analyze citation patterns
        pattern_analysis = analyzer.analyze_citation_patterns(temporal_df)
        
        # Calculate temporal trends
        trends_summary = analyzer.calculate_temporal_trends(temporal_df)
        
        # Create temporal predictions
        temporal_df, prediction_metrics = analyzer.create_temporal_predictions(temporal_df)
        
        # Save results
        results_file, features_file = analyzer.save_temporal_analysis_results(
            temporal_df, pattern_analysis, trends_summary, prediction_metrics
        )
        
        # Print summary
        print("\n" + "="*80)
        print("🕒 TEMPORAL CITATION ANALYSIS - RESULTS")
        print("="*80)
        print(f"Dataset: {len(temporal_df)} papers analyzed")
        print(f"Temporal Features: {len([col for col in temporal_df.columns if col.startswith('temporal_')])} features")
        print(f"Temporal Clusters: {len(pattern_analysis['cluster_summary'])} patterns identified")
        
        if 'accuracy' in prediction_metrics:
            print(f"\nTemporal Prediction Performance:")
            print(f"  Accuracy: {prediction_metrics['accuracy']:.3f}")
            print(f"  Precision: {prediction_metrics['precision']:.3f}")
            print(f"  Recall: {prediction_metrics['recall']:.3f}")
            print(f"  F1-Score: {prediction_metrics['f1_score']:.3f}")
        
        print(f"\nTemporal Pattern Clusters:")
        for cluster_id, summary in pattern_analysis['cluster_summary'].items():
            print(f"  {cluster_id}: {summary['count']} papers - {summary['characteristics']}")
        
        print(f"\n✅ TEMPORAL ANALYSIS COMPLETE!")
        print(f"   • Results: {results_file}")
        print(f"   • Features: {features_file}")
        print(f"\n🕒 Temporal dynamics analyzed for enhanced early prediction!")
        
    except Exception as e:
        logger.error(f"Error in temporal analysis: {e}")
        raise

if __name__ == "__main__":
    main()