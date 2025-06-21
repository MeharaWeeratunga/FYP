"""
Data Leakage & Overfitting Debugging Script
Comprehensive analysis to identify sources of perfect performance
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLeakageDetector:
    """Comprehensive data leakage and overfitting detection"""
    
    def __init__(self):
        self.results = {}
        
    def load_latest_results(self):
        """Load the latest results and dataset"""
        logger.info("Loading latest results and dataset...")
        
        # Load OpenAlex dataset
        possible_paths = [
            Path("data/datasets/openalex_5000_papers.json"),
            Path("data/openalex_extract/").glob("openalex_cs_papers_*.json")
        ]
        
        extract_files = []
        for path in possible_paths:
            if isinstance(path, Path) and path.exists():
                extract_files.append(path)
            else:
                extract_files.extend(list(path))
        
        if not extract_files:
            raise FileNotFoundError("No OpenAlex dataset found")
        
        latest_dataset = max(extract_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_dataset, 'r') as f:
            dataset = json.load(f)
        
        self.df = pd.json_normalize(dataset['papers'])
        logger.info(f"Loaded dataset: {len(self.df)} papers")
        
        # Load latest results
        result_files = list(Path("results/final/comprehensive_enhanced/").glob("comprehensive_enhanced_results_*.json"))
        if result_files:
            latest_results = max(result_files, key=lambda x: x.stat().st_mtime)
            with open(latest_results, 'r') as f:
                self.results = json.load(f)
            logger.info(f"Loaded results from: {latest_results}")
        
        return self.df
    
    def check_basic_statistics(self):
        """Check basic dataset statistics for anomalies"""
        logger.info("Checking basic dataset statistics...")
        
        print("\n" + "="*60)
        print("📊 BASIC DATASET STATISTICS")
        print("="*60)
        
        # Dataset shape
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Citation count distribution
        citations = self.df['citation_count'].fillna(0)
        print(f"\nCitation Count Distribution:")
        print(f"  Mean: {citations.mean():.2f}")
        print(f"  Median: {citations.median():.2f}")
        print(f"  Std: {citations.std():.2f}")
        print(f"  Min: {citations.min()}")
        print(f"  Max: {citations.max()}")
        print(f"  Zero citations: {(citations == 0).sum()}/{len(citations)} ({(citations == 0).mean():.1%})")
        
        # Check for suspicious patterns
        print(f"\nSuspicious Patterns:")
        print(f"  Identical citation counts: {citations.value_counts().max()}")
        print(f"  Most common citation count: {citations.mode().iloc[0] if len(citations.mode()) > 0 else 'N/A'}")
        
        # Year distribution
        years = self.df['year'].fillna(0)
        print(f"\nYear Distribution:")
        print(f"  Range: {years.min()}-{years.max()}")
        print(f"  Most common year: {years.mode().iloc[0] if len(years.mode()) > 0 else 'N/A'}")
        
        # Quality score distribution
        if 'quality_score' in self.df.columns:
            quality = self.df['quality_score']
            print(f"\nQuality Score Distribution:")
            print(f"  Mean: {quality.mean():.2f}")
            print(f"  Unique values: {quality.nunique()}")
            print(f"  Most common: {quality.mode().iloc[0] if len(quality.mode()) > 0 else 'N/A'}")
        
        return {
            'citation_stats': citations.describe().to_dict(),
            'zero_citations_pct': (citations == 0).mean(),
            'identical_citations': citations.value_counts().max()
        }
    
    def check_duplicate_papers(self):
        """Check for duplicate papers that could cause leakage"""
        logger.info("Checking for duplicate papers...")
        
        print("\n" + "="*60)
        print("🔍 DUPLICATE PAPER ANALYSIS")
        print("="*60)
        
        # Check for exact duplicates
        original_count = len(self.df)
        
        # Check ID duplicates
        id_duplicates = self.df['id'].duplicated().sum() if 'id' in self.df.columns else 0
        print(f"ID duplicates: {id_duplicates}")
        
        # Check title duplicates
        title_duplicates = self.df['title'].duplicated().sum()
        print(f"Title duplicates: {title_duplicates}")
        
        # Check abstract duplicates
        abstract_duplicates = self.df['abstract'].duplicated().sum()
        print(f"Abstract duplicates: {abstract_duplicates}")
        
        # Check title+abstract combinations
        self.df['title_abstract'] = self.df['title'].fillna('') + ' ' + self.df['abstract'].fillna('')
        content_duplicates = self.df['title_abstract'].duplicated().sum()
        print(f"Title+Abstract duplicates: {content_duplicates}")
        
        # Check for near-duplicates (same title, similar citations)
        if title_duplicates > 0:
            duplicate_titles = self.df[self.df['title'].duplicated(keep=False)]
            print(f"\nDuplicate title examples:")
            for title in duplicate_titles['title'].unique()[:3]:
                subset = duplicate_titles[duplicate_titles['title'] == title]
                citations = subset['citation_count'].tolist()
                print(f"  '{title[:50]}...' - Citations: {citations}")
        
        return {
            'id_duplicates': id_duplicates,
            'title_duplicates': title_duplicates,
            'abstract_duplicates': abstract_duplicates,
            'content_duplicates': content_duplicates
        }
    
    def check_feature_target_correlation(self):
        """Check for suspiciously high correlations between features and targets"""
        logger.info("Checking feature-target correlations...")
        
        print("\n" + "="*60)
        print("🔗 FEATURE-TARGET CORRELATION ANALYSIS")
        print("="*60)
        
        # Recreate basic features to check correlations
        features = {}
        
        # Basic metadata features
        features['citation_count_orig'] = self.df['citation_count'].fillna(0)
        features['year'] = self.df['year'].fillna(2020)
        features['author_count'] = self.df['author_count'].fillna(1)
        features['quality_score'] = self.df.get('quality_score', 50)
        
        # Text features
        features['title_length'] = self.df['title'].str.len().fillna(0)
        features['abstract_length'] = self.df['abstract'].str.len().fillna(0)
        
        # Check if citation_count appears as a feature
        feature_df = pd.DataFrame(features)
        target = self.df['citation_count'].fillna(0)
        
        correlations = feature_df.corrwith(target).abs().sort_values(ascending=False)
        
        print("Feature correlations with citation_count:")
        print(correlations)
        
        # Check for perfect correlations (data leakage indicators)
        perfect_correlations = correlations[correlations > 0.99]
        if len(perfect_correlations) > 0:
            print(f"\n🚨 PERFECT CORRELATIONS DETECTED (>0.99):")
            for feature, corr in perfect_correlations.items():
                print(f"  {feature}: {corr:.6f}")
        
        # Check for very high correlations
        high_correlations = correlations[(correlations > 0.8) & (correlations <= 0.99)]
        if len(high_correlations) > 0:
            print(f"\n⚠️  HIGH CORRELATIONS (>0.8):")
            for feature, corr in high_correlations.items():
                print(f"  {feature}: {corr:.3f}")
        
        return correlations.to_dict()
    
    def simulate_clean_experiment(self):
        """Simulate a clean experiment without potential leakage"""
        logger.info("Running clean experiment simulation...")
        
        print("\n" + "="*60)
        print("🧪 CLEAN EXPERIMENT SIMULATION")
        print("="*60)
        
        # Create clean features (no potential leakage)
        clean_features = pd.DataFrame({
            'year': self.df['year'].fillna(2020),
            'author_count': self.df['author_count'].fillna(1),
            'title_length': self.df['title'].str.len().fillna(0),
            'abstract_length': self.df['abstract'].str.len().fillna(0),
            'has_doi': (self.df['doi'].notna()).astype(int),
            'venue_length': self.df['venue'].str.len().fillna(0),
            'paper_age': 2025 - self.df['year'].fillna(2020)
        })
        
        # Add some text features (basic TF-IDF)
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = (self.df['title'].fillna('') + ' ' + self.df['abstract'].fillna('')).tolist()
        
        tfidf = TfidfVectorizer(max_features=50, stop_words='english', min_df=5)
        try:
            tfidf_features = tfidf.fit_transform(texts)
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                                  columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            clean_features = pd.concat([clean_features, tfidf_df], axis=1)
            print(f"Added {tfidf_features.shape[1]} TF-IDF features")
        except:
            print("Could not add TF-IDF features")
        
        # Target variable
        target = self.df['citation_count'].fillna(0)
        
        # Remove rows with missing targets
        mask = target.notna() & (clean_features.notna().all(axis=1))
        X_clean = clean_features[mask]
        y_clean = target[mask]
        
        print(f"Clean dataset: {len(X_clean)} samples, {X_clean.shape[1]} features")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test simple models
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {'mae': mae, 'r2': r2}
                print(f"{name:15} | MAE: {mae:8.2f} | R²: {r2:6.3f}")
                
            except Exception as e:
                print(f"{name:15} | Error: {str(e)}")
        
        # Compare with previous results
        print(f"\n📊 COMPARISON WITH PREVIOUS RESULTS:")
        if 'performance_results' in self.results:
            prev_results = self.results['performance_results'].get('citation_count', {})
            for model_name, metrics in prev_results.items():
                if model_name in ['LinearRegression', 'RandomForestRegressor']:
                    prev_mae = metrics.get('mae', 'N/A')
                    prev_r2 = metrics.get('r2', 'N/A')
                    print(f"Previous {model_name}: MAE {prev_mae}, R² {prev_r2}")
        
        return results
    
    def check_train_test_contamination(self):
        """Check for train-test contamination"""
        logger.info("Checking train-test contamination...")
        
        print("\n" + "="*60)
        print("🔬 TRAIN-TEST CONTAMINATION CHECK")
        print("="*60)
        
        # Simulate the exact split used in the model
        target = self.df['citation_count'].fillna(0)
        
        # Create a simple feature matrix
        X_simple = pd.DataFrame({
            'year': self.df['year'].fillna(2020),
            'citation_count': self.df['citation_count'].fillna(0)  # Include this to test
        })
        
        # Multiple train-test splits to check consistency
        maes = []
        r2s = []
        
        for random_state in [42, 1, 2, 3, 4]:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_simple, target, test_size=0.2, random_state=random_state
                )
                
                # Check if citation_count is in features (obvious leakage)
                if 'citation_count' in X_train.columns:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    maes.append(mae)
                    r2s.append(r2)
                    
            except Exception as e:
                print(f"Error with random_state {random_state}: {e}")
        
        if maes:
            print(f"MAE across different splits: {maes}")
            print(f"R² across different splits: {r2s}")
            print(f"MAE std: {np.std(maes):.6f}")
            print(f"R² std: {np.std(r2s):.6f}")
            
            if np.std(maes) < 0.001:
                print("🚨 EXTREMELY LOW VARIANCE - POTENTIAL DATA LEAKAGE!")
        
        # Check for ID overlap
        X_train, X_test, y_train, y_test = train_test_split(
            X_simple, target, test_size=0.2, random_state=42
        )
        
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        overlap = train_indices.intersection(test_indices)
        
        print(f"Train-test index overlap: {len(overlap)} samples")
        if len(overlap) > 0:
            print("🚨 TRAIN-TEST CONTAMINATION DETECTED!")
        
    def generate_report(self):
        """Generate comprehensive debugging report"""
        logger.info("Generating comprehensive debugging report...")
        
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE DATA LEAKAGE & OVERFITTING REPORT")
        print("="*80)
        
        # Run all checks
        basic_stats = self.check_basic_statistics()
        duplicate_check = self.check_duplicate_papers()
        correlation_check = self.check_feature_target_correlation()
        self.check_train_test_contamination()
        clean_results = self.simulate_clean_experiment()
        
        # Summary diagnosis
        print("\n" + "="*60)
        print("🎯 DIAGNOSIS SUMMARY")
        print("="*60)
        
        issues_found = []
        
        # Check for suspicious patterns
        if basic_stats['zero_citations_pct'] > 0.8:
            issues_found.append("High percentage of zero citations (suspicious)")
        
        if duplicate_check['content_duplicates'] > 0:
            issues_found.append(f"Content duplicates found: {duplicate_check['content_duplicates']}")
        
        # Check correlations
        high_corr_count = sum(1 for corr in correlation_check.values() if abs(corr) > 0.8)
        if high_corr_count > 0:
            issues_found.append(f"High correlations found: {high_corr_count}")
        
        # Check clean experiment results
        if clean_results:
            clean_mae = min(result['mae'] for result in clean_results.values())
            if clean_mae < 10:
                issues_found.append("Even clean experiment shows suspiciously low MAE")
        
        if issues_found:
            print("🚨 ISSUES DETECTED:")
            for i, issue in enumerate(issues_found, 1):
                print(f"  {i}. {issue}")
        else:
            print("✅ No obvious data leakage detected")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Remove citation_count from features if present")
        print(f"  2. Check feature engineering pipeline for leakage")
        print(f"  3. Verify train-test split methodology")
        print(f"  4. Use time-based splits for temporal data")
        print(f"  5. Cross-validate with multiple random seeds")
        
        return {
            'basic_stats': basic_stats,
            'duplicates': duplicate_check,
            'correlations': correlation_check,
            'clean_results': clean_results,
            'issues_found': issues_found
        }

def main():
    """Main execution"""
    print("🔍 DATA LEAKAGE & OVERFITTING DEBUG ANALYSIS")
    print("="*60)
    
    try:
        detector = DataLeakageDetector()
        dataset = detector.load_latest_results()
        
        if len(dataset) == 0:
            print("❌ No dataset loaded")
            return
        
        # Generate comprehensive report
        report = detector.generate_report()
        
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print(f"Dataset analyzed: {len(dataset)} papers")
        print(f"Issues found: {len(report['issues_found'])}")
        
        if report['issues_found']:
            print(f"\n🚨 MAJOR CONCLUSION: Results likely due to data leakage/overfitting")
            print(f"   Investigation needed before claiming research breakthrough")
        else:
            print(f"\n✅ No obvious issues found - results may be legitimate")
            
    except Exception as e:
        logger.error(f"Error in debugging analysis: {e}")
        raise

if __name__ == "__main__":
    main()