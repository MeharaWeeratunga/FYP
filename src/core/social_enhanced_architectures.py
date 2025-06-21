"""
Social Media Enhanced Architectures with Altmetrics Integration
Combines existing multi-modal features with social media signals for improved early prediction
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

# Import our existing components
import sys
sys.path.append('src/core')

try:
    from altmetric_integration import AltmetricAPIIntegrator
    ALTMETRIC_AVAILABLE = True
except ImportError:
    ALTMETRIC_AVAILABLE = False
    print("Altmetric integration not available")

# Core ML libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ML models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb
import lightgbm as lgb

# Try to import transformers for SPECTER
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DEVICE = 'cpu'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SocialEnhancedArchitectures:
    """
    Advanced architectures enhanced with social media and altmetrics data
    Integrates Altmetric API data with existing multi-modal features
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.features_ready = False
        self.altmetric_integrator = None
        
        if ALTMETRIC_AVAILABLE:
            self.altmetric_integrator = AltmetricAPIIntegrator(rate_limit_delay=0.5)
        
    def load_openalex_dataset(self):
        """Load OpenAlex dataset"""
        logger.info("Loading OpenAlex dataset for social enhancement...")
        
        # Find the dataset - FIXED VERSION: Prioritize new ArXiv dataset
        possible_paths = [
            "data/datasets/cs_papers_arxiv_50k.json",  # NEW ArXiv dataset (50K papers)
            "data/datasets/openalex_5000_papers.json",  # OLD problematic dataset
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
            raise FileNotFoundError(f"No OpenAlex dataset found. Checked: {possible_paths}")
        
        latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
        logger.info(f"Loading dataset from: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', [])
        df = pd.json_normalize(papers)
        
        logger.info(f"Raw dataset: {len(df)} papers")
        
        # Apply quality filtering
        df = self.apply_quality_filters(df)
        
        logger.info(f"After quality filtering: {len(df)} papers")
        return df
    
    def apply_quality_filters(self, df):
        """Apply quality filters similar to original implementation"""
        logger.info("Applying quality filters...")
        
        original_len = len(df)
        
        # Basic requirements
        df = df.dropna(subset=['title', 'abstract'])
        df = df[df['title'].str.len() >= 5]
        df = df[df['abstract'].str.len() >= 50]
        df = df[df['year'] >= 2015]
        df = df[df['year'] <= 2023]
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        logger.info(f"After basic filtering: {len(df)}/{original_len}")
        
        # Stratified sampling for citation balance (similar to previous implementations)
        citation_counts = df['citation_count'].fillna(0)
        
        zero_citations = df[citation_counts == 0]
        low_citations = df[(citation_counts > 0) & (citation_counts <= 5)]
        med_citations = df[(citation_counts > 5) & (citation_counts <= 20)]
        high_citations = df[citation_counts > 20]
        
        # Sample to create balanced dataset
        max_per_bucket = 500  # Smaller for social media integration testing
        sampled_parts = []
        
        for bucket, name in [(zero_citations, "zero"), (low_citations, "low"), 
                           (med_citations, "medium"), (high_citations, "high")]:
            if len(bucket) > 0:
                n_sample = min(len(bucket), max_per_bucket)
                if len(bucket) > n_sample:
                    sample = bucket.sample(n=n_sample, random_state=42)
                else:
                    sample = bucket
                sampled_parts.append(sample)
                logger.info(f"  Sampled {len(sample)} {name} citation papers")
        
        if sampled_parts:
            df_final = pd.concat(sampled_parts, ignore_index=True)
        else:
            df_final = df.head(1000)  # Fallback
        
        # Shuffle and reset index
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Final dataset: {len(df_final)} papers")
        return df_final
    
    def extract_social_media_features(self, df):
        """Extract social media features using Altmetric API"""
        if not ALTMETRIC_AVAILABLE or not self.altmetric_integrator:
            logger.warning("Altmetric integration not available - using mock social features")
            return self.create_mock_social_features(df)
        
        logger.info("Extracting social media features via Altmetric API...")
        
        # Process papers for Altmetric data
        altmetric_df = self.altmetric_integrator.process_paper_batch(df, batch_size=50)
        
        # Extract just the feature columns (remove metadata)
        feature_columns = [col for col in altmetric_df.columns 
                          if col not in ['paper_index', 'doi']]
        
        social_features_df = altmetric_df[feature_columns].copy()
        
        logger.info(f"Social media features extracted: {social_features_df.shape[1]} features")
        return social_features_df
    
    def create_mock_social_features(self, df):
        """Create mock social features when API is not available"""
        logger.info("Creating mock social media features...")
        
        # Create realistic mock data based on typical distributions
        n_papers = len(df)
        
        # Most papers have no social media attention (90-95%)
        # Some have moderate attention (5-8%)
        # Few have high attention (1-2%)
        
        mock_features = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_papers):
            # Simulate social media attention distribution
            has_attention = np.random.random() < 0.08  # 8% have some attention
            
            if has_attention:
                # Generate realistic social media numbers
                twitter_mentions = np.random.poisson(3)
                news_mentions = np.random.poisson(1)
                altmetric_score = twitter_mentions * 0.5 + news_mentions * 3.0 + np.random.exponential(2)
            else:
                twitter_mentions = 0
                news_mentions = 0
                altmetric_score = 0
            
            features = {
                'altmetric_score': altmetric_score,
                'twitter_mentions': twitter_mentions,
                'facebook_mentions': np.random.poisson(0.5) if has_attention else 0,
                'news_mentions': news_mentions,
                'blog_mentions': np.random.poisson(0.8) if has_attention else 0,
                'wikipedia_mentions': 1 if np.random.random() < 0.02 else 0,
                'policy_mentions': 1 if np.random.random() < 0.01 else 0,
                'mendeley_readers': np.random.poisson(15) if has_attention else np.random.poisson(2),
                'social_media_total': twitter_mentions + (1 if has_attention else 0),
                'media_coverage_total': news_mentions + (1 if has_attention else 0),
                'academic_attention_total': (1 if np.random.random() < 0.02 else 0) + np.random.poisson(2),
                'has_altmetric_data': 1 if has_attention else 0,
                'twitter_influence_score': twitter_mentions * np.random.uniform(1.0, 3.0) if twitter_mentions > 0 else 0,
            }
            
            mock_features.append(features)
        
        mock_df = pd.DataFrame(mock_features)
        logger.info(f"Mock social features created: {mock_df.shape[1]} features")
        logger.info(f"Papers with social attention: {mock_df['has_altmetric_data'].sum()} ({mock_df['has_altmetric_data'].mean()*100:.1f}%)")
        
        return mock_df
    
    def extract_core_features(self, df):
        """Extract core features (text, metadata) from our existing implementations"""
        logger.info("Extracting core features...")
        
        # Text features
        text_features = []
        for _, paper in df.iterrows():
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            combined_text = f"{title} {abstract}"
            
            features = {
                'title_length': len(title),
                'abstract_length': len(abstract),
                'title_words': len(title.split()),
                'abstract_words': len(abstract.split()),
                'avg_word_length': np.mean([len(w) for w in combined_text.split()]) if combined_text.split() else 0,
                'sentence_count': combined_text.count('.') + combined_text.count('!') + combined_text.count('?'),
                'math_symbols': combined_text.count('=') + combined_text.count('+') + combined_text.count('-'),
                'parentheses_count': combined_text.count('(') + combined_text.count(')'),
                'numbers_count': len([c for c in combined_text if c.isdigit()]),
                'hypothesis_words': sum(1 for word in ['hypothesis', 'propose', 'novel', 'new'] 
                                      if word in combined_text.lower()),
            }
            text_features.append(features)
        
        text_df = pd.DataFrame(text_features)
        
        # CS keyword features
        cs_keywords = {
            'machine_learning': ['machine learning', 'deep learning', 'neural network', 'ai'],
            'algorithms': ['algorithm', 'optimization', 'complexity'],
            'systems': ['system', 'distributed', 'parallel'],
            'data': ['data', 'database', 'analytics'],
            'security': ['security', 'cryptography', 'privacy']
        }
        
        keyword_features = []
        for _, paper in df.iterrows():
            combined_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            
            features = {}
            for category, keywords in cs_keywords.items():
                count = sum(combined_text.count(kw) for kw in keywords)
                features[f'cs_{category}_count'] = count
            
            features['cs_total_keywords'] = sum(features.values())
            keyword_features.append(features)
        
        keyword_df = pd.DataFrame(keyword_features)
        
        # Metadata features
        metadata_features = []
        for _, paper in df.iterrows():
            features = {
                'publication_year': paper.get('year', 2020),
                'author_count': min(paper.get('author_count', 1), 20),
                'reference_count': min(paper.get('reference_count', 0), 200),
                'is_open_access': 1 if paper.get('is_oa', False) else 0,
                'has_doi': 1 if paper.get('doi') else 0,
                'single_author': 1 if paper.get('author_count', 1) == 1 else 0,
                'many_authors': 1 if paper.get('author_count', 1) >= 5 else 0,
                'paper_age_from_2015': max(0, paper.get('year', 2020) - 2015),
            }
            metadata_features.append(features)
        
        metadata_df = pd.DataFrame(metadata_features)
        
        logger.info(f"Core features extracted:")
        logger.info(f"  Text features: {text_df.shape[1]}")
        logger.info(f"  Keyword features: {keyword_df.shape[1]}")
        logger.info(f"  Metadata features: {metadata_df.shape[1]}")
        
        return text_df, keyword_df, metadata_df
    
    def extract_specter_embeddings(self, df, max_papers=1000):
        """Extract SPECTER embeddings (simplified version)"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("SPECTER not available, using TF-IDF alternative")
            return self.extract_tfidf_embeddings(df)
        
        logger.info("Extracting SPECTER embeddings...")
        
        # Limit papers for memory efficiency
        if len(df) > max_papers:
            logger.info(f"Sampling {max_papers} papers for SPECTER")
            df_sample = df.sample(n=max_papers, random_state=42)
        else:
            df_sample = df
        
        try:
            # Load SPECTER model
            tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
            model = AutoModel.from_pretrained("allenai/specter")
            model.to(DEVICE)
            model.eval()
            
            # Prepare texts
            texts = []
            for _, paper in df_sample.iterrows():
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                combined = f"{title} [SEP] {abstract}"
                texts.append(combined)
            
            # Extract embeddings in small batches
            batch_size = 8
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(embeddings)
            
            final_embeddings = np.vstack(all_embeddings)
            
            # Handle full dataset if we sampled
            if len(df) > max_papers:
                remaining_df = df.drop(df_sample.index)
                tfidf_embeddings = self.extract_tfidf_embeddings(remaining_df, target_dims=768)
                
                full_embeddings = np.zeros((len(df), 768))
                full_embeddings[df_sample.index] = final_embeddings
                full_embeddings[remaining_df.index] = tfidf_embeddings
                final_embeddings = full_embeddings
            
            logger.info(f"SPECTER embeddings extracted: {final_embeddings.shape}")
            return final_embeddings
            
        except Exception as e:
            logger.error(f"Error extracting SPECTER embeddings: {e}")
            return self.extract_tfidf_embeddings(df)
    
    def extract_tfidf_embeddings(self, df, target_dims=768):
        """Fallback TF-IDF embeddings"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import PCA
        
        logger.info(f"Extracting TF-IDF embeddings (target dims: {target_dims})...")
        
        texts = []
        for _, paper in df.iterrows():
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            texts.append(f"{title} {abstract}")
        
        tfidf = TfidfVectorizer(
            max_features=min(2000, len(texts) // 2),
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        tfidf_features = tfidf.fit_transform(texts)
        
        # PCA to target dimensions
        n_components = min(target_dims, tfidf_features.shape[1], tfidf_features.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        embeddings = pca.fit_transform(tfidf_features.toarray())
        
        # Pad to target dimensions if necessary
        if embeddings.shape[1] < target_dims:
            padding = np.zeros((embeddings.shape[0], target_dims - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        
        logger.info(f"TF-IDF embeddings extracted: {embeddings.shape}")
        return embeddings
    
    def combine_all_features(self, text_features, keyword_features, metadata_features, 
                           social_features, embeddings):
        """Combine all feature types including social media features"""
        logger.info("Combining all features including social media...")
        
        # Create embedding dataframe
        embedding_cols = [f'embed_{i}' for i in range(embeddings.shape[1])]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
        
        # Ensure all dataframes have same length and reset indices
        dfs = [text_features, keyword_features, metadata_features, social_features, embedding_df]
        for df in dfs:
            df.reset_index(drop=True, inplace=True)
        
        # Combine all features
        combined_features = pd.concat(dfs, axis=1)
        
        # Verify no target leakage
        forbidden_columns = ['citation_count', 'cited_by_count', 'citations', 'impact']
        leaked_columns = [col for col in combined_features.columns 
                         if any(forbidden in col.lower() for forbidden in forbidden_columns)]
        
        if leaked_columns:
            logger.error(f"POTENTIAL LEAKAGE DETECTED: {leaked_columns}")
            raise ValueError(f"Found potentially leaked columns: {leaked_columns}")
        
        logger.info(f"Combined features: {combined_features.shape}")
        logger.info(f"Feature breakdown:")
        logger.info(f"  - Text features: {len(text_features.columns)}")
        logger.info(f"  - Keyword features: {len(keyword_features.columns)}")
        logger.info(f"  - Metadata features: {len(metadata_features.columns)}")
        logger.info(f"  - Social media features: {len(social_features.columns)}")
        logger.info(f"  - Embedding features: {len(embedding_df.columns)}")
        
        return combined_features
    
    def create_clean_targets(self, df):
        """Create target variables ensuring no leakage"""
        logger.info("Creating clean target variables...")
        
        citation_counts = df['citation_count'].fillna(0).astype(float)
        
        targets = {
            'citation_count': citation_counts,
            'log_citation_count': np.log1p(citation_counts),
            'sqrt_citation_count': np.sqrt(citation_counts)
        }
        
        # Classification targets
        targets['high_impact_top10'] = (citation_counts >= citation_counts.quantile(0.9)).astype(int)
        targets['high_impact_top20'] = (citation_counts >= citation_counts.quantile(0.8)).astype(int)
        targets['cited_vs_uncited'] = (citation_counts > 0).astype(int)
        targets['any_impact'] = (citation_counts > 0).astype(int)
        
        logger.info("Target variable statistics:")
        for name, target in targets.items():
            if target.dtype in ['int32', 'int64']:
                pos_rate = target.mean()
                logger.info(f"  {name}: {target.sum()}/{len(target)} positive ({pos_rate:.1%})")
            else:
                logger.info(f"  {name}: mean={target.mean():.2f}, std={target.std():.2f}, max={target.max():.0f}")
        
        return targets
    
    def evaluate_social_enhanced_models(self, features_df, targets):
        """Evaluate models with social media enhanced features"""
        logger.info("Starting social media enhanced model evaluation...")
        
        # Clean feature matrix
        X = features_df.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Model configurations
        models = {
            'regression': {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'XGBoost': xgb.XGBRegressor(random_state=42, max_depth=6, n_estimators=100),
                'LightGBM': lgb.LGBMRegressor(random_state=42, max_depth=6, n_estimators=100, verbose=-1),
                'SVR': SVR(kernel='rbf', C=1.0)
            },
            'classification': {
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                'XGBoost': xgb.XGBClassifier(random_state=42, max_depth=6, n_estimators=100),
                'LightGBM': lgb.LGBMClassifier(random_state=42, max_depth=6, n_estimators=100, verbose=-1),
                'SVC': SVC(kernel='rbf', probability=True, random_state=42)
            }
        }
        
        results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'total_papers': len(X),
                'total_features': len(X.columns),
                'social_media_enhanced': True,
                'social_features': len([c for c in X.columns if any(sf in c for sf in ['altmetric', 'twitter', 'news', 'social', 'media'])]),
                'transformers_used': TRANSFORMERS_AVAILABLE,
                'data_leakage_checked': True
            },
            'performance_results': {}
        }
        
        # Evaluate regression tasks
        regression_tasks = ['citation_count', 'log_citation_count']
        for task in regression_tasks:
            if task not in targets:
                continue
            
            logger.info(f"Evaluating regression: {task}")
            y = targets[task]
            
            if y.std() == 0:
                logger.warning(f"Skipping {task} - no variance")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            task_results = {}
            for model_name, model in models['regression'].items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation
                    try:
                        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
                        cv_mae = -cv_scores.mean()
                        cv_std = cv_scores.std()
                    except:
                        cv_mae = cv_std = None
                    
                    task_results[model_name] = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'r2': r2,
                        'cv_mae': cv_mae,
                        'cv_std': cv_std
                    }
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} failed for {task}: {e}")
            
            results['performance_results'][task] = task_results
        
        # Evaluate classification tasks
        classification_tasks = ['cited_vs_uncited', 'any_impact']
        for task in classification_tasks:
            if task not in targets:
                continue
            
            y = targets[task]
            
            if y.sum() < 10:
                logger.warning(f"Skipping {task} - too few positive examples ({y.sum()})")
                continue
            
            logger.info(f"Evaluating classification: {task}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            task_results = {}
            for model_name, model in models['classification'].items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    # AUC if possible
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_prob = model.predict_proba(X_test)[:, 1]
                            auc = roc_auc_score(y_test, y_prob)
                        else:
                            y_score = model.decision_function(X_test)
                            auc = roc_auc_score(y_test, y_score)
                    except:
                        auc = None
                    
                    task_results[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc
                    }
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} failed for {task}: {e}")
            
            results['performance_results'][task] = task_results
        
        return results
    
    def save_social_enhanced_results(self, results, features_df):
        """Save social media enhanced results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = Path("results/social_enhanced")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"social_enhanced_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save feature summary
        feature_summary = {
            'timestamp': timestamp,
            'total_features': len(features_df.columns),
            'feature_types': {
                'text_features': len([c for c in features_df.columns if any(prefix in c for prefix in ['title_', 'abstract_', 'avg_', 'sentence_'])]),
                'keyword_features': len([c for c in features_df.columns if c.startswith('cs_')]),
                'metadata_features': len([c for c in features_df.columns if any(prefix in c for prefix in ['publication_', 'author_', 'has_', 'is_'])]),
                'social_features': len([c for c in features_df.columns if any(sf in c for sf in ['altmetric', 'twitter', 'news', 'social', 'media'])]),
                'embedding_features': len([c for c in features_df.columns if c.startswith('embed_')])
            }
        }
        
        summary_file = results_dir / f"social_feature_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(feature_summary, f, indent=2)
        
        logger.info(f"Social enhanced results saved to: {results_file}")
        logger.info(f"Feature summary saved to: {summary_file}")
        
        return results_file, summary_file

def print_social_enhanced_results(results):
    """Print social media enhanced results"""
    print("\n" + "="*80)
    print("🌐 SOCIAL MEDIA ENHANCED ARCHITECTURES - RESULTS")
    print("="*80)
    
    info = results['experiment_info']
    print(f"Dataset: {info['total_papers']} papers")
    print(f"Total Features: {info['total_features']} (social enhanced)")
    print(f"Social Media Features: {info['social_features']}")
    print(f"Transformers Used: {info['transformers_used']}")
    print(f"Social Enhanced: {info['social_media_enhanced']}")
    
    print("\nSOCIAL MEDIA ENHANCED PERFORMANCE RESULTS:")
    print("-" * 70)
    
    for task, task_results in results['performance_results'].items():
        print(f"\n{task.upper()}:")
        
        if 'mae' in str(task_results):  # Regression
            best_mae = float('inf')
            best_model = None
            
            for model_name, metrics in task_results.items():
                mae = metrics['mae']
                r2 = metrics['r2']
                cv_mae = metrics.get('cv_mae')
                
                if cv_mae:
                    print(f"  {model_name:15} | MAE: {mae:6.2f} | R²: {r2:6.3f} | CV-MAE: {cv_mae:6.2f}")
                else:
                    print(f"  {model_name:15} | MAE: {mae:6.2f} | R²: {r2:6.3f}")
                
                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name
            
            if best_model:
                print(f"  🏆 Best: {best_model} (MAE: {best_mae:.2f})")
        
        else:  # Classification
            best_auc = 0
            best_model = None
            
            for model_name, metrics in task_results.items():
                acc = metrics['accuracy']
                f1 = metrics['f1']
                auc = metrics.get('auc', 0) or 0
                
                print(f"  {model_name:15} | Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_model = model_name
            
            if best_model:
                print(f"  🏆 Best: {best_model} (AUC: {best_auc:.3f})")
    
    print("\n" + "="*80)

def main():
    """Main execution for social media enhanced architecture"""
    logger.info("Starting Social Media Enhanced Advanced Architectures...")
    
    try:
        # Initialize social enhanced pipeline
        pipeline = SocialEnhancedArchitectures()
        
        # Load dataset
        df = pipeline.load_openalex_dataset()
        
        # Extract core features (text, metadata, CS keywords)
        text_features, keyword_features, metadata_features = pipeline.extract_core_features(df)
        
        # Extract social media features
        social_features = pipeline.extract_social_media_features(df)
        
        # Extract embeddings (SPECTER or TF-IDF)
        embeddings = pipeline.extract_specter_embeddings(df)
        
        # Combine all features including social media
        features_df = pipeline.combine_all_features(
            text_features, keyword_features, metadata_features, 
            social_features, embeddings
        )
        
        # Create targets
        targets = pipeline.create_clean_targets(df)
        
        # Evaluate social enhanced models
        results = pipeline.evaluate_social_enhanced_models(features_df, targets)
        
        # Save results
        results_file, summary_file = pipeline.save_social_enhanced_results(results, features_df)
        
        # Print results
        print_social_enhanced_results(results)
        
        print(f"\n✅ SOCIAL MEDIA ENHANCED IMPLEMENTATION COMPLETE!")
        print(f"   • {len(df)} papers with social media integration")
        print(f"   • {features_df.shape[1]} total features (including social media)")
        print(f"   • {results['experiment_info']['social_features']} social media features")
        print(f"   • Altmetric API integration: {'Available' if ALTMETRIC_AVAILABLE else 'Mock data used'}")
        print(f"   • Results: {results_file}")
        print(f"\n🌐 Social media signals integrated for enhanced early virality prediction!")
        
    except Exception as e:
        logger.error(f"Error in social media enhanced implementation: {e}")
        raise

if __name__ == "__main__":
    main()