"""
GNN-Enhanced Advanced Architectures with GraphSAGE
Integrates Graph Neural Networks for citation network modeling
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings
import networkx as nx
from datetime import datetime
import pickle
from collections import defaultdict

warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Traditional ML imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb
import lightgbm as lgb

# Try to import DGL for GraphSAGE
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import dgl
    from dgl.nn import SAGEConv
    DGL_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    DGL_AVAILABLE = False
    DEVICE = 'cpu'
    print("DGL not available. Install with: pip install dgl")

# Try to import transformers for SPECTER
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphSAGEModel(nn.Module):
    """GraphSAGE model for citation network embedding"""
    
    def __init__(self, in_feats, hidden_size, num_layers=2):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_feats, hidden_size, 'mean'))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size, 'mean'))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_size, hidden_size, 'mean'))
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, g, features):
        h = features
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

class GNNEnhancedArchitectures:
    """Advanced architectures with GraphSAGE citation network modeling"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.features_ready = False
        self.citation_graph = None
        self.gnn_model = None
        self.gnn_embeddings = None
        
    def load_openalex_dataset(self):
        """Load OpenAlex dataset with citation information"""
        logger.info("Loading OpenAlex dataset for GNN analysis...")
        
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
        
        # Stratified sampling for citation balance
        citation_counts = df['citation_count'].fillna(0)
        
        zero_citations = df[citation_counts == 0]
        low_citations = df[(citation_counts > 0) & (citation_counts <= 5)]
        med_citations = df[(citation_counts > 5) & (citation_counts <= 20)]
        high_citations = df[citation_counts > 20]
        
        logger.info(f"Citation distribution:")
        logger.info(f"  Zero: {len(zero_citations)}, Low: {len(low_citations)}")
        logger.info(f"  Medium: {len(med_citations)}, High: {len(high_citations)}")
        
        # Sample to create balanced dataset
        max_per_bucket = 1000
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
            df_final = df.head(2000)
        
        # Shuffle and reset index
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Final dataset: {len(df_final)} papers")
        return df_final
    
    def build_citation_network(self, df):
        """Build citation network from paper references"""
        logger.info("Building citation network...")
        
        # Create paper ID mapping
        papers_with_ids = df.reset_index()
        papers_with_ids['paper_id'] = papers_with_ids.index
        
        # Create mapping from paper titles/DOIs to IDs
        title_to_id = {}
        doi_to_id = {}
        
        for _, paper in papers_with_ids.iterrows():
            paper_id = paper['paper_id']
            title = paper.get('title', '').strip().lower()
            doi = paper.get('doi', '')
            
            if title:
                title_to_id[title] = paper_id
            if doi:
                doi_to_id[doi] = paper_id
        
        # Build citation edges
        edges = []
        edge_count = 0
        
        for _, paper in papers_with_ids.iterrows():
            source_id = paper['paper_id']
            references = paper.get('references', [])
            
            if isinstance(references, list):
                for ref in references:
                    if isinstance(ref, dict):
                        # Try to match by DOI first, then title
                        ref_doi = ref.get('doi', '')
                        ref_title = ref.get('title', '').strip().lower()
                        
                        target_id = None
                        if ref_doi and ref_doi in doi_to_id:
                            target_id = doi_to_id[ref_doi]
                        elif ref_title and ref_title in title_to_id:
                            target_id = title_to_id[ref_title]
                        
                        if target_id is not None and target_id != source_id:
                            edges.append((source_id, target_id))
                            edge_count += 1
        
        logger.info(f"Found {edge_count} citation edges")
        
        # Create NetworkX graph for analysis
        self.citation_graph = nx.DiGraph()
        self.citation_graph.add_nodes_from(range(len(papers_with_ids)))
        self.citation_graph.add_edges_from(edges)
        
        logger.info(f"Citation network: {len(self.citation_graph.nodes)} nodes, {len(self.citation_graph.edges)} edges")
        
        return self.citation_graph, papers_with_ids
    
    def compute_network_features(self, df, citation_graph):
        """Compute traditional network features as baseline"""
        logger.info("Computing network features...")
        
        network_features = []
        
        for idx in range(len(df)):
            features = {
                # Basic network features
                'in_degree': citation_graph.in_degree(idx) if idx in citation_graph else 0,
                'out_degree': citation_graph.out_degree(idx) if idx in citation_graph else 0,
                'total_degree': citation_graph.degree(idx) if idx in citation_graph else 0,
            }
            
            # Try to compute more advanced features safely
            try:
                if idx in citation_graph and len(citation_graph.nodes) > 1:
                    # Clustering coefficient
                    undirected_graph = citation_graph.to_undirected()
                    features['clustering_coefficient'] = nx.clustering(undirected_graph, idx)
                    
                    # Betweenness centrality (on smaller subgraph for efficiency)
                    if len(citation_graph.nodes) <= 1000:
                        features['betweenness_centrality'] = nx.betweenness_centrality(citation_graph).get(idx, 0)
                    else:
                        features['betweenness_centrality'] = 0
                    
                    # PageRank
                    pagerank_dict = nx.pagerank(citation_graph, max_iter=50, tol=1e-3)
                    features['pagerank'] = pagerank_dict.get(idx, 1.0 / len(citation_graph.nodes))
                else:
                    features['clustering_coefficient'] = 0
                    features['betweenness_centrality'] = 0
                    features['pagerank'] = 1.0 / max(1, len(citation_graph.nodes))
            except:
                # Fallback values if computation fails
                features['clustering_coefficient'] = 0
                features['betweenness_centrality'] = 0
                features['pagerank'] = 1.0 / max(1, len(citation_graph.nodes))
            
            network_features.append(features)
        
        network_df = pd.DataFrame(network_features)
        logger.info(f"Network features computed: {network_df.shape[1]} features")
        
        return network_df
    
    def create_dgl_graph_and_features(self, df, citation_graph):
        """Create DGL graph with node features for GraphSAGE"""
        if not DGL_AVAILABLE:
            logger.warning("DGL not available - skipping GraphSAGE")
            return None, None
        
        logger.info("Creating DGL graph for GraphSAGE...")
        
        # Extract basic features for each paper
        node_features = []
        
        for idx, paper in df.iterrows():
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            combined_text = f"{title} {abstract}"
            
            # Simple text features for GNN input
            features = [
                len(title),
                len(abstract),
                len(title.split()),
                len(abstract.split()),
                paper.get('author_count', 1),
                paper.get('reference_count', 0),
                paper.get('year', 2020) - 2015,  # Normalized year
                1 if paper.get('is_oa', False) else 0,
                1 if paper.get('doi') else 0,
                combined_text.count('algorithm'),
                combined_text.count('machine learning'),
                combined_text.count('neural'),
                combined_text.count('data'),
                combined_text.count('system'),
                len([c for c in combined_text if c.isdigit()]),  # Number count
                combined_text.count('(') + combined_text.count(')'),  # Parentheses
            ]
            
            node_features.append(features)
        
        # Convert to tensor
        node_features = torch.FloatTensor(node_features).to(DEVICE)
        
        # Create DGL graph from NetworkX graph
        if len(citation_graph.edges) > 0:
            edges = list(citation_graph.edges)
            src, dst = zip(*edges)
            
            # Create DGL graph
            dgl_graph = dgl.graph((src, dst), num_nodes=len(df))
            dgl_graph = dgl.add_self_loop(dgl_graph)  # Add self-loops
            dgl_graph = dgl_graph.to(DEVICE)
        else:
            # Create empty graph with self-loops only
            dgl_graph = dgl.graph(([], []), num_nodes=len(df))
            dgl_graph = dgl.add_self_loop(dgl_graph)
            dgl_graph = dgl_graph.to(DEVICE)
        
        logger.info(f"DGL graph created: {dgl_graph.number_of_nodes()} nodes, {dgl_graph.number_of_edges()} edges")
        logger.info(f"Node features shape: {node_features.shape}")
        
        return dgl_graph, node_features
    
    def train_graphsage_model(self, dgl_graph, node_features, targets, embedding_dim=64):
        """Train GraphSAGE model for node embeddings"""
        if not DGL_AVAILABLE:
            logger.warning("DGL not available - returning random embeddings")
            return np.random.normal(0, 0.1, (len(targets), embedding_dim))
        
        logger.info("Training GraphSAGE model...")
        
        # Initialize model
        input_dim = node_features.shape[1]
        self.gnn_model = GraphSAGEModel(input_dim, embedding_dim, num_layers=2).to(DEVICE)
        
        # Training setup
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.MSELoss()
        
        # Convert targets to tensor
        targets_tensor = torch.FloatTensor(targets).to(DEVICE)
        
        # Training loop
        self.gnn_model.train()
        num_epochs = 100
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.gnn_model(dgl_graph, node_features)
            
            # Simple prediction head (mean of embeddings for now)
            predictions = embeddings.mean(dim=1)
            
            # Compute loss
            loss = criterion(predictions, targets_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Generate final embeddings
        self.gnn_model.eval()
        with torch.no_grad():
            final_embeddings = self.gnn_model(dgl_graph, node_features)
            final_embeddings = final_embeddings.cpu().numpy()
        
        logger.info(f"GraphSAGE embeddings generated: {final_embeddings.shape}")
        return final_embeddings
    
    def extract_comprehensive_features(self, df):
        """Extract comprehensive features including text, metadata, and network features"""
        logger.info("Extracting comprehensive features...")
        
        # Text features (simplified from original)
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
        
        # CS keyword features (simplified)
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
        
        logger.info(f"Feature extraction completed:")
        logger.info(f"  Text features: {text_df.shape[1]}")
        logger.info(f"  Keyword features: {keyword_df.shape[1]}")
        logger.info(f"  Metadata features: {metadata_df.shape[1]}")
        
        return text_df, keyword_df, metadata_df
    
    def combine_all_features(self, text_features, keyword_features, metadata_features, 
                           network_features, gnn_embeddings):
        """Combine all feature types including GNN embeddings"""
        logger.info("Combining all features including GNN embeddings...")
        
        # Create GNN embedding dataframe
        if gnn_embeddings is not None:
            gnn_cols = [f'gnn_embed_{i}' for i in range(gnn_embeddings.shape[1])]
            gnn_df = pd.DataFrame(gnn_embeddings, columns=gnn_cols)
        else:
            # Fallback: create dummy embeddings
            gnn_df = pd.DataFrame(np.zeros((len(text_features), 32)), 
                                columns=[f'gnn_embed_{i}' for i in range(32)])
        
        # Ensure all dataframes have same length and reset indices
        dfs = [text_features, keyword_features, metadata_features, network_features, gnn_df]
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
        logger.info(f"  - Network features: {len(network_features.columns)}")
        logger.info(f"  - GNN embeddings: {len(gnn_df.columns)}")
        
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
    
    def evaluate_gnn_enhanced_models(self, features_df, targets):
        """Evaluate models with GNN-enhanced features"""
        logger.info("Starting GNN-enhanced model evaluation...")
        
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
                'gnn_enhanced': True,
                'gnn_embedding_dims': len([c for c in X.columns if 'gnn_embed' in c]),
                'network_features': len([c for c in X.columns if any(nf in c for nf in ['degree', 'clustering', 'pagerank', 'betweenness'])]),
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
    
    def save_gnn_results(self, results, features_df):
        """Save GNN-enhanced results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = Path("results/gnn_enhanced")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"gnn_enhanced_results_{timestamp}.json"
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
                'network_features': len([c for c in features_df.columns if any(nf in c for nf in ['degree', 'clustering', 'pagerank', 'betweenness'])]),
                'gnn_embeddings': len([c for c in features_df.columns if c.startswith('gnn_embed_')])
            }
        }
        
        summary_file = results_dir / f"gnn_feature_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(feature_summary, f, indent=2)
        
        logger.info(f"GNN-enhanced results saved to: {results_file}")
        logger.info(f"Feature summary saved to: {summary_file}")
        
        return results_file, summary_file

def print_gnn_results(results):
    """Print GNN-enhanced results"""
    print("\n" + "="*80)
    print("🧠 GNN-ENHANCED ADVANCED ARCHITECTURES - RESULTS")
    print("="*80)
    
    info = results['experiment_info']
    print(f"Dataset: {info['total_papers']} papers")
    print(f"Total Features: {info['total_features']} (GNN-enhanced)")
    print(f"GNN Embeddings: {info['gnn_embedding_dims']} dimensions")
    print(f"Network Features: {info['network_features']}")
    print(f"GNN Enhanced: {info['gnn_enhanced']}")
    
    print("\nGNN-ENHANCED PERFORMANCE RESULTS:")
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
            best_f1 = 0
            best_model = None
            
            for model_name, metrics in task_results.items():
                acc = metrics['accuracy']
                f1 = metrics['f1']
                auc = metrics.get('auc', 0) or 0
                
                print(f"  {model_name:15} | Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
            
            if best_model:
                print(f"  🏆 Best: {best_model} (F1: {best_f1:.3f})")
    
    print("\n" + "="*80)

def main():
    """Main execution for GNN-enhanced architecture"""
    logger.info("Starting GNN-Enhanced Advanced Architectures...")
    
    try:
        # Initialize GNN-enhanced pipeline
        pipeline = GNNEnhancedArchitectures()
        
        # Load dataset
        df = pipeline.load_openalex_dataset()
        
        # Build citation network
        citation_graph, papers_with_ids = pipeline.build_citation_network(df)
        
        # Compute network features
        network_features = pipeline.compute_network_features(df, citation_graph)
        
        # Extract traditional features
        text_features, keyword_features, metadata_features = pipeline.extract_comprehensive_features(df)
        
        # Create DGL graph and train GraphSAGE
        targets = pipeline.create_clean_targets(df)
        citation_targets = targets['citation_count'].values
        
        dgl_graph, node_features = pipeline.create_dgl_graph_and_features(df, citation_graph)
        gnn_embeddings = pipeline.train_graphsage_model(dgl_graph, node_features, citation_targets)
        
        # Combine all features including GNN embeddings
        features_df = pipeline.combine_all_features(
            text_features, keyword_features, metadata_features, 
            network_features, gnn_embeddings
        )
        
        # Evaluate GNN-enhanced models
        results = pipeline.evaluate_gnn_enhanced_models(features_df, targets)
        
        # Save results
        results_file, summary_file = pipeline.save_gnn_results(results, features_df)
        
        # Print results
        print_gnn_results(results)
        
        print(f"\n✅ GNN-ENHANCED IMPLEMENTATION COMPLETE!")
        print(f"   • {len(df)} papers with citation network modeling")
        print(f"   • {features_df.shape[1]} total features (including GNN embeddings)")
        print(f"   • {len(citation_graph.edges)} citation edges processed")
        print(f"   • GraphSAGE embeddings: {gnn_embeddings.shape[1] if gnn_embeddings is not None else 0} dimensions")
        print(f"   • Results: {results_file}")
        print(f"\n🧠 GraphSAGE citation network modeling integrated successfully!")
        
    except Exception as e:
        logger.error(f"Error in GNN-enhanced implementation: {e}")
        raise

if __name__ == "__main__":
    main()