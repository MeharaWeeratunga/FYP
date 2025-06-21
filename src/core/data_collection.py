"""
OpenAlex Snapshot Extractor for Computer Science Papers
Downloads and processes OpenAlex snapshot data efficiently
Target: Extract 10,000+ high-quality CS papers for transformer training
"""

import json
import gzip
import boto3
import os
import logging
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import concurrent.futures
from collections import defaultdict
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAlexSnapshotExtractor:
    """Efficient OpenAlex snapshot processor for CS papers"""
    
    def __init__(self, target_papers=10000, max_files=50):
        self.target_papers = target_papers
        self.max_files = max_files
        self.collected_papers = []
        self.seen_papers = set()
        self.bucket_name = 'openalex'
        from botocore import UNSIGNED
        from botocore.config import Config
        self.s3_client = boto3.client('s3', 
                                     config=Config(signature_version=UNSIGNED))
        
        # CS-related concepts and venues for filtering
        self.cs_concepts = {
            'computer_science', 'machine_learning', 'artificial_intelligence', 
            'deep_learning', 'neural_network', 'computer_vision', 'natural_language_processing',
            'data_mining', 'algorithm', 'software_engineering', 'database', 'distributed_computing',
            'information_retrieval', 'human_computer_interaction', 'robotics', 'programming_language',
            'operating_system', 'computer_network', 'cryptography', 'computational_complexity',
            'data_structure', 'pattern_recognition', 'knowledge_representation'
        }
        
        self.cs_venues = {
            'neurips', 'nips', 'icml', 'iclr', 'aaai', 'ijcai', 'kdd', 'icdm',
            'cvpr', 'iccv', 'eccv', 'acl', 'emnlp', 'naacl', 'coling',
            'sigir', 'www', 'wsdm', 'cikm', 'recsys', 'sigmod', 'vldb',
            'icde', 'stoc', 'focs', 'soda', 'popl', 'pldi', 'icse',
            'osdi', 'sosp', 'nsdi', 'sigcomm', 'infocom', 'mobicom'
        }
        
        self.lock = threading.Lock()
        
    def list_works_files(self, limit=None):
        """List available works files from OpenAlex S3 bucket"""
        logger.info("Listing OpenAlex works files...")
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix='data/works/',
                PaginationConfig={'MaxItems': 2000}
            )
            
            files = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        size = obj['Size']
                        
                        # Only get .gz files, skip manifest and very small files
                        if (key.endswith('.gz') and 'part_' in key and 
                            size > 1000000):  # At least 1MB files
                            files.append({
                                'key': key,
                                'size': size,
                                'last_modified': obj['LastModified']
                            })
                            
                            if limit and len(files) >= limit:
                                break
                
                if limit and len(files) >= limit:
                    break
            
            # Sort by size (larger files first for more papers per file)
            files.sort(key=lambda x: x['size'], reverse=True)
            
            logger.info(f"Found {len(files)} substantial works files")
            
            # Log file size distribution
            if files:
                sizes_mb = [f['size'] / 1024 / 1024 for f in files[:10]]
                logger.info(f"Top file sizes (MB): {[f'{s:.1f}' for s in sizes_mb]}")
            
            return files[:limit] if limit else files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def download_and_process_file(self, file_info):
        """Download and process a single works file"""
        key = file_info['key']
        filename = key.split('/')[-1]
        local_path = f"/tmp/{filename}"
        
        try:
            logger.info(f"Processing {filename} ({file_info['size']/1024/1024:.1f} MB)")
            
            # Download file
            self.s3_client.download_file(self.bucket_name, key, local_path)
            
            # Process file
            papers_found = self.process_works_file(local_path)
            
            # Clean up
            os.remove(local_path)
            
            logger.info(f"Completed {filename}: {papers_found} CS papers found")
            return papers_found
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return 0
    
    def process_works_file(self, file_path):
        """Process a single gzipped works file"""
        papers_found = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        work = json.loads(line.strip())
                        
                        # Check if we've reached target
                        with self.lock:
                            if len(self.collected_papers) >= self.target_papers:
                                return papers_found
                        
                        # Filter for CS papers
                        if self.is_cs_paper(work):
                            # Avoid duplicates
                            paper_id = work.get('id', '')
                            title_hash = hash(work.get('title', '').lower().strip())
                            
                            with self.lock:
                                if paper_id not in self.seen_papers and title_hash not in self.seen_papers:
                                    self.seen_papers.add(paper_id)
                                    self.seen_papers.add(title_hash)
                                    
                                    # Process and add paper
                                    processed_paper = self.process_work(work)
                                    if processed_paper:
                                        self.collected_papers.append(processed_paper)
                                        papers_found += 1
                                        
                                        if len(self.collected_papers) % 500 == 0:
                                            logger.info(f"Collected {len(self.collected_papers)} papers so far...")
                        
                        # Progress logging
                        if line_num % 10000 == 0 and line_num > 0:
                            logger.debug(f"Processed {line_num} lines in {file_path}")
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
        
        return papers_found
    
    def is_cs_paper(self, work):
        """Determine if a work is a computer science paper"""
        try:
            # Check basic requirements
            if not work.get('title') or not work.get('abstract_inverted_index'):
                return False
            
            # Year filter (recent papers)
            pub_year = work.get('publication_year')
            if not pub_year or pub_year < 2015 or pub_year > 2024:
                return False
            
            # Language filter
            if work.get('language') and work.get('language') != 'en':
                return False
            
            # Type filter (articles and conference papers)
            work_type = work.get('type', '').lower()
            if work_type not in ['article', 'proceedings-article', 'book-chapter']:
                return False
            
            # Check for CS concepts
            concepts = work.get('concepts', [])
            for concept in concepts:
                if concept.get('display_name', '').lower().replace(' ', '_') in self.cs_concepts:
                    return True
            
            # Check venue
            primary_location = work.get('primary_location', {})
            if primary_location:
                source = primary_location.get('source', {})
                if source:
                    venue_name = source.get('display_name', '').lower()
                    if any(cs_venue in venue_name for cs_venue in self.cs_venues):
                        return True
            
            # Check title and abstract for CS keywords
            title = work.get('title', '').lower()
            abstract_text = self.reconstruct_abstract(work.get('abstract_inverted_index', {}))
            combined_text = (title + ' ' + abstract_text).lower()
            
            cs_keywords = [
                'algorithm', 'machine learning', 'neural network', 'computer', 'software',
                'artificial intelligence', 'data mining', 'deep learning', 'classification',
                'optimization', 'database', 'network', 'system', 'programming', 'model'
            ]
            
            if any(keyword in combined_text for keyword in cs_keywords):
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking CS paper: {e}")
            return False
    
    def reconstruct_abstract(self, inverted_index):
        """Reconstruct abstract from inverted index"""
        if not inverted_index:
            return ""
        
        try:
            # Create word position mapping
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            
            # Sort by position and reconstruct
            word_positions.sort(key=lambda x: x[0])
            abstract = ' '.join([word for _, word in word_positions])
            
            return abstract[:2000]  # Limit length
            
        except Exception as e:
            logger.warning(f"Error reconstructing abstract: {e}")
            return ""
    
    def process_work(self, work):
        """Process and standardize a work object"""
        try:
            # Basic information
            paper = {
                'id': work.get('id', ''),
                'title': work.get('title', ''),
                'abstract': self.reconstruct_abstract(work.get('abstract_inverted_index', {})),
                'year': work.get('publication_year'),
                'publication_date': work.get('publication_date'),
                'type': work.get('type'),
                'language': work.get('language', 'en')
            }
            
            # Citation metrics
            paper['citation_count'] = work.get('cited_by_count', 0)
            paper['reference_count'] = len(work.get('referenced_works', []))
            
            # Authors
            authors = work.get('authorships', [])
            paper['authors'] = []
            paper['author_count'] = len(authors)
            
            for authorship in authors[:10]:  # Limit to first 10 authors
                author = authorship.get('author', {})
                if author:
                    paper['authors'].append({
                        'id': author.get('id', ''),
                        'name': author.get('display_name', ''),
                        'orcid': author.get('orcid')
                    })
            
            # Venue information
            primary_location = work.get('primary_location', {})
            if primary_location:
                source = primary_location.get('source', {})
                paper['venue'] = source.get('display_name', '')
                paper['venue_id'] = source.get('id', '')
                paper['venue_type'] = source.get('type', '')
                paper['is_oa'] = primary_location.get('is_oa', False)
            else:
                paper['venue'] = ''
                paper['venue_id'] = ''
                paper['venue_type'] = ''
                paper['is_oa'] = False
            
            # Concepts (research areas)
            concepts = work.get('concepts', [])
            paper['concepts'] = []
            for concept in concepts[:10]:  # Top 10 concepts
                paper['concepts'].append({
                    'id': concept.get('id', ''),
                    'name': concept.get('display_name', ''),
                    'score': concept.get('score', 0),
                    'level': concept.get('level', 0)
                })
            
            # DOI and external IDs
            paper['doi'] = work.get('doi')
            external_ids = work.get('ids', {})
            paper['openalex_id'] = external_ids.get('openalex', '')
            paper['mag_id'] = external_ids.get('mag')
            paper['pmid'] = external_ids.get('pmid')
            
            # Quality checks
            if not paper['title'] or len(paper['title']) < 10:
                return None
            
            if not paper['abstract'] or len(paper['abstract']) < 100:
                return None
            
            # Add processing metadata
            paper['source'] = 'openalex_snapshot'
            paper['collection_timestamp'] = datetime.now().isoformat()
            paper['quality_score'] = self.calculate_quality_score(paper)
            
            return paper
            
        except Exception as e:
            logger.warning(f"Error processing work: {e}")
            return None
    
    def calculate_quality_score(self, paper):
        """Calculate a quality score for the paper"""
        score = 0
        
        # Title length (10-200 chars)
        title_len = len(paper.get('title', ''))
        if 10 <= title_len <= 200:
            score += 20
        
        # Abstract length (100-3000 chars)
        abstract_len = len(paper.get('abstract', ''))
        if 100 <= abstract_len <= 3000:
            score += 30
        
        # Has DOI
        if paper.get('doi'):
            score += 10
        
        # Recent publication
        year = paper.get('year', 0)
        if year >= 2020:
            score += 15
        elif year >= 2015:
            score += 10
        
        # Has venue
        if paper.get('venue'):
            score += 10
        
        # Author count (1-20 authors)
        author_count = paper.get('author_count', 0)
        if 1 <= author_count <= 20:
            score += 10
        
        # Citations (diminishing returns)
        citations = paper.get('citation_count', 0)
        if citations > 0:
            score += min(15, citations // 5)
        
        return score
    
    def run_extraction(self):
        """Run the complete extraction process"""
        logger.info(f"Starting OpenAlex extraction - target: {self.target_papers} papers")
        
        # List available files
        files = self.list_works_files(limit=self.max_files)
        
        if not files:
            logger.error("No files found to process")
            return []
        
        logger.info(f"Will process {len(files)} files")
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_file = {
                executor.submit(self.download_and_process_file, file_info): file_info 
                for file_info in files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    papers_found = future.result()
                    logger.info(f"Completed {file_info['key']}: {papers_found} papers")
                    
                    # Check if we've reached target
                    with self.lock:
                        if len(self.collected_papers) >= self.target_papers:
                            logger.info(f"Target of {self.target_papers} papers reached!")
                            # Cancel remaining futures
                            for remaining_future in future_to_file:
                                remaining_future.cancel()
                            break
                            
                except Exception as e:
                    logger.error(f"Error processing {file_info['key']}: {e}")
        
        logger.info(f"Extraction completed: {len(self.collected_papers)} papers collected")
        return self.collected_papers
    
    def save_dataset(self):
        """Save the extracted dataset"""
        if not self.collected_papers:
            logger.warning("No papers to save")
            return None, None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = Path("../data/openalex_extract/")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sort papers by quality score
        self.collected_papers.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Prepare dataset
        dataset = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'method': 'openalex_snapshot_extraction',
                'total_papers': len(self.collected_papers),
                'target_papers': self.target_papers,
                'source': 'OpenAlex snapshot via S3',
                'filters': {
                    'cs_papers_only': True,
                    'year_range': '2015-2024',
                    'language': 'english',
                    'min_abstract_length': 100,
                    'quality_filtered': True
                }
            },
            'papers': self.collected_papers
        }
        
        # Save main dataset
        main_file = output_dir / f"openalex_cs_papers_{len(self.collected_papers)}_{timestamp}.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Generate and save statistics
        stats = self.generate_statistics()
        stats_file = output_dir / f"extraction_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset saved to: {main_file}")
        logger.info(f"Statistics saved to: {stats_file}")
        
        return main_file, stats_file
    
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        if not self.collected_papers:
            return {}
        
        df = pd.DataFrame(self.collected_papers)
        
        stats = {
            'basic_stats': {
                'total_papers': len(df),
                'unique_titles': df['title'].nunique(),
                'duplicate_rate': 1 - (df['title'].nunique() / len(df)),
                'avg_quality_score': float(df['quality_score'].mean())
            },
            'year_distribution': df['year'].value_counts().sort_index().to_dict(),
            'citation_stats': {
                'mean': float(df['citation_count'].mean()),
                'median': float(df['citation_count'].median()),
                'max': int(df['citation_count'].max()),
                'papers_with_10plus_citations': int((df['citation_count'] >= 10).sum()),
                'papers_with_50plus_citations': int((df['citation_count'] >= 50).sum())
            },
            'venue_stats': {
                'unique_venues': df['venue'].nunique(),
                'top_venues': df['venue'].value_counts().head(20).to_dict()
            },
            'text_stats': {
                'avg_title_length': float(df['title'].str.len().mean()),
                'avg_abstract_length': float(df['abstract'].str.len().mean()),
                'avg_author_count': float(df['author_count'].mean())
            },
            'quality_distribution': {
                'high_quality_70plus': int((df['quality_score'] >= 70).sum()),
                'medium_quality_50_69': int(((df['quality_score'] >= 50) & (df['quality_score'] < 70)).sum()),
                'basic_quality_below_50': int((df['quality_score'] < 50).sum())
            }
        }
        
        return stats

def main():
    """Main execution"""
    print("🚀 OPENALEX SNAPSHOT EXTRACTOR")
    print("=" * 50)
    print("Extracting high-quality CS papers for transformer training")
    print()
    
    # Initialize extractor
    extractor = OpenAlexSnapshotExtractor(target_papers=5000, max_files=100)
    
    # Run extraction
    papers = extractor.run_extraction()
    
    if papers:
        # Save dataset
        dataset_file, stats_file = extractor.save_dataset()
        
        # Generate summary
        stats = extractor.generate_statistics()
        
        print("\n" + "=" * 50)
        print("📊 OPENALEX EXTRACTION COMPLETED")
        print("=" * 50)
        print(f"Papers Collected: {len(papers)}")
        print(f"Dataset File: {dataset_file}")
        print(f"Statistics File: {stats_file}")
        
        if stats:
            print(f"\n📈 DATASET QUALITY:")
            print(f"  • Average Quality Score: {stats['basic_stats']['avg_quality_score']:.1f}/100")
            print(f"  • High Quality Papers (70+): {stats['quality_distribution']['high_quality_70plus']}")
            print(f"  • Papers with 50+ Citations: {stats['citation_stats']['papers_with_50plus_citations']}")
            print(f"  • Unique Venues: {stats['venue_stats']['unique_venues']}")
            print(f"  • Year Range: {min(stats['year_distribution'].keys())}-{max(stats['year_distribution'].keys())}")
        
        print(f"\n🎯 READY FOR ADVANCED ARCHITECTURES:")
        print(f"  • Large dataset suitable for transformer fine-tuning")
        print(f"  • High-quality CS papers with abstracts")
        print(f"  • Rich metadata for feature engineering")
        print(f"  • Ready to re-run with larger dataset!")
        
    else:
        print("❌ No papers extracted. Check connection and try again.")

if __name__ == "__main__":
    main()