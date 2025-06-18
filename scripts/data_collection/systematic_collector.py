"""
Systematic Data Collection Script for Research Paper Virality Project

This script executes the systematic collection plan defined in TRACKER.md
with real-time progress updates and quality monitoring.
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.collectors.production_collector import ProductionCollector, CollectionConfig, PaperRecord
from src.data.validators.data_quality_validator import DataQualityValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystematicCollector:
    """
    Executes systematic data collection with progress tracking
    """
    
    def __init__(self):
        # Collection phases defined in TRACKER.md
        self.collection_phases = {
            "Phase 1: Major ML Conferences": {
                "target": 150,
                "venues": [
                    ("ICML", [2022, 2023], 30),  # (venue, years, papers_per_year)
                    ("NeurIPS", [2022, 2023], 40),
                    ("ICLR", [2022, 2023], 20)
                ]
            },
            "Phase 2: Computer Vision Conferences": {
                "target": 100,
                "venues": [
                    ("CVPR", [2022, 2023], 25),
                    ("ICCV", [2022], 25),  # ICCV is biannual
                    ("ECCV", [2022], 25)   # ECCV is biannual
                ]
            },
            "Phase 3: NLP Conferences": {
                "target": 100,
                "venues": [
                    ("ACL", [2022, 2023], 25),
                    ("EMNLP", [2022, 2023], 25)
                ]
            },
            "Phase 4: Systems & HCI Conferences": {
                "target": 50,
                "venues": [
                    ("CHI", [2022, 2023], 15),
                    ("OSDI", [2022, 2023], 10)
                ]
            }
        }
        
        # Initialize collector with optimized config
        self.config = CollectionConfig(
            rate_limit_seconds=3.0,  # Conservative rate limiting
            papers_per_venue=50,     # Increased for systematic collection
            papers_per_keyword=0,    # Disable keyword search for systematic collection
            min_citation_count=10,   # Focus on impactful papers
            target_years=[2022, 2023]  # Optimal for early prediction
        )
        
        self.collector = ProductionCollector(self.config)
        self.validator = DataQualityValidator()
        
        # Progress tracking
        self.total_collected = 0
        self.phase_progress = {}
        self.collection_start_time = datetime.now()
        
        # Output paths
        self.output_dir = Path("data/raw/systematic_collection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker_file = Path("TRACKER.md")
        
    def update_tracker(self, phase_name: str, venue: str, year: int, papers_collected: int):
        """Update TRACKER.md with real-time progress"""
        try:
            # Read current tracker
            if self.tracker_file.exists():
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update the specific venue line
                search_pattern = f"- [ ] **{venue} {year}** (0/"
                replace_pattern = f"- [{'x' if papers_collected > 0 else ' '}] **{venue} {year}** ({papers_collected}/"
                
                # If the pattern exists, replace it
                if search_pattern in content:
                    content = content.replace(search_pattern, replace_pattern)
                
                # Update overall progress
                old_progress_pattern = f"### **Collection Status**: {self.total_collected - papers_collected}/"
                new_progress_pattern = f"### **Collection Status**: {self.total_collected}/"
                if old_progress_pattern in content:
                    content = content.replace(old_progress_pattern, new_progress_pattern)
                
                # Write back
                with open(self.tracker_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                logger.info(f"📊 Updated tracker: {venue} {year} -> {papers_collected} papers")
                
        except Exception as e:
            logger.error(f"Failed to update tracker: {e}")
    
    def collect_venue_papers(self, venue: str, year: int, target_count: int) -> list:
        """Collect papers from specific venue and year"""
        logger.info(f"\n🎯 Starting {venue} {year} collection (target: {target_count} papers)")
        
        papers = []
        offset = 0
        batch_size = 20
        
        while len(papers) < target_count:
            try:
                # Use the production collector's venue collection method
                batch_papers = self.collector.collect_papers_by_venue(venue, year)
                
                # Filter and add new papers
                for paper in batch_papers:
                    if len(papers) >= target_count:
                        break
                    
                    # Avoid duplicates
                    if paper.paper_id not in {p.paper_id for p in papers}:
                        papers.append(paper)
                
                # If no new papers found, break
                if len(batch_papers) == 0:
                    logger.warning(f"No more papers found for {venue} {year}")
                    break
                
                # Progress update
                logger.info(f"  📄 {venue} {year}: {len(papers)}/{target_count} papers collected")
                
                # Update tracker every 10 papers
                if len(papers) % 10 == 0 or len(papers) >= target_count:
                    self.update_tracker("Phase", venue, year, len(papers))
                
            except Exception as e:
                logger.error(f"Error collecting from {venue} {year}: {e}")
                break
        
        final_count = len(papers)
        logger.info(f"✅ {venue} {year} complete: {final_count}/{target_count} papers")
        
        # Final tracker update
        self.update_tracker("Phase", venue, year, final_count)
        self.total_collected += final_count
        
        return papers
    
    def validate_collection_quality(self, papers: list) -> dict:
        """Validate quality of collected papers"""
        logger.info("\n🔍 Running quality validation...")
        
        # Convert PaperRecord objects to dictionaries for validation
        papers_dict = [
            {
                'paper_id': p.paper_id,
                'title': p.title,
                'abstract': p.abstract,
                'authors': p.authors,
                'venue': p.venue,
                'year': p.year,
                'publication_date': p.publication_date,
                'citation_count': p.citation_count,
                'reference_count': p.reference_count,
                'fields_of_study': p.fields_of_study,
                'is_cs_paper': p.is_cs_paper,
                'age_days': p.age_days,
                'citation_velocity': p.citation_velocity
            }
            for p in papers
        ]
        
        validation_results = self.validator.validate_dataset(papers_dict)
        
        # Print quality summary
        overall = validation_results.get('overall_quality', {})
        logger.info(f"📊 Quality Grade: {overall.get('grade', 'N/A')} ({overall.get('score', 0)}/100)")
        logger.info(f"✅ Valid papers: {validation_results['valid_papers']}/{validation_results['total_papers']}")
        
        # Check CS paper percentage
        if 'quality_metrics' in validation_results:
            cs_metrics = validation_results['quality_metrics'].get('cs_validation', {})
            cs_ratio = cs_metrics.get('cs_ratio', 0)
            logger.info(f"🖥️ CS papers: {cs_ratio:.1%}")
        
        return validation_results
    
    def save_collection_results(self, papers: list, phase_name: str) -> str:
        """Save collection results with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save papers
        filename = self.output_dir / f"papers_{phase_name.lower().replace(' ', '_')}_{timestamp}.json"
        papers_data = [
            {
                'paper_id': p.paper_id,
                'title': p.title,
                'abstract': p.abstract,
                'authors': p.authors,
                'venue': p.venue,
                'year': p.year,
                'publication_date': p.publication_date,
                'citation_count': p.citation_count,
                'reference_count': p.reference_count,
                'fields_of_study': p.fields_of_study,
                'is_cs_paper': p.is_cs_paper,
                'age_days': p.age_days,
                'citation_velocity': p.citation_velocity,
                'collection_date': p.collection_date,
                'data_source': p.data_source
            }
            for p in papers
        ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(papers)} papers to {filename}")
        return str(filename)
    
    def execute_systematic_collection(self):
        """Execute the complete systematic collection plan"""
        logger.info("🚀 Starting Systematic Data Collection")
        logger.info(f"📅 Started at: {self.collection_start_time}")
        logger.info(f"🎯 Target: 400+ papers from top CS venues")
        
        all_papers = []
        
        for phase_name, phase_config in self.collection_phases.items():
            logger.info(f"\n" + "="*50)
            logger.info(f"🎯 {phase_name}")
            logger.info(f"Target: {phase_config['target']} papers")
            logger.info("="*50)
            
            phase_papers = []
            
            for venue, years, papers_per_year in phase_config['venues']:
                for year in years:
                    try:
                        venue_papers = self.collect_venue_papers(venue, year, papers_per_year)
                        phase_papers.extend(venue_papers)
                        
                        # Save intermediate results
                        if len(all_papers + phase_papers) % 50 == 0:
                            temp_filename = self.save_collection_results(
                                all_papers + phase_papers, 
                                f"intermediate_{len(all_papers + phase_papers)}"
                            )
                            logger.info(f"💾 Intermediate save: {len(all_papers + phase_papers)} papers")
                        
                    except Exception as e:
                        logger.error(f"Failed to collect {venue} {year}: {e}")
                        continue
            
            # Validate phase quality
            if phase_papers:
                logger.info(f"\n📊 {phase_name} Quality Check:")
                validation_results = self.validate_collection_quality(phase_papers)
                
                # Save phase results
                phase_filename = self.save_collection_results(phase_papers, phase_name)
                
                # Add to total collection
                all_papers.extend(phase_papers)
                
                logger.info(f"✅ {phase_name} complete: {len(phase_papers)} papers")
                logger.info(f"📈 Total collected so far: {len(all_papers)} papers")
        
        # Final validation and save
        logger.info("\n" + "="*50)
        logger.info("🏁 COLLECTION COMPLETE")
        logger.info("="*50)
        
        if all_papers:
            # Final quality validation
            final_validation = self.validate_collection_quality(all_papers)
            
            # Save final dataset
            final_filename = self.save_collection_results(all_papers, "final_dataset")
            
            # Collection summary
            collection_time = datetime.now() - self.collection_start_time
            
            logger.info(f"📊 Final Results:")
            logger.info(f"  📄 Total papers: {len(all_papers)}")
            logger.info(f"  ⏱️ Collection time: {collection_time}")
            logger.info(f"  📈 Rate: {len(all_papers) / (collection_time.total_seconds() / 3600):.1f} papers/hour")
            logger.info(f"  💾 Final dataset: {final_filename}")
            
            # Update final tracker
            with open("TRACKER.md", "r") as f:
                content = f.read()
            
            # Update overall status
            content = content.replace(
                "### **Collection Status**: 8/400 papers (2.0%)",
                f"### **Collection Status**: {len(all_papers)}/400 papers ({len(all_papers)/4:.1f}%)"
            )
            
            with open("TRACKER.md", "w") as f:
                f.write(content)
            
            return all_papers, final_filename
        
        else:
            logger.error("❌ No papers collected!")
            return [], None

def main():
    """Main collection function"""
    try:
        collector = SystematicCollector()
        papers, filename = collector.execute_systematic_collection()
        
        if papers:
            print(f"\n🎉 SUCCESS!")
            print(f"📊 Collected {len(papers)} papers")
            print(f"💾 Saved to: {filename}")
            print(f"📋 Check TRACKER.md for detailed progress")
        else:
            print(f"\n❌ Collection failed - check logs")
            
    except KeyboardInterrupt:
        print("\n⏹️ Collection interrupted by user")
        print("💾 Progress has been saved to TRACKER.md")
        print("🔄 You can resume by running this script again")
        
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        logging.exception("Collection failed with exception")

if __name__ == "__main__":
    main()