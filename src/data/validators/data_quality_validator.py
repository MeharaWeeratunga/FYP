"""
Data Quality Validation for Research Paper Dataset

This module provides comprehensive validation of collected paper data
to ensure quality and consistency for virality prediction research.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """
    Validates collected paper data for quality and completeness
    
    Checks include:
    - Required field presence
    - Data type validation
    - Range validation
    - Consistency checks
    - Quality metrics
    """
    
    def __init__(self):
        self.validation_results = {
            'total_papers': 0,
            'valid_papers': 0,
            'invalid_papers': 0,
            'warnings': [],
            'errors': [],
            'quality_metrics': {},
            'field_completeness': {},
            'validation_date': datetime.now().isoformat()
        }
    
    def validate_dataset(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate complete dataset
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Validation results with quality metrics
        """
        logger.info(f"Validating dataset with {len(papers)} papers...")
        
        self.validation_results['total_papers'] = len(papers)
        
        if not papers:
            self.validation_results['errors'].append("Empty dataset provided")
            return self.validation_results
        
        # Validate individual papers
        valid_papers = []
        for i, paper in enumerate(papers):
            try:
                is_valid, paper_errors = self.validate_paper(paper)
                
                if is_valid:
                    valid_papers.append(paper)
                else:
                    self.validation_results['errors'].extend([
                        f"Paper {i} ({paper.get('paper_id', 'unknown')}): {error}"
                        for error in paper_errors
                    ])
                    
            except Exception as e:
                self.validation_results['errors'].append(f"Paper {i}: Validation exception - {e}")
        
        self.validation_results['valid_papers'] = len(valid_papers)
        self.validation_results['invalid_papers'] = len(papers) - len(valid_papers)
        
        # Generate quality metrics
        if valid_papers:
            self.validation_results['quality_metrics'] = self.calculate_quality_metrics(valid_papers)
            self.validation_results['field_completeness'] = self.calculate_field_completeness(valid_papers)
        
        # Generate overall quality assessment
        self.validation_results['overall_quality'] = self.assess_overall_quality()
        
        logger.info(f"Validation complete: {len(valid_papers)}/{len(papers)} papers valid")
        
        return self.validation_results
    
    def validate_paper(self, paper: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate individual paper record
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required field validation
        required_fields = ['paper_id', 'title', 'citation_count']
        for field in required_fields:
            if field not in paper or paper[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Data type validation
        if 'paper_id' in paper and not isinstance(paper['paper_id'], str):
            errors.append("paper_id must be string")
        
        if 'title' in paper and not isinstance(paper['title'], str):
            errors.append("title must be string")
        
        if 'citation_count' in paper:
            if not isinstance(paper['citation_count'], int) or paper['citation_count'] < 0:
                errors.append("citation_count must be non-negative integer")
        
        if 'year' in paper and paper['year'] is not None:
            if not isinstance(paper['year'], int) or paper['year'] < 1900 or paper['year'] > 2030:
                errors.append("year must be reasonable integer (1900-2030)")
        
        # Content validation
        if 'title' in paper and paper['title']:
            if len(paper['title']) < 10:
                errors.append("title too short (< 10 characters)")
            elif len(paper['title']) > 500:
                errors.append("title too long (> 500 characters)")
        
        if 'abstract' in paper and paper['abstract']:
            if len(paper['abstract']) < 50:
                errors.append("abstract too short (< 50 characters)")
        
        # Consistency checks
        if 'publication_date' in paper and paper['publication_date']:
            try:
                pub_date = datetime.strptime(paper['publication_date'], '%Y-%m-%d')
                current_date = datetime.now()
                
                if pub_date > current_date:
                    errors.append("publication_date is in the future")
                elif pub_date.year < 1900:
                    errors.append("publication_date too old (before 1900)")
                    
                # Check year consistency
                if 'year' in paper and paper['year'] and paper['year'] != pub_date.year:
                    errors.append("year and publication_date inconsistent")
                    
            except ValueError:
                errors.append("invalid publication_date format (expected YYYY-MM-DD)")
        
        # Citation validation
        if 'citation_count' in paper and 'age_days' in paper:
            citation_count = paper['citation_count']
            age_days = paper.get('age_days', 0)
            
            if age_days > 0:
                citation_velocity = citation_count / (age_days / 30.44)  # citations per month
                
                # Flag unrealistic citation velocities
                if citation_velocity > 1000:  # More than 1000 citations per month
                    errors.append(f"unrealistic citation velocity: {citation_velocity:.1f} citations/month")
        
        # Author validation
        if 'authors' in paper and paper['authors']:
            if not isinstance(paper['authors'], list):
                errors.append("authors must be list")
            elif len(paper['authors']) > 50:  # Unusually large author count
                errors.append(f"unusually large author count: {len(paper['authors'])}")
        
        return len(errors) == 0, errors
    
    def calculate_quality_metrics(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate various quality metrics for the dataset"""
        metrics = {}
        
        if not papers:
            return metrics
        
        # Citation statistics
        citations = [p.get('citation_count', 0) for p in papers]
        metrics['citation_stats'] = {
            'mean': statistics.mean(citations),
            'median': statistics.median(citations),
            'std_dev': statistics.stdev(citations) if len(citations) > 1 else 0,
            'min': min(citations),
            'max': max(citations),
            'total': sum(citations)
        }
        
        # Citation velocity statistics (if available)
        velocities = [p.get('citation_velocity') for p in papers if p.get('citation_velocity') is not None]
        if velocities:
            metrics['citation_velocity_stats'] = {
                'mean': statistics.mean(velocities),
                'median': statistics.median(velocities),
                'std_dev': statistics.stdev(velocities) if len(velocities) > 1 else 0,
                'min': min(velocities),
                'max': max(velocities)
            }
        
        # Temporal distribution
        years = [p.get('year') for p in papers if p.get('year') is not None]
        if years:
            year_counts = {}
            for year in years:
                year_counts[year] = year_counts.get(year, 0) + 1
            metrics['year_distribution'] = year_counts
        
        # Venue diversity
        venues = [p.get('venue') for p in papers if p.get('venue')]
        unique_venues = len(set(venues))
        metrics['venue_diversity'] = {
            'unique_venues': unique_venues,
            'total_papers_with_venue': len(venues),
            'diversity_ratio': unique_venues / len(venues) if venues else 0
        }
        
        # Abstract quality
        abstracts = [p.get('abstract') for p in papers if p.get('abstract')]
        if abstracts:
            abstract_lengths = [len(abstract.split()) for abstract in abstracts]
            metrics['abstract_quality'] = {
                'papers_with_abstract': len(abstracts),
                'avg_length_words': statistics.mean(abstract_lengths),
                'median_length_words': statistics.median(abstract_lengths),
                'min_length_words': min(abstract_lengths),
                'max_length_words': max(abstract_lengths)
            }
        
        # CS paper validation
        cs_indicators = 0
        for paper in papers:
            fields = paper.get('fields_of_study', []) or []
            if any(field in ['Computer Science', 'Machine Learning', 'Artificial Intelligence'] 
                   for field in fields):
                cs_indicators += 1
        
        metrics['cs_validation'] = {
            'papers_with_cs_fields': cs_indicators,
            'cs_ratio': cs_indicators / len(papers)
        }
        
        return metrics
    
    def calculate_field_completeness(self, papers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate completeness statistics for each field"""
        if not papers:
            return {}
        
        fields_to_check = [
            'paper_id', 'title', 'abstract', 'authors', 'venue', 'year',
            'publication_date', 'citation_count', 'reference_count',
            'fields_of_study', 'url', 'open_access_pdf'
        ]
        
        completeness = {}
        total_papers = len(papers)
        
        for field in fields_to_check:
            count = 0
            non_empty_count = 0
            
            for paper in papers:
                if field in paper:
                    count += 1
                    value = paper[field]
                    
                    # Check if value is non-empty
                    if value is not None and value != '' and value != []:
                        non_empty_count += 1
            
            completeness[field] = {
                'present_count': count,
                'present_percentage': (count / total_papers) * 100,
                'non_empty_count': non_empty_count,
                'non_empty_percentage': (non_empty_count / total_papers) * 100
            }
        
        return completeness
    
    def assess_overall_quality(self) -> Dict[str, Any]:
        """Assess overall dataset quality"""
        total = self.validation_results['total_papers']
        valid = self.validation_results['valid_papers']
        
        if total == 0:
            return {'grade': 'F', 'score': 0, 'assessment': 'No data'}
        
        validity_ratio = valid / total
        error_count = len(self.validation_results['errors'])
        warning_count = len(self.validation_results['warnings'])
        
        # Calculate score based on multiple factors
        score = 0
        
        # Validity (40% of score)
        score += validity_ratio * 40
        
        # Error rate (30% of score)
        error_penalty = min(error_count / total, 1.0)  # Cap at 100%
        score += (1 - error_penalty) * 30
        
        # Completeness (30% of score)
        if 'field_completeness' in self.validation_results:
            completeness_scores = []
            critical_fields = ['title', 'abstract', 'citation_count', 'publication_date']
            
            for field in critical_fields:
                if field in self.validation_results['field_completeness']:
                    completeness_scores.append(
                        self.validation_results['field_completeness'][field]['non_empty_percentage']
                    )
            
            if completeness_scores:
                avg_completeness = sum(completeness_scores) / len(completeness_scores)
                score += (avg_completeness / 100) * 30
        
        # Assign grade
        if score >= 90:
            grade = 'A'
            assessment = 'Excellent quality dataset'
        elif score >= 80:
            grade = 'B'
            assessment = 'Good quality dataset'
        elif score >= 70:
            grade = 'C'
            assessment = 'Acceptable quality dataset'
        elif score >= 60:
            grade = 'D'
            assessment = 'Below average quality dataset'
        else:
            grade = 'F'
            assessment = 'Poor quality dataset'
        
        return {
            'grade': grade,
            'score': round(score, 1),
            'assessment': assessment,
            'validity_ratio': round(validity_ratio, 3),
            'error_count': error_count,
            'warning_count': warning_count
        }
    
    def generate_quality_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive quality report"""
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("DATA QUALITY VALIDATION REPORT")
        report_lines.append("=" * 60)
        
        # Overall assessment
        overall = self.validation_results.get('overall_quality', {})
        report_lines.append(f"\nOVERALL QUALITY: {overall.get('grade', 'N/A')} ({overall.get('score', 0)}/100)")
        report_lines.append(f"Assessment: {overall.get('assessment', 'No assessment')}")
        
        # Basic statistics
        report_lines.append(f"\nDATASET STATISTICS:")
        report_lines.append(f"  Total papers: {self.validation_results['total_papers']}")
        report_lines.append(f"  Valid papers: {self.validation_results['valid_papers']}")
        report_lines.append(f"  Invalid papers: {self.validation_results['invalid_papers']}")
        report_lines.append(f"  Validity ratio: {overall.get('validity_ratio', 0):.1%}")
        
        # Error summary
        error_count = len(self.validation_results['errors'])
        warning_count = len(self.validation_results['warnings'])
        
        report_lines.append(f"\nISSUE SUMMARY:")
        report_lines.append(f"  Errors: {error_count}")
        report_lines.append(f"  Warnings: {warning_count}")
        
        # Quality metrics
        if 'quality_metrics' in self.validation_results:
            metrics = self.validation_results['quality_metrics']
            
            if 'citation_stats' in metrics:
                stats = metrics['citation_stats']
                report_lines.append(f"\nCITATION STATISTICS:")
                report_lines.append(f"  Mean: {stats['mean']:.1f}")
                report_lines.append(f"  Median: {stats['median']:.1f}")
                report_lines.append(f"  Range: {stats['min']} - {stats['max']}")
                report_lines.append(f"  Total: {stats['total']:,}")
            
            if 'cs_validation' in metrics:
                cs = metrics['cs_validation']
                report_lines.append(f"\nCS PAPER VALIDATION:")
                report_lines.append(f"  Papers with CS fields: {cs['papers_with_cs_fields']}")
                report_lines.append(f"  CS ratio: {cs['cs_ratio']:.1%}")
        
        # Field completeness
        if 'field_completeness' in self.validation_results:
            report_lines.append(f"\nFIELD COMPLETENESS:")
            completeness = self.validation_results['field_completeness']
            
            for field, stats in completeness.items():
                report_lines.append(f"  {field}: {stats['non_empty_percentage']:.1f}% complete")
        
        # Recommendations
        report_lines.append(f"\nRECOMMENDATIONS:")
        
        if error_count > 0:
            report_lines.append("  - Address data validation errors before proceeding")
        
        if overall.get('score', 0) < 80:
            report_lines.append("  - Consider additional data cleaning and validation")
        
        # Critical field completeness check
        if 'field_completeness' in self.validation_results:
            critical_fields = ['abstract', 'citation_count', 'publication_date']
            for field in critical_fields:
                if field in self.validation_results['field_completeness']:
                    completeness_pct = self.validation_results['field_completeness'][field]['non_empty_percentage']
                    if completeness_pct < 80:
                        report_lines.append(f"  - Improve {field} completeness (currently {completeness_pct:.1f}%)")
        
        report_lines.append(f"\nValidation completed: {self.validation_results['validation_date']}")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Quality report saved to {output_path}")
        
        return report_text

def validate_json_file(file_path: str, save_report: bool = True) -> Dict[str, Any]:
    """
    Validate papers from JSON file
    
    Args:
        file_path: Path to JSON file containing papers
        save_report: Whether to save validation report
        
    Returns:
        Validation results
    """
    logger.info(f"Validating data from {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        return {'error': str(e)}
    
    # Run validation
    validator = DataQualityValidator()
    results = validator.validate_dataset(papers)
    
    # Save report if requested
    if save_report:
        report_path = Path(file_path).parent / f"validation_report_{Path(file_path).stem}.txt"
        validator.generate_quality_report(str(report_path))
    
    return results

def main():
    """Main validation function for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_quality_validator.py <path_to_json_file>")
        return
    
    file_path = sys.argv[1]
    results = validate_json_file(file_path)
    
    if 'error' in results:
        print(f"❌ Validation failed: {results['error']}")
        return
    
    # Print summary
    overall = results.get('overall_quality', {})
    print(f"\n🔍 Validation Results:")
    print(f"📊 Grade: {overall.get('grade', 'N/A')} ({overall.get('score', 0)}/100)")
    print(f"✅ Valid papers: {results['valid_papers']}/{results['total_papers']}")
    print(f"❌ Errors: {len(results['errors'])}")
    
    if results['errors']:
        print(f"\n🚨 First 5 errors:")
        for error in results['errors'][:5]:
            print(f"  - {error}")

if __name__ == "__main__":
    main()