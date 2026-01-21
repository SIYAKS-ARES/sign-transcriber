"""
Report Generator for TID Translation Experiments
================================================
Generates markdown and JSON reports from experiment results.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from experiments.experiment_runner import ExperimentBatchResult, ExperimentResult


class ReportGenerator:
    """
    Generates reports from experiment results.
    
    Supports:
    - Markdown reports for documentation
    - JSON reports for further analysis
    - Summary statistics
    """
    
    def __init__(self):
        self.report_dir = Path(__file__).parent / "reports"
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_report(
        self,
        results: Dict[int, ExperimentBatchResult],
        provider: str = "gemini",
        title: str = "TID Translation Benchmark Report",
    ) -> str:
        """
        Generate a markdown report from experiment results.
        
        Args:
            results: Dictionary mapping word count to batch results
            provider: LLM provider used
            title: Report title
            
        Returns:
            Markdown report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = []
        report.append(f"# {title}")
        report.append(f"\n**Generated:** {timestamp}")
        report.append(f"**Provider:** {provider}")
        report.append(f"**RAG System:** {'Available' if any(r.rag_available for r in results.values()) else 'Not Available'}")
        report.append("")
        
        # Summary table
        report.append("## Summary")
        report.append("")
        report.append("| Word Count | Samples | Successful | Success Rate | Avg Confidence | Avg Latency |")
        report.append("|------------|---------|------------|--------------|----------------|-------------|")
        
        total_samples = 0
        total_successful = 0
        total_confidence = 0
        total_latency = 0
        
        for word_count in sorted(results.keys()):
            batch = results[word_count]
            success_rate = batch.successful / batch.total_samples * 100 if batch.total_samples > 0 else 0
            
            report.append(
                f"| {word_count} kelime | {batch.total_samples} | {batch.successful} | "
                f"{success_rate:.1f}% | {batch.avg_confidence}/10 | {batch.avg_latency_ms:.0f}ms |"
            )
            
            total_samples += batch.total_samples
            total_successful += batch.successful
            total_confidence += batch.avg_confidence * batch.successful
            total_latency += batch.avg_latency_ms * batch.total_samples
        
        # Totals
        overall_success_rate = total_successful / total_samples * 100 if total_samples > 0 else 0
        overall_confidence = total_confidence / total_successful if total_successful > 0 else 0
        overall_latency = total_latency / total_samples if total_samples > 0 else 0
        
        report.append(
            f"| **Toplam** | **{total_samples}** | **{total_successful}** | "
            f"**{overall_success_rate:.1f}%** | **{overall_confidence:.2f}/10** | **{overall_latency:.0f}ms** |"
        )
        report.append("")
        
        # Detailed results per word count
        report.append("## Detailed Results")
        report.append("")
        
        for word_count in sorted(results.keys()):
            batch = results[word_count]
            report.append(f"### {word_count} Kelimelik Cumleler")
            report.append("")
            report.append("| Gloss | Reference | Translation | Confidence | Status |")
            report.append("|-------|-----------|-------------|------------|--------|")
            
            for r in batch.results:
                status = "OK" if not r.error else f"ERROR"
                translation = r.translation[:40] + "..." if len(r.translation) > 40 else r.translation
                gloss_short = r.gloss[:30] + "..." if len(r.gloss) > 30 else r.gloss
                ref_short = r.reference[:30] + "..." if len(r.reference) > 30 else r.reference
                
                report.append(
                    f"| {gloss_short} | {ref_short} | {translation} | {r.confidence}/10 | {status} |"
                )
            
            report.append("")
        
        # Error analysis
        errors = []
        for batch in results.values():
            for r in batch.results:
                if r.error:
                    errors.append({"gloss": r.gloss, "error": r.error})
        
        if errors:
            report.append("## Hata Analizi")
            report.append("")
            report.append(f"Toplam {len(errors)} hata tespit edildi:")
            report.append("")
            
            for e in errors[:10]:  # Show first 10 errors
                report.append(f"- **{e['gloss'][:30]}...**: {e['error']}")
            
            if len(errors) > 10:
                report.append(f"- ... ve {len(errors) - 10} daha fazla hata")
            
            report.append("")
        
        # Observations
        report.append("## Gozlemler")
        report.append("")
        report.append("- Kelime sayisi arttikca performans degisimi analiz edilmeli")
        report.append("- RAG sistemi cevirilerin kalitesini artirabilir")
        report.append("- Hata oranini dusuren faktörler incelenmeli")
        report.append("")
        
        return "\n".join(report)
    
    def save_json_report(
        self,
        results: Dict[int, ExperimentBatchResult],
        filename: str,
        provider: str = "gemini",
    ) -> Path:
        """
        Save results as JSON report.
        
        Args:
            results: Dictionary mapping word count to batch results
            filename: Output filename
            provider: LLM provider used
            
        Returns:
            Path to saved file
        """
        import json
        
        output_path = self.report_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "summary": {
                "total_samples": sum(b.total_samples for b in results.values()),
                "total_successful": sum(b.successful for b in results.values()),
            },
            "results": {
                str(k): v.to_dict() for k, v in results.items()
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path


if __name__ == "__main__":
    # Test report generation with dummy data
    from experiments.experiment_runner import ExperimentRunner
    
    print("Testing report generator with dummy data...")
    
    runner = ExperimentRunner(provider="gemini", use_rag=False)
    
    # Run small test
    results = runner.run_all(word_counts=[3], limit=2, verbose=False)
    
    generator = ReportGenerator()
    report = generator.generate_report(results)
    
    print(report)
