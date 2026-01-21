#!/usr/bin/env python3
"""
TID Translation Experiment CLI
==============================
Command-line interface for running translation experiments.

Usage:
    python run_experiments.py --words 3 --provider gemini
    python run_experiments.py --words 3 4 5 --limit 5 --output results.json
    python run_experiments.py --all --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root (sign-transcriber/)
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)

from experiments.experiment_runner import ExperimentRunner
from experiments.report_generator import ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TID translation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 3-word experiments with Gemini
    python run_experiments.py --words 3 --provider gemini
    
    # Run all word counts with limit of 5 samples each
    python run_experiments.py --all --limit 5
    
    # Run specific word counts and save results
    python run_experiments.py --words 3 4 5 --output results.json
    
    # Generate markdown report
    python run_experiments.py --all --report benchmark_report.md
        """
    )
    
    parser.add_argument(
        "--words", "-w",
        type=int,
        nargs="+",
        choices=[3, 4, 5],
        help="Word counts to test (3, 4, 5)"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all word counts (3, 4, 5)"
    )
    
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="gemini",
        choices=["gemini", "openai", "claude"],
        help="LLM provider (default: gemini)"
    )
    
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG system (use direct LLM)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of samples per word count"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )
    
    parser.add_argument(
        "--report", "-r",
        type=str,
        help="Output markdown report file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (minimal output)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine word counts to test
    if args.all:
        word_counts = [3, 4, 5]
    elif args.words:
        word_counts = args.words
    else:
        print("Error: Specify --words or --all")
        sys.exit(1)
    
    verbose = not args.quiet
    
    # Initialize runner
    runner = ExperimentRunner(
        provider=args.provider,
        use_rag=not args.no_rag
    )
    
    print(f"\n{'='*60}")
    print("TID Translation Experiment Runner")
    print(f"{'='*60}")
    print(f"Provider: {args.provider}")
    print(f"RAG: {'Enabled' if not args.no_rag else 'Disabled'}")
    print(f"Word counts: {word_counts}")
    if args.limit:
        print(f"Limit: {args.limit} samples per word count")
    print(f"{'='*60}\n")
    
    # Run experiments
    results = runner.run_all(
        word_counts=word_counts,
        limit=args.limit,
        verbose=verbose
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_successful = 0
    total_samples = 0
    
    for word_count, batch in results.items():
        total_successful += batch.successful
        total_samples += batch.total_samples
        print(f"{word_count}-word: {batch.successful}/{batch.total_samples} "
              f"(Avg conf: {batch.avg_confidence}/10, Latency: {batch.avg_latency_ms:.0f}ms)")
    
    print(f"\nTotal: {total_successful}/{total_samples} successful")
    print(f"{'='*60}")
    
    # Save JSON results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "provider": args.provider,
            "rag_enabled": not args.no_rag,
            "results": {
                str(k): v.to_dict() for k, v in results.items()
            }
        }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {args.output}")
    
    # Generate markdown report
    if args.report:
        generator = ReportGenerator()
        report = generator.generate_report(results, provider=args.provider)
        
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"Report saved to: {args.report}")


if __name__ == "__main__":
    main()
