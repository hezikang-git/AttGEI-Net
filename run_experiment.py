#!/usr/bin/env python
import os
import subprocess
import logging
import argparse
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"experiment_log_{timestamp}.log"),
        logging.StreamHandler()
    ]
)

def main():
    """Main program, parse arguments and run experiment"""
    parser = argparse.ArgumentParser(description="Run crop trait prediction experiment")
    parser.add_argument("--basedata", type=str, default="basedata", help="basedata directory path")
    parser.add_argument("--basedata1", type=str, default="basedata1", help="basedata1 directory path")
    parser.add_argument("--testdata", type=str, default="testdata", help="testdata directory path")
    parser.add_argument("--output", type=str, default="results", help="Output directory path")
    parser.add_argument("--model", type=str, default="attgeinet", help="Model type (only AttGEI-Net is supported)")
    parser.add_argument("--traits", nargs='+', help="Specify traits list for training (optional)")
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only summarize results")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Log experiment configuration
    logging.info(f"Experiment configuration:")
    logging.info(f"- basedata: {args.basedata}")
    logging.info(f"- basedata1: {args.basedata1}")
    logging.info(f"- testdata: {args.testdata}")
    logging.info(f"- Output directory: {args.output}")
    logging.info(f"- Model type: {args.model} (AttGEI-Net)")
    
    # Get all trait directories
    if args.traits:
        trait_dirs = args.traits
    else:
        trait_dirs = [d for d in os.listdir(args.basedata) 
                     if os.path.isdir(os.path.join(args.basedata, d))]
    
    logging.info(f"Will process the following traits: {trait_dirs}")
    
    if not args.skip_training:
        for trait in trait_dirs:
            # Create separate output directory for each trait
            trait_output_dir = os.path.join(args.output, trait)
            os.makedirs(trait_output_dir, exist_ok=True)
            
            logging.info(f"Starting to train trait: {trait}")
            
            # Build command
            cmd = [
                "python", "train_evaluate.py",
                "--basedata", args.basedata,
                "--basedata1", args.basedata1,
                "--testdata", args.testdata,
                "--output", trait_output_dir,
                "--model", args.model,
                "--trait", trait
            ]
            
            # Execute command
            try:
                subprocess.run(cmd, check=True)
                logging.info(f"Trait {trait} training completed")
            except subprocess.CalledProcessError as e:
                logging.error(f"Trait {trait} training failed: {e}")
    else:
        logging.info("Skipping training process, directly summarizing results")
    
    # Summarize all results
    logging.info("All traits training completed, starting to summarize results")
    summarize_results(args.output, trait_dirs, args.model)

def summarize_results(results_dir, trait_dirs, model_type):
    """Summarize all trait results and generate detailed report"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # Create results and plots directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Configure matplotlib for Chinese font support
    plt.rcParams['font.sans-serif'] = ['SimHei']  # For normal display of Chinese labels
    plt.rcParams['axes.unicode_minus'] = False  # For normal display of minus sign
    
    # Collect results for all traits
    all_results = []
    trait_predictions = {}
    
    for trait in trait_dirs:
        # Check result files
        trait_result_file = os.path.join(results_dir, trait, f"{trait}_model.pth")
        trait_pred_file = os.path.join(results_dir, trait, f"{trait}_predictions.csv")
        
        # Load prediction results
        if os.path.exists(trait_pred_file):
            try:
                pred_df = pd.read_csv(trait_pred_file)
                trait_predictions[trait] = pred_df
            except Exception as e:
                logging.error(f"Cannot read prediction file for trait {trait}: {e}")
        
        # Read results from all_traits_results.csv
        csv_result_file = os.path.join(results_dir, trait, "all_traits_results.csv")
        if os.path.exists(csv_result_file):
            try:
                df = pd.read_csv(csv_result_file)
                all_results.append(df)
            except Exception as e:
                logging.error(f"Cannot read results file for trait {trait}: {e}")
    
    # If no results found
    if not all_results:
        logging.error("No results found for any trait!")
        return
    
    # Merge all results
    combined_results = pd.concat(all_results)
    
    # Calculate aggregate statistics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary results
    summary_file = os.path.join(results_dir, f"summary_results_{timestamp}.csv")
    combined_results.to_csv(summary_file, index=False)
    logging.info(f"Summary results saved to {summary_file}")
    
    # Calculate average metrics
    avg_are = combined_results['test_are'].mean()
    avg_mse = combined_results['test_mse'].mean()
    avg_pearson = combined_results['test_pearson'].mean()
    
    logging.info(f"Overall average metrics: ARE={avg_are:.4f}, MSE={avg_mse:.4f}, Pearson={avg_pearson:.4f}")
    
    # Create text report
    report_file = os.path.join(results_dir, f"results_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write("Crop Trait Prediction Results Report\n")
        f.write("==================================\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Number of traits: {len(trait_dirs)}\n\n")
        
        f.write("Overall Results:\n")
        f.write(f"- Average ARE: {avg_are:.4f}\n")
        f.write(f"- Average MSE: {avg_mse:.4f}\n")
        f.write(f"- Average Pearson correlation: {avg_pearson:.4f}\n\n")
        
        f.write("Results by Trait:\n")
        f.write("----------------\n\n")
        
        # Group by trait
        trait_groups = combined_results.groupby('trait')
        for trait, group in trait_groups:
            avg_trait_are = group['test_are'].mean()
            avg_trait_mse = group['test_mse'].mean()
            avg_trait_pearson = group['test_pearson'].mean()
            
            f.write(f"Trait: {trait}\n")
            f.write(f"- Average ARE: {avg_trait_are:.4f}\n")
            f.write(f"- Average MSE: {avg_trait_mse:.4f}\n")
            f.write(f"- Average Pearson correlation: {avg_trait_pearson:.4f}\n")
            
            if 'test_location' in group.columns:
                f.write("\n  Location-specific results:\n")
                loc_groups = group.groupby('test_location')
                for loc, loc_group in loc_groups:
                    avg_loc_are = loc_group['test_are'].mean()
                    avg_loc_mse = loc_group['test_mse'].mean()
                    avg_loc_pearson = loc_group['test_pearson'].mean()
                    
                    f.write(f"  - Location {loc}: ARE={avg_loc_are:.4f}, MSE={avg_loc_mse:.4f}, Pearson={avg_loc_pearson:.4f}\n")
            
            f.write("\n")
        
        f.write("\nMetrics Distribution:\n")
        f.write("-------------------\n\n")
        
        # Calculate distribution statistics for each metric
        metrics = ['test_are', 'test_mse', 'test_pearson']
        metric_names = ['ARE', 'MSE', 'Pearson']
        
        for metric, name in zip(metrics, metric_names):
            values = combined_results[metric].dropna()
            f.write(f"{name}:\n")
            f.write(f"- Min: {values.min():.4f}\n")
            f.write(f"- Max: {values.max():.4f}\n")
            f.write(f"- Mean: {values.mean():.4f}\n")
            f.write(f"- Median: {values.median():.4f}\n")
            f.write(f"- Standard deviation: {values.std():.4f}\n\n")
    
    logging.info(f"Text report generated at {report_file}")
    
    # Create visualization plots
    try:
        # 1. Distribution of metrics
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.histplot(combined_results['test_are'], kde=True)
        plt.title('ARE Distribution')
        plt.xlabel('ARE')
        
        plt.subplot(1, 3, 2)
        sns.histplot(combined_results['test_mse'], kde=True)
        plt.title('MSE Distribution')
        plt.xlabel('MSE')
        
        plt.subplot(1, 3, 3)
        sns.histplot(combined_results['test_pearson'], kde=True)
        plt.title('Pearson Correlation Distribution')
        plt.xlabel('Pearson')
        
        plt.tight_layout()
        metrics_dist_file = os.path.join(plots_dir, f"metrics_distribution_{timestamp}.png")
        plt.savefig(metrics_dist_file)
        logging.info(f"Metrics distribution plot saved to {metrics_dist_file}")
        
        # 2. Comparison of metrics across traits
        if len(trait_dirs) > 1:
            # Aggregate by trait
            trait_metrics = combined_results.groupby('trait').agg({
                'test_are': 'mean',
                'test_mse': 'mean',
                'test_pearson': 'mean'
            }).reset_index()
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            sns.barplot(x='trait', y='test_are', data=trait_metrics)
            plt.title('Average ARE by Trait')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 2)
            sns.barplot(x='trait', y='test_mse', data=trait_metrics)
            plt.title('Average MSE by Trait')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 3)
            sns.barplot(x='trait', y='test_pearson', data=trait_metrics)
            plt.title('Average Pearson Correlation by Trait')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            metrics_comp_file = os.path.join(plots_dir, f"metrics_comparison_{timestamp}.png")
            plt.savefig(metrics_comp_file)
            logging.info(f"Metrics comparison plot saved to {metrics_comp_file}")
        
        # 3. Individual trait prediction plots
        for trait, pred_df in trait_predictions.items():
            plt.figure(figsize=(8, 6))
            plt.scatter(pred_df['true'], pred_df['pred'], alpha=0.5)
            
            # Add identity line
            min_val = min(pred_df['true'].min(), pred_df['pred'].min())
            max_val = max(pred_df['true'].max(), pred_df['pred'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Calculate metrics for this plot
            corr = np.corrcoef(pred_df['true'], pred_df['pred'])[0, 1]
            mse = ((pred_df['true'] - pred_df['pred']) ** 2).mean()
            
            plt.title(f"{trait} Predictions (Pearson: {corr:.4f}, MSE: {mse:.4f})")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.grid(True, alpha=0.3)
            
            pred_plot_file = os.path.join(plots_dir, f"{trait}_predictions_{timestamp}.png")
            plt.savefig(pred_plot_file)
            logging.info(f"Prediction plot for {trait} saved to {pred_plot_file}")
            plt.close()
            
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    main()