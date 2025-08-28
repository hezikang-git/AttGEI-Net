#!/usr/bin/env python
import os
import argparse
import subprocess
from pathlib import Path
import logging
import datetime

# 设置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def main():
    """主程序，解析参数并运行实验"""
    parser = argparse.ArgumentParser(description="运行作物性状预测实验")
    parser.add_argument("--basedata", type=str, default="basedata", help="basedata目录路径")
    parser.add_argument("--basedata1", type=str, default="basedata1", help="basedata1目录路径")
    parser.add_argument("--testdata", type=str, default="testdata", help="testdata目录路径")
    parser.add_argument("--output", type=str, default="results", help="输出目录路径")
    parser.add_argument("--model", type=str, default="attention", 
                      choices=["deepgxe", "crossattention", "attention"],
                      help="模型类型: deepgxe, crossattention, attention")
    parser.add_argument("--traits", nargs='+', help="指定性状列表进行训练(可选)")
    parser.add_argument("--skip_training", action="store_true", help="跳过训练，只汇总结果")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 记录实验配置
    logging.info(f"实验配置:")
    logging.info(f"- basedata: {args.basedata}")
    logging.info(f"- basedata1: {args.basedata1}")
    logging.info(f"- testdata: {args.testdata}")
    logging.info(f"- 输出目录: {args.output}")
    logging.info(f"- 模型类型: {args.model}")
    
    # 获取所有性状目录
    if args.traits:
        trait_dirs = args.traits
    else:
        trait_dirs = [d for d in os.listdir(args.basedata) 
                     if os.path.isdir(os.path.join(args.basedata, d))]
    
    logging.info(f"将处理以下性状: {trait_dirs}")
    
    if not args.skip_training:
        for trait in trait_dirs:
            # 为每个性状创建单独的输出目录
            trait_output_dir = os.path.join(args.output, trait)
            os.makedirs(trait_output_dir, exist_ok=True)
            
            logging.info(f"开始训练性状: {trait}")
            
            # 构建命令
            cmd = [
                "python", "train_evaluate.py",
                "--basedata", args.basedata,
                "--basedata1", args.basedata1,
                "--testdata", args.testdata,
                "--output", trait_output_dir,
                "--model", args.model,
                "--trait", trait
            ]
            
            # 执行命令
            try:
                subprocess.run(cmd, check=True)
                logging.info(f"性状 {trait} 训练完成")
            except subprocess.CalledProcessError as e:
                logging.error(f"性状 {trait} 训练失败: {e}")
    else:
        logging.info("跳过训练过程，直接汇总结果")
    
    # 汇总所有结果
    logging.info("所有性状训练完成，开始汇总结果")
    summarize_results(args.output, trait_dirs, args.model)

def summarize_results(results_dir, trait_dirs, model_type):
    """汇总所有性状的结果并生成详细报告"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # 创建结果和图表目录
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 设置matplotlib中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 收集所有性状的结果
    all_results = []
    trait_predictions = {}
    
    for trait in trait_dirs:
        # 检查结果文件
        trait_result_file = os.path.join(results_dir, trait, f"{trait}_model.pth")
        trait_pred_file = os.path.join(results_dir, trait, f"{trait}_predictions.csv")
        
        # 加载预测结果
        if os.path.exists(trait_pred_file):
            try:
                pred_df = pd.read_csv(trait_pred_file)
                trait_predictions[trait] = pred_df
            except Exception as e:
                logging.error(f"无法读取性状 {trait} 的预测文件: {e}")
        
        # 从all_traits_results.csv读取结果
        csv_result_file = os.path.join(results_dir, trait, "all_traits_results.csv")
        if os.path.exists(csv_result_file):
            try:
                df = pd.read_csv(csv_result_file)
                all_results.append(df)
            except Exception as e:
                logging.error(f"无法读取性状 {trait} 的CSV结果文件: {e}")
                continue
        
        # 如果CSV文件不存在，尝试从模型文件中读取结果信息
        elif os.path.exists(trait_result_file):
            try:
                import torch
                model_data = torch.load(trait_result_file, map_location='cpu')
                result_dict = model_data.get('results', {})
                if result_dict:
                    result_df = pd.DataFrame([result_dict])
                    all_results.append(result_df)
                else:
                    logging.warning(f"性状 {trait} 的模型文件中没有结果信息")
            except Exception as e:
                logging.error(f"无法读取性状 {trait} 的模型文件: {e}")
    
    # 如果没有收集到任何结果，返回
    if not all_results:
        logging.warning("没有找到任何可用的结果数据")
        return
    
    # 合并所有结果
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # 保存汇总结果CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(results_dir, f"summary_results_{timestamp}.csv")
    combined_results.to_csv(summary_file, index=False)
    
    # 计算平均指标
    avg_are = combined_results['test_are'].mean()
    avg_mse = combined_results['test_mse'].mean()
    avg_pearson = combined_results['test_pearson'].mean()
    
    # 计算每个性状的排名
    combined_results['are_rank'] = combined_results['test_are'].rank()
    combined_results['mse_rank'] = combined_results['test_mse'].rank()
    # 皮尔逊相关系数越高越好，所以用负排名
    combined_results['pearson_rank'] = combined_results['test_pearson'].rank(ascending=False)
    combined_results['avg_rank'] = (combined_results['are_rank'] + 
                                    combined_results['mse_rank'] + 
                                    combined_results['pearson_rank']) / 3
    
    # 打印汇总统计
    logging.info("\n" + "="*50)
    logging.info("====== 所有性状评估汇总 ======")
    logging.info("="*50)
    logging.info(f"模型类型: {model_type}")
    logging.info(f"性状总数: {len(combined_results)}")
    logging.info(f"平均 ARE: {avg_are:.4f}")
    logging.info(f"平均 MSE: {avg_mse:.4f}")
    logging.info(f"平均 Pearson: {avg_pearson:.4f}")
    logging.info(f"汇总结果已保存到: {summary_file}")
    logging.info("="*50 + "\n")
    
    # 输出每个性状的详细结果，按皮尔逊相关系数排序
    logging.info("各性状结果详情 (按皮尔逊相关系数降序排列):")
    sorted_results = combined_results.sort_values(by='test_pearson', ascending=False)
    
    for idx, (_, row) in enumerate(sorted_results.iterrows(), 1):
        logging.info(f"{idx}. 性状: {row['trait']}")
        logging.info(f"   ARE: {row['test_are']:.4f}, MSE: {row['test_mse']:.4f}, Pearson: {row['test_pearson']:.4f}")
        logging.info(f"   样本数 - 训练: {row.get('train_samples', 'N/A')}, 验证: {row.get('val_samples', 'N/A')}, 测试: {row.get('test_samples', 'N/A')}")
    

    
    # 生成文本报告
    report_file = os.path.join(results_dir, f"results_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("====== 作物性状预测结果报告 ======\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"模型类型: {model_type}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"性状总数: {len(combined_results)}\n\n")
        
        f.write("====== 总体性能指标 ======\n")
        f.write(f"平均 ARE: {avg_are:.4f}\n")
        f.write(f"平均 MSE: {avg_mse:.4f}\n")
        f.write(f"平均 Pearson: {avg_pearson:.4f}\n\n")
        
        f.write("====== 性能分布 ======\n")
        f.write(f"ARE - 最小值: {combined_results['test_are'].min():.4f}, 最大值: {combined_results['test_are'].max():.4f}, 中位数: {combined_results['test_are'].median():.4f}\n")
        f.write(f"MSE - 最小值: {combined_results['test_mse'].min():.4f}, 最大值: {combined_results['test_mse'].max():.4f}, 中位数: {combined_results['test_mse'].median():.4f}\n")
        f.write(f"Pearson - 最小值: {combined_results['test_pearson'].min():.4f}, 最大值: {combined_results['test_pearson'].max():.4f}, 中位数: {combined_results['test_pearson'].median():.4f}\n\n")
        
        f.write("====== 各性状详细结果 ======\n")
        for _, row in sorted_results.iterrows():
            f.write(f"- 性状: {row['trait']}, 模型: {row.get('model_type', model_type)}\n")
            f.write(f"  ARE: {row['test_are']:.4f}, MSE: {row['test_mse']:.4f}, Pearson: {row['test_pearson']:.4f}\n")
            f.write(f"  训练样本: {row.get('train_samples', 'N/A')}, 验证样本: {row.get('val_samples', 'N/A')}, 测试样本: {row.get('test_samples', 'N/A')}\n\n")
    
    logging.info(f"文本报告已保存到: {report_file}")
    
    # 生成可视化图表
    try:
        # 1. 性能指标对比图
        plt.figure(figsize=(15, 10))
        
        # 绘制ARE柱状图
        plt.subplot(3, 1, 1)
        sns.barplot(x='trait', y='test_are', data=sorted_results)
        plt.title('各性状ARE指标对比')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制MSE柱状图
        plt.subplot(3, 1, 2)
        sns.barplot(x='trait', y='test_mse', data=sorted_results)
        plt.title('各性状MSE指标对比')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制Pearson柱状图
        plt.subplot(3, 1, 3)
        sns.barplot(x='trait', y='test_pearson', data=sorted_results)
        plt.title('各性状Pearson相关系数对比')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        metrics_plot_file = os.path.join(plots_dir, f"metrics_comparison_{timestamp}.png")
        plt.savefig(metrics_plot_file, dpi=300)
        plt.close()
        
        # 2. 指标分布图
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.histplot(combined_results['test_are'], kde=True)
        plt.title('ARE分布')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 3, 2)
        sns.histplot(combined_results['test_mse'], kde=True)
        plt.title('MSE分布')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 3, 3)
        sns.histplot(combined_results['test_pearson'], kde=True)
        plt.title('Pearson相关系数分布')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        dist_plot_file = os.path.join(plots_dir, f"metrics_distribution_{timestamp}.png")
        plt.savefig(dist_plot_file, dpi=300)
        plt.close()
        
        # 3. 为每个性状生成预测vs真实值散点图
        for trait, pred_df in trait_predictions.items():
            plt.figure(figsize=(8, 8))
            sns.scatterplot(x='true', y='pred', data=pred_df)
            
            # 添加对角线 (理想预测线)
            min_val = min(pred_df['true'].min(), pred_df['pred'].min())
            max_val = max(pred_df['true'].max(), pred_df['pred'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # 添加皮尔逊相关系数
            pearson = sorted_results[sorted_results['trait'] == trait]['test_pearson'].values
            if len(pearson) > 0:
                plt.title(f'{trait} - 预测值 vs 真实值 (Pearson r = {pearson[0]:.4f})')
            else:
                plt.title(f'{trait} - 预测值 vs 真实值')
            
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            pred_plot_file = os.path.join(plots_dir, f"{trait}_predictions_{timestamp}.png")
            plt.savefig(pred_plot_file, dpi=300)
            plt.close()
        
        logging.info(f"可视化图表已保存到: {plots_dir}")
        
    except Exception as e:
        logging.error(f"生成可视化图表时出错: {e}")

def generate_html_report(results_df, trait_predictions, html_file, model_type):
    """生成HTML格式的结果报告"""
    import pandas as pd
    from datetime import datetime
    
    # 按皮尔逊相关系数排序
    sorted_results = results_df.sort_values(by='test_pearson', ascending=False)
    
    # 计算总体平均指标
    avg_are = results_df['test_are'].mean()
    avg_mse = results_df['test_mse'].mean()
    avg_pearson = results_df['test_pearson'].mean()
    
    # 生成HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>作物性状预测结果报告</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid #3498db;
            }}
            .summary {{
                background-color: #f1f8ff;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #e3f2fd;
            }}
            .good {{
                color: green;
                font-weight: bold;
            }}
            .average {{
                color: orange;
            }}
            .poor {{
                color: red;
            }}
            .timestamp {{
                text-align: right;
                color: #777;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>作物性状预测结果报告</h1>
            <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>总体性能摘要</h2>
                <p><strong>模型类型:</strong> {model_type}</p>
                <p><strong>性状总数:</strong> {len(results_df)}</p>
                <p><strong>平均 ARE:</strong> {avg_are:.4f}</p>
                <p><strong>平均 MSE:</strong> {avg_mse:.4f}</p>
                <p><strong>平均 Pearson相关系数:</strong> {avg_pearson:.4f}</p>
            </div>
            
            <h2>各性状性能指标 (按Pearson相关系数降序排列)</h2>
            <table>
                <tr>
                    <th>排名</th>
                    <th>性状</th>
                    <th>ARE</th>
                    <th>MSE</th>
                    <th>Pearson相关系数</th>
                    <th>训练样本</th>
                    <th>验证样本</th>
                    <th>测试样本</th>
                </tr>
    """
    
    # 添加每个性状的结果行
    for idx, (_, row) in enumerate(sorted_results.iterrows(), 1):
        # 根据皮尔逊相关系数设置颜色类
        if row['test_pearson'] > 0.8:
            pearson_class = "good"
        elif row['test_pearson'] > 0.6:
            pearson_class = "average"
        else:
            pearson_class = "poor"
            
        html_content += f"""
                <tr>
                    <td>{idx}</td>
                    <td><strong>{row['trait']}</strong></td>
                    <td>{row['test_are']:.4f}</td>
                    <td>{row['test_mse']:.4f}</td>
                    <td class="{pearson_class}">{row['test_pearson']:.4f}</td>
                    <td>{row.get('train_samples', 'N/A')}</td>
                    <td>{row.get('val_samples', 'N/A')}</td>
                    <td>{row.get('test_samples', 'N/A')}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>性能分布统计</h2>
            <table>
                <tr>
                    <th>指标</th>
                    <th>最小值</th>
                    <th>最大值</th>
                    <th>平均值</th>
                    <th>中位数</th>
                    <th>标准差</th>
                </tr>
    """
    
    # 添加性能分布统计
    metrics = ['test_are', 'test_mse', 'test_pearson']
    metric_names = ['ARE', 'MSE', 'Pearson相关系数']
    
    for metric, name in zip(metrics, metric_names):
        html_content += f"""
                <tr>
                    <td><strong>{name}</strong></td>
                    <td>{results_df[metric].min():.4f}</td>
                    <td>{results_df[metric].max():.4f}</td>
                    <td>{results_df[metric].mean():.4f}</td>
                    <td>{results_df[metric].median():.4f}</td>
                    <td>{results_df[metric].std():.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>结论与建议</h2>
            <p>基于以上结果分析，可以得出以下结论：</p>
            <ul>
                <li>性能较好的性状（Pearson > 0.8）可能具有更强的遗传决定性或环境响应模式更加稳定。</li>
                <li>性能较差的性状可能需要更复杂的模型或更多的训练数据。</li>
                <li>建议对Pearson相关系数低于0.6的性状进一步优化模型或收集更多数据。</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    main() 