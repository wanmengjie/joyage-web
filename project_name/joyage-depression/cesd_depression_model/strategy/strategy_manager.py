"""
系统化策略管理框架
Strategy Management Framework for CESD Depression Prediction
"""

import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class StrategyType(Enum):
    """策略类型枚举"""
    BASELINE = "baseline"
    FEATURE_SELECTION = "feature_selection"  
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ENSEMBLE_OPTIMIZATION = "ensemble_optimization"
    CROSS_VALIDATION = "cross_validation"
    EXTERNAL_VALIDATION = "external_validation"
    FULL_PIPELINE = "full_pipeline"

class Priority(Enum):
    """优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class StrategyConfig:
    """策略配置数据类"""
    name: str
    description: str
    strategy_type: StrategyType
    priority: Priority
    estimated_time: int  # 分钟
    dependencies: List[str]  # 依赖的策略
    parameters: Dict
    expected_improvement: float  # 预期性能提升百分比
    resource_requirement: str  # 资源需求（低/中/高）
    
class StrategyResult:
    """策略执行结果"""
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.start_time = None
        self.end_time = None
        self.duration = 0
        self.success = False
        self.metrics = {}
        self.artifacts = []  # 生成的文件
        self.error_message = None
        self.performance_gain = 0
        
    def to_dict(self):
        return {
            'strategy_name': self.strategy_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'success': self.success,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'error_message': self.error_message,
            'performance_gain': self.performance_gain
        }

class StrategyManager:
    """策略管理器 - 系统化管理所有优化策略"""
    
    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir)
        self.strategies = {}
        self.execution_history = []
        self.current_baseline = None
        self.best_strategy = None
        self.results_dir = self.workspace_dir / "strategy_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化预定义策略
        self._initialize_strategies()
        
    def _initialize_strategies(self):
        """初始化预定义的优化策略"""
        
        # 1. 基线策略
        self.add_strategy(StrategyConfig(
            name="baseline",
            description="基线策略：全特征+默认参数+标准验证",
            strategy_type=StrategyType.BASELINE,
            priority=Priority.HIGH,
            estimated_time=5,
            dependencies=[],
            parameters={
                'use_feature_selection': False,
                'use_hyperparameter_tuning': False,
                'use_cross_validation': True,
                'cv_folds': 5
            },
            expected_improvement=0.0,
            resource_requirement="低"
        ))
        
        # 2. 特征选择策略
        self.add_strategy(StrategyConfig(
            name="feature_selection_conservative",
            description="保守特征选择：多方法集成选择关键特征",
            strategy_type=StrategyType.FEATURE_SELECTION,
            priority=Priority.HIGH,
            estimated_time=8,
            dependencies=["baseline"],
            parameters={
                'methods': ['variance', 'univariate', 'rfe'],
                'k_best': 20,
                'variance_threshold': 0.01,
                'use_ensemble_voting': True
            },
            expected_improvement=2.5,
            resource_requirement="中"
        ))
        
        # 3. 激进特征选择策略
        self.add_strategy(StrategyConfig(
            name="feature_selection_aggressive",
            description="激进特征选择：深度特征筛选",
            strategy_type=StrategyType.FEATURE_SELECTION,
            priority=Priority.MEDIUM,
            estimated_time=12,
            dependencies=["baseline"],
            parameters={
                'methods': ['variance', 'univariate', 'rfe', 'model_based'],
                'k_best': 15,
                'variance_threshold': 0.05,
                'use_ensemble_voting': True,
                'recursive_elimination': True
            },
            expected_improvement=3.8,
            resource_requirement="中"
        ))
        
        # 4. 轻量级参数调优
        self.add_strategy(StrategyConfig(
            name="hyperparameter_tuning_light",
            description="轻量级参数调优：快速参数搜索",
            strategy_type=StrategyType.HYPERPARAMETER_TUNING,
            priority=Priority.HIGH,
            estimated_time=15,
            dependencies=["baseline"],
            parameters={
                'search_method': 'random',
                'n_iter': 10,
                'cv_folds': 3,
                'models': ['rf', 'gb', 'xgb']
            },
            expected_improvement=3.2,
            resource_requirement="中"
        ))
        
        # 5. 深度参数调优
        self.add_strategy(StrategyConfig(
            name="hyperparameter_tuning_deep",
            description="深度参数调优：全面参数搜索",
            strategy_type=StrategyType.HYPERPARAMETER_TUNING,
            priority=Priority.MEDIUM,
            estimated_time=45,
            dependencies=["baseline"],
            parameters={
                'search_method': 'random',
                'n_iter': 50,
                'cv_folds': 5,
                'models': ['rf', 'gb', 'xgb', 'lgb', 'lr', 'svm']
            },
            expected_improvement=5.1,
            resource_requirement="高"
        ))
        
        # 6. 双重优化策略
        self.add_strategy(StrategyConfig(
            name="double_optimization",
            description="双重优化：特征选择+参数调优",
            strategy_type=StrategyType.ENSEMBLE_OPTIMIZATION,
            priority=Priority.HIGH,
            estimated_time=25,
            dependencies=["feature_selection_conservative", "hyperparameter_tuning_light"],
            parameters={
                'feature_selection': {
                    'methods': ['variance', 'univariate', 'rfe'],
                    'k_best': 20
                },
                'hyperparameter_tuning': {
                    'search_method': 'random',
                    'n_iter': 15
                }
            },
            expected_improvement=4.8,
            resource_requirement="中"
        ))
        
        # 7. 终极优化策略
        self.add_strategy(StrategyConfig(
            name="ultimate_optimization",
            description="终极优化：最全面的优化组合",
            strategy_type=StrategyType.ENSEMBLE_OPTIMIZATION,
            priority=Priority.MEDIUM,
            estimated_time=60,
            dependencies=["feature_selection_aggressive", "hyperparameter_tuning_deep"],
            parameters={
                'feature_selection': {
                    'methods': ['variance', 'univariate', 'rfe', 'model_based'],
                    'k_best': 15
                },
                'hyperparameter_tuning': {
                    'search_method': 'random',
                    'n_iter': 30
                },
                'ensemble_methods': True,
                'stacking_optimization': True
            },
            expected_improvement=7.5,
            resource_requirement="高"
        ))
        
        # 8. 外部验证策略
        self.add_strategy(StrategyConfig(
            name="external_validation",
            description="外部验证：KLOSA数据集验证",
            strategy_type=StrategyType.EXTERNAL_VALIDATION,
            priority=Priority.HIGH,
            estimated_time=10,
            dependencies=["double_optimization"],
            parameters={
                'external_datasets': ['klosa2018 20250722.csv'],
                'generate_shap': True,
                'save_predictions': True
            },
            expected_improvement=0.0,  # 验证不改善性能，但验证泛化能力
            resource_requirement="低"
        ))
        
    def add_strategy(self, strategy_config: StrategyConfig):
        """添加新策略"""
        self.strategies[strategy_config.name] = strategy_config
        print(f"✅ 策略已添加: {strategy_config.name}")
        
    def get_strategy_dependency_graph(self) -> Dict:
        """获取策略依赖图"""
        graph = {}
        for name, strategy in self.strategies.items():
            graph[name] = {
                'dependencies': strategy.dependencies,
                'priority': strategy.priority.value,
                'estimated_time': strategy.estimated_time,
                'expected_improvement': strategy.expected_improvement
            }
        return graph
        
    def recommend_strategy_sequence(self, 
                                   available_time: int = 60,  # 可用时间（分钟）
                                   performance_target: float = 5.0,  # 性能提升目标（%）
                                   resource_constraint: str = "中"  # 资源约束
                                   ) -> List[str]:
        """智能推荐策略执行序列"""
        
        print(f"\n🎯 策略推荐分析")
        print(f"可用时间: {available_time}分钟")
        print(f"性能目标: +{performance_target}%")
        print(f"资源约束: {resource_constraint}")
        print("-" * 50)
        
        # 资源约束映射
        resource_map = {"低": 1, "中": 2, "高": 3}
        max_resource = resource_map.get(resource_constraint, 2)
        
        # 过滤可执行的策略
        eligible_strategies = []
        for name, strategy in self.strategies.items():
            strategy_resource = resource_map.get(strategy.resource_requirement, 2)
            if strategy_resource <= max_resource:
                eligible_strategies.append((name, strategy))
        
        # 依赖关系排序
        sequence = []
        completed = set()
        total_time = 0
        total_improvement = 0
        
        # 首先添加基线策略
        if "baseline" in self.strategies and "baseline" not in completed:
            sequence.append("baseline")
            completed.add("baseline")
            total_time += self.strategies["baseline"].estimated_time
            print(f"📍 基线策略: {self.strategies['baseline'].estimated_time}分钟")
        
        # 按优先级和效益比排序其他策略
        remaining_strategies = [
            (name, strategy) for name, strategy in eligible_strategies 
            if name not in completed
        ]
        
        # 计算效益时间比
        strategy_scores = []
        for name, strategy in remaining_strategies:
            if self._dependencies_satisfied(strategy.dependencies, completed):
                efficiency = strategy.expected_improvement / max(strategy.estimated_time, 1)
                priority_weight = strategy.priority.value
                score = efficiency * priority_weight
                strategy_scores.append((score, name, strategy))
        
        # 按分数排序
        strategy_scores.sort(reverse=True)
        
        # 贪心选择策略
        for score, name, strategy in strategy_scores:
            if total_time + strategy.estimated_time <= available_time:
                if self._dependencies_satisfied(strategy.dependencies, completed):
                    sequence.append(name)
                    completed.add(name)
                    total_time += strategy.estimated_time
                    total_improvement += strategy.expected_improvement
                    
                    print(f"✅ {name}: +{strategy.expected_improvement}%, {strategy.estimated_time}分钟")
                    
                    if total_improvement >= performance_target:
                        print(f"🎯 已达到性能目标!")
                        break
        
        print(f"\n📊 推荐结果:")
        print(f"策略序列: {' → '.join(sequence)}")
        print(f"预计总时间: {total_time}分钟")
        print(f"预期性能提升: +{total_improvement:.1f}%")
        
        return sequence
    
    def _dependencies_satisfied(self, dependencies: List[str], completed: set) -> bool:
        """检查依赖是否满足"""
        return all(dep in completed for dep in dependencies)
    
    def execute_strategy_sequence(self, 
                                 strategy_names: List[str],
                                 pipeline,
                                 train_path: str,
                                 test_path: Optional[str] = None) -> Dict[str, StrategyResult]:
        """执行策略序列"""
        
        print(f"\n{'='*80}")
        print("🚀 开始执行策略序列")
        print(f"策略数量: {len(strategy_names)}")
        print(f"{'='*80}")
        
        results = {}
        baseline_performance = None
        
        for i, strategy_name in enumerate(strategy_names, 1):
            print(f"\n{'='*60}")
            print(f"📋 执行策略 {i}/{len(strategy_names)}: {strategy_name}")
            print(f"{'='*60}")
            
            if strategy_name not in self.strategies:
                print(f"❌ 策略不存在: {strategy_name}")
                continue
                
            strategy_config = self.strategies[strategy_name]
            result = self._execute_single_strategy(strategy_config, pipeline, train_path, test_path)
            results[strategy_name] = result
            
            # 更新基线性能
            if strategy_name == "baseline":
                baseline_performance = result.metrics.get('best_auroc', 0)
                self.current_baseline = result
            
            # 计算性能提升
            if baseline_performance and result.success:
                current_performance = result.metrics.get('best_auroc', 0)
                result.performance_gain = ((current_performance - baseline_performance) / baseline_performance) * 100
                
                # 更新最佳策略
                if not self.best_strategy or current_performance > self.best_strategy.metrics.get('best_auroc', 0):
                    self.best_strategy = result
        
        # 保存执行历史
        self.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategies': strategy_names,
            'results': {name: result.to_dict() for name, result in results.items()}
        })
        
        # 生成执行报告
        self._generate_execution_report(results)
        
        return results
    
    def _execute_single_strategy(self, 
                                strategy_config: StrategyConfig, 
                                pipeline,
                                train_path: str,
                                test_path: Optional[str] = None) -> StrategyResult:
        """执行单个策略"""
        
        result = StrategyResult(strategy_config.name)
        result.start_time = datetime.now()
        
        try:
            print(f"📝 策略描述: {strategy_config.description}")
            print(f"⏱️ 预计时间: {strategy_config.estimated_time}分钟")
            print(f"🎯 预期提升: +{strategy_config.expected_improvement}%")
            
            if strategy_config.strategy_type == StrategyType.BASELINE:
                result = self._execute_baseline_strategy(pipeline, train_path, test_path, result)
                
            elif strategy_config.strategy_type == StrategyType.FEATURE_SELECTION:
                result = self._execute_feature_selection_strategy(pipeline, strategy_config.parameters, result)
                
            elif strategy_config.strategy_type == StrategyType.HYPERPARAMETER_TUNING:
                result = self._execute_hyperparameter_tuning_strategy(pipeline, strategy_config.parameters, result)
                
            elif strategy_config.strategy_type == StrategyType.ENSEMBLE_OPTIMIZATION:
                result = self._execute_ensemble_optimization_strategy(pipeline, strategy_config.parameters, result)
                
            elif strategy_config.strategy_type == StrategyType.EXTERNAL_VALIDATION:
                result = self._execute_external_validation_strategy(pipeline, strategy_config.parameters, test_path, result)
            
            result.success = True
            print(f"✅ 策略执行成功")
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
            print(f"❌ 策略执行失败: {e}")
            
        result.end_time = datetime.now()
        result.duration = (result.end_time - result.start_time).total_seconds() / 60  # 转换为分钟
        
        return result
    
    def _execute_baseline_strategy(self, pipeline, train_path, test_path, result):
        """执行基线策略"""
        
        # 加载和预处理数据
        pipeline.load_and_preprocess_data(train_path, test_path, use_smote=False)
        
        # 训练模型
        models = pipeline.train_models(use_feature_selection=False)
        
        # 评估模型
        evaluation_results = pipeline.evaluate_models()
        
        # 记录结果
        if evaluation_results:
            aurocs = [res['auroc']['value'] for res in evaluation_results.values()]
            result.metrics = {
                'best_auroc': max(aurocs),
                'avg_auroc': np.mean(aurocs),
                'model_count': len(models),
                'feature_count': pipeline.X_train.shape[1],
                'sample_count': pipeline.X_train.shape[0]
            }
            
        result.artifacts = ['model_comparison_results.csv']
        return result
    
    def _execute_feature_selection_strategy(self, pipeline, parameters, result):
        """执行特征选择策略"""
        
        # 应用特征选择
        summary = pipeline.apply_feature_selection(
            methods=parameters.get('methods', ['variance', 'univariate', 'rfe']),
            k_best=parameters.get('k_best', 20),
            variance_threshold=parameters.get('variance_threshold', 0.01)
        )
        
        # 训练模型
        models = pipeline.train_models(use_feature_selection=True)
        
        # 评估模型
        evaluation_results = pipeline.evaluate_models(use_feature_selection=True)
        
        # 记录结果
        if evaluation_results:
            aurocs = [res['auroc']['value'] for res in evaluation_results.values()]
            result.metrics = {
                'best_auroc': max(aurocs),
                'avg_auroc': np.mean(aurocs),
                'selected_features': pipeline.X_train_selected.shape[1],
                'original_features': pipeline.X_train.shape[1],
                'feature_reduction': pipeline.X_train.shape[1] - pipeline.X_train_selected.shape[1]
            }
            
        result.artifacts = ['feature_selection_results.csv', 'model_comparison_results.csv']
        return result
    
    def _execute_hyperparameter_tuning_strategy(self, pipeline, parameters, result):
        """执行超参数调优策略"""
        
        tuned_models, benchmark_df = pipeline.run_hyperparameter_tuning(
            search_method=parameters.get('search_method', 'random'),
            n_iter=parameters.get('n_iter', 20)
        )
        
        # 记录结果
        if not benchmark_df.empty:
            result.metrics = {
                'best_auroc': benchmark_df.iloc[0]['Best_CV_AUC'],
                'best_model': benchmark_df.iloc[0]['Model'],
                'tuned_models': len(tuned_models),
                'search_iterations': parameters.get('n_iter', 20)
            }
            
        result.artifacts = ['hyperparameter_tuning_results.csv']
        return result
    
    def _execute_ensemble_optimization_strategy(self, pipeline, parameters, result):
        """执行集成优化策略"""
        
        # 先特征选择
        if 'feature_selection' in parameters:
            fs_params = parameters['feature_selection']
            pipeline.apply_feature_selection(
                methods=fs_params.get('methods', ['variance', 'univariate', 'rfe']),
                k_best=fs_params.get('k_best', 20)
            )
            use_fs = True
        else:
            use_fs = False
        
        # 训练模型
        models = pipeline.train_models(use_feature_selection=use_fs)
        
        # 参数调优
        if 'hyperparameter_tuning' in parameters:
            hp_params = parameters['hyperparameter_tuning']
            tuned_models, benchmark_df = pipeline.run_hyperparameter_tuning(
                search_method=hp_params.get('search_method', 'random'),
                n_iter=hp_params.get('n_iter', 15)
            )
            
            if not benchmark_df.empty:
                result.metrics = {
                    'best_auroc': benchmark_df.iloc[0]['Best_CV_AUC'],
                    'best_model': benchmark_df.iloc[0]['Model'],
                    'features_used': pipeline.X_train_selected.shape[1] if use_fs else pipeline.X_train.shape[1],
                    'optimization_type': 'double' if use_fs else 'hyperparameter_only'
                }
        
        result.artifacts = ['hyperparameter_tuning_results.csv']
        if use_fs:
            result.artifacts.append('feature_selection_results.csv')
            
        return result
    
    def _execute_external_validation_strategy(self, pipeline, parameters, test_path, result):
        """执行外部验证策略"""
        
        if not test_path:
            raise ValueError("外部验证需要提供测试数据路径")
            
        # 执行外部验证
        external_results = pipeline.run_external_validation(test_path)
        
        # SHAP分析
        if parameters.get('generate_shap', False):
            pipeline.generate_shap_analysis(
                datasets=['train', 'test'], 
                external_data_paths=[test_path]
            )
            
        result.metrics = {
            'external_auroc': external_results.get('auroc', 0) if external_results else 0,
            'external_dataset': test_path,
            'shap_generated': parameters.get('generate_shap', False)
        }
        
        result.artifacts = ['klosa_external_validation_results.csv']
        if parameters.get('generate_shap', False):
            result.artifacts.extend(['shap_plots_klosa_external/'])
            
        return result
    
    def _generate_execution_report(self, results: Dict[str, StrategyResult]):
        """生成执行报告"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"strategy_execution_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 策略执行报告\n\n")
            f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"执行策略数: {len(results)}\n\n")
            
            # 执行摘要
            f.write("## 📊 执行摘要\n\n")
            f.write("| 策略名称 | 状态 | 执行时间(分) | 最佳AUROC | 性能提升 |\n")
            f.write("|----------|------|-------------|-----------|----------|\n")
            
            for name, result in results.items():
                status = "✅ 成功" if result.success else "❌ 失败"
                auroc = result.metrics.get('best_auroc', 0)
                gain = result.performance_gain
                f.write(f"| {name} | {status} | {result.duration:.1f} | {auroc:.3f} | {gain:+.1f}% |\n")
            
            # 详细结果
            f.write("\n## 📋 详细结果\n\n")
            for name, result in results.items():
                f.write(f"### {name}\n")
                f.write(f"- 描述: {self.strategies[name].description}\n")
                f.write(f"- 状态: {'成功' if result.success else '失败'}\n")
                f.write(f"- 执行时间: {result.duration:.1f}分钟\n")
                
                if result.success:
                    f.write(f"- 主要指标:\n")
                    for metric, value in result.metrics.items():
                        f.write(f"  - {metric}: {value}\n")
                    f.write(f"- 生成文件: {', '.join(result.artifacts)}\n")
                else:
                    f.write(f"- 错误信息: {result.error_message}\n")
                f.write("\n")
            
            # 最佳策略推荐
            if self.best_strategy:
                f.write("## 🏆 最佳策略\n\n")
                f.write(f"最佳策略: **{self.best_strategy.strategy_name}**\n")
                f.write(f"最佳性能: {self.best_strategy.metrics.get('best_auroc', 0):.3f}\n")
                f.write(f"性能提升: {self.best_strategy.performance_gain:+.1f}%\n\n")
        
        print(f"\n📄 执行报告已保存: {report_path}")
        
    def get_strategy_recommendations(self, context: Dict) -> Dict:
        """基于上下文获取策略推荐"""
        
        recommendations = {
            'quick_wins': [],      # 快速见效的策略
            'high_impact': [],     # 高影响力策略  
            'resource_efficient': [], # 资源高效策略
            'comprehensive': []    # 综合策略
        }
        
        for name, strategy in self.strategies.items():
            efficiency = strategy.expected_improvement / max(strategy.estimated_time, 1)
            
            # 快速见效（时间短、有一定效果）
            if strategy.estimated_time <= 10 and strategy.expected_improvement >= 2:
                recommendations['quick_wins'].append((name, strategy))
            
            # 高影响力（期望提升大）
            if strategy.expected_improvement >= 4:
                recommendations['high_impact'].append((name, strategy))
            
            # 资源高效（效率高）
            if efficiency >= 0.3:
                recommendations['resource_efficient'].append((name, strategy))
            
            # 综合策略（集成优化）
            if strategy.strategy_type == StrategyType.ENSEMBLE_OPTIMIZATION:
                recommendations['comprehensive'].append((name, strategy))
        
        return recommendations

# 预定义的策略组合方案
STRATEGY_PRESETS = {
    'quick_start': {
        'name': '快速启动方案',
        'description': '适合初次尝试，快速获得基本结果',
        'strategies': ['baseline', 'feature_selection_conservative'],
        'estimated_time': 13,
        'target_users': '初学者、时间紧张'
    },
    
    'balanced': {
        'name': '平衡优化方案', 
        'description': '在时间和性能间取得平衡',
        'strategies': ['baseline', 'feature_selection_conservative', 'hyperparameter_tuning_light', 'double_optimization'],
        'estimated_time': 48,
        'target_users': '一般研究、论文发表'
    },
    
    'performance_focused': {
        'name': '性能导向方案',
        'description': '追求最佳性能，适合竞赛和关键应用',
        'strategies': ['baseline', 'feature_selection_aggressive', 'hyperparameter_tuning_deep', 'ultimate_optimization', 'external_validation'],
        'estimated_time': 127,
        'target_users': '机器学习竞赛、关键业务'
    },
    
    'research_grade': {
        'name': '科研级方案',
        'description': '全面的实验设计，适合高质量研究',
        'strategies': ['baseline', 'feature_selection_conservative', 'feature_selection_aggressive', 
                      'hyperparameter_tuning_light', 'hyperparameter_tuning_deep', 'double_optimization', 
                      'ultimate_optimization', 'external_validation'],
        'estimated_time': 185,
        'target_users': '学术研究、顶级期刊'
    }
}

def create_strategy_manager_demo():
    """创建策略管理器演示"""
    
    print("🎯 策略管理器演示")
    print("="*60)
    
    # 初始化策略管理器
    manager = StrategyManager()
    
    print(f"\n📋 可用策略: {len(manager.strategies)} 个")
    for name, strategy in manager.strategies.items():
        print(f"  {name}: {strategy.description}")
    
    print(f"\n📦 预设方案: {len(STRATEGY_PRESETS)} 个")
    for preset_name, preset in STRATEGY_PRESETS.items():
        print(f"  {preset_name}: {preset['name']} ({preset['estimated_time']}分钟)")
    
    # 推荐策略序列
    print(f"\n🔍 策略推荐分析")
    sequences = {
        '30分钟快速': manager.recommend_strategy_sequence(30, 3.0, "中"),
        '60分钟平衡': manager.recommend_strategy_sequence(60, 5.0, "中"), 
        '120分钟深度': manager.recommend_strategy_sequence(120, 7.0, "高")
    }
    
    return manager, sequences

if __name__ == "__main__":
    create_strategy_manager_demo() 