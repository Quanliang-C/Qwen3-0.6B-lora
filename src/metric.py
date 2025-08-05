"""
简单的指标计算函数
"""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_metrics(y_true, y_pred):
    """
    计算分类指标
    
    Args:
        y_true: 真实标签列表 [0, 1, 1, 0, ...]
        y_pred: 预测标签列表 [1, 1, 0, 0, ...]
    
    Returns:
        dict: 包含各项指标的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_metrics(metrics):
    """打印指标结果"""
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
