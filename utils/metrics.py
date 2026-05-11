import numpy as np
import cv2
from typing import List, Tuple

def get_cm(gt:np.ndarray, pred:np.ndarray) -> Tuple[int, int, int, int]:
        tp = cv2.countNonZero(cv2.bitwise_and(gt, pred))
        fp = cv2.countNonZero(cv2.bitwise_and(cv2.bitwise_not(gt), pred))
        fn = cv2.countNonZero(cv2.bitwise_and(gt, cv2.bitwise_not(pred)))
        tn = gt.size - tp - fp - fn
        return tn, fp, fn, tp

def get_iou(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    y_pred = y_pred.astype(np.int_)
    y_true = y_true.astype(np.int_)
    # cm = confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1])
    # tn, fp, fn, tp = cm.ravel()
    tn, fp, fn, tp = get_cm(y_true, y_pred)
    epsilon = 1e-15  # Small epsilon value to avoid division by zero
    iou_score = tp / (tp + fp + fn + epsilon)
    return iou_score 


def get_iou_multiclass(y_true:np.ndarray, y_pred:np.ndarray, categories:List[int]) -> List[float]:
    """Computes the Intersection over Union (IoU) for multi-class segmentation in a per class basis. 
    It outputs a list of IoU values for each class.

    Args:
        y_true (np.ndarray): true segmentation mask of shape (H, W), where values are integers representing the class
        y_pred (np.ndarray): predicted segmentation mask of shape (H, W), where values are integers representing the class
        categories (List[int]): list of categories to compute the IoU for. Not all need to be present in the mask

    Returns:
        list: the IoU values for each class
    """
    present_classes = np.unique(y_true)
    iou_values = []

    for i in categories:
        if i in present_classes:
            y_true_i = y_true == i
            y_pred_i = y_pred == i
            iou_values.append(get_iou(y_true_i, y_pred_i))
        else:
            iou_values.append(np.nan)
    
    return iou_values 


def get_multi_class_metrics(y_true:np.ndarray, y_pred:np.ndarray, categories:List[int]) -> Tuple[List[float], List[float], List[float]]:
    """Computes the Intersection over Union (IoU), Dice coefficient, and Matthews Correlation Coefficient (MCC) for multi-class segmentation. 
    It outputs a dictionary of IoU, Dice, and MCC values for each class.

    Args:
        y_true (np.ndarray): true segmentation mask of shape (H, W), where values are integers representing the class
        y_pred (np.ndarray): predicted segmentation mask of shape (H, W), where values are integers representing the class
        categories (List[int]): list of categories to compute the metrics for. Not all need to be present in the mask

    Returns:
        dict: the IoU, Dice, and MCC values for each class
    """
    def compute_dice(tp:int, fp:int, fn:int) -> float:
        epsilon = 1e-15  # Small epsilon value to avoid division by zero
        return 2 * tp / (2 * tp + fp + fn + epsilon)
    def compute_iou(tp:int, fp:int, fn:int) -> float:
        epsilon = 1e-15  # Small epsilon value to avoid division by zero
        return tp / (tp + fp + fn + epsilon)
    def compute_mcc(tp:int, tn:int, fp:int, fn:int) -> float:
        epsilon = 1e-15  # Small epsilon value to avoid division by zero
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp + epsilon) * (tp + fn + epsilon) * (tn + fp + epsilon) * (tn + fn + epsilon))
        # Check if denominator is close to zero to avoid division by zero
        if denominator < epsilon:
            return 0.0
        return numerator / denominator

    present_in_gt = np.unique(y_true)
    present_in_preds = np.unique(y_pred)
    iou_values, dice_values, mcc_values = [], [], []
    for i in categories:
        if i in present_in_gt or i in present_in_preds:
            y_true_i = (y_true == i).astype(np.uint8)
            y_pred_i = (y_pred == i).astype(np.uint8)
            
            # Compute true positives, false positives, false negatives, and true negatives using OpenCV
            tn, fp, fn, tp = get_cm(y_true_i, y_pred_i)
            # tn_, fp_, fn_, tp_ = confusion_matrix(y_true_i.ravel(), y_pred_i.ravel()).ravel()
            dice_values.append(compute_dice(tp, fp, fn) if i in present_in_gt else 0.)
            iou_values.append(compute_iou(tp, fp, fn) if i in present_in_gt else 0.)
            mcc_values.append(compute_mcc(tp, tn, fp, fn))
        else:
            iou_values.append(np.nan)
            dice_values.append(np.nan)
            mcc_values.append(np.nan)
    return iou_values, dice_values, mcc_values