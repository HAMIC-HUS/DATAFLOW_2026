import numpy as np
import pandas as pd

# 6 thuộc tính cần dự đoán
TARGET_COLS = [f"attr_{i}" for i in range(1, 7)]

# Hằng số chuẩn hóa và trọng số cho metric vòng Chung kết
M_CONSTANTS = np.array([12, 31, 99, 12, 31, 99], dtype=float)
W_CONSTANTS = np.array([1, 1, 100, 1, 1, 100], dtype=float)

def exact_match_accuracy(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> float:
    """
    Tính Exact Match Accuracy (phải đúng cả 6 thuộc tính cho một bản ghi).
    """
    Y_true = Y_true.astype(np.int64)
    Y_pred = np.round(Y_pred).astype(np.int64)

    # So sánh từng dòng (mỗi bản ghi) có khớp hoàn toàn cả 6 thuộc tính không
    exact_matches = np.all(Y_true == Y_pred, axis=1)
    
    # Tính accuracy (%)
    accuracy = np.mean(exact_matches) * 100.0
    return accuracy

def weighted_l2_mse(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> float:
    """
    Tính Weighted L2 (MSE) cho vòng Chung kết.
    """
    Y_true = Y_true.astype(float)
    Y_pred = Y_pred.astype(float)
    
    diff = (Y_pred / M_CONSTANTS) - (Y_true / M_CONSTANTS)
    squared_diff = diff ** 2
    weighted_squared_diff = squared_diff * W_CONSTANTS
    
    mean_score = np.mean(np.sum(weighted_squared_diff, axis=1) / 6.0)
    return mean_score

def score(gold_path: str, pred_path: str, metric: str = 'weighted_l2_mse') -> float:
    """
    Hàm tính điểm chính cho hệ thống.
    Trả về điểm theo metric được chỉ định ('exact_match' hoặc 'weighted_l2_mse').
    """
    df_gold = pd.read_csv(gold_path)
    df_pred = pd.read_csv(pred_path)
    
    id_col = None
    if "id" in df_pred.columns:
        id_col = "id"
    
    if id_col is None:
        raise ValueError(f"Không tìm thấy cột ID trong file dự đoán. Các cột hiện có: {list(df_pred.columns)}")
        
    if id_col in df_gold.columns and id_col in df_pred.columns:
        df_gold = df_gold.sort_values(by=id_col).reset_index(drop=True)
        df_pred = df_pred.sort_values(by=id_col).reset_index(drop=True)
    
    if len(df_gold) != len(df_pred):
        raise ValueError(f"Số lượng dự đoán ({len(df_pred)}) KHÔNG khớp nhãn ({len(df_gold)})")
        
    Y_true = df_gold[TARGET_COLS].values
    Y_pred = df_pred[TARGET_COLS].values
        
    if metric == 'exact_match':
        return exact_match_accuracy(Y_true, Y_pred)
    elif metric == 'weighted_l2_mse':
        return weighted_l2_mse(Y_true, Y_pred)
    else:
        raise ValueError("Metric không hợp lệ. Vui lòng chọn 'exact_match' hoặc 'weighted_l2_mse'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script đánh giá dự án User Behavior Prediction")
    parser.add_argument("gold_path", type=str, help="Đường dẫn file nhãn thực tế (gold)")
    parser.add_argument("pred_path", type=str, help="Đường dẫn file dự đoán")
    parser.add_argument("--metric", type=str, default="weighted_l2_mse", choices=["exact_match", "weighted_l2_mse"], help="Metric sử dụng để đánh giá (mặc định: weighted_l2_mse)")
    
    args = parser.parse_args()
    score_val = score(args.gold_path, args.pred_path, metric=args.metric)
    print(f"Final Score ({args.metric}): {score_val:.4f}")

