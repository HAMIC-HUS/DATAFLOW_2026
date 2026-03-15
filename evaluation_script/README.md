# Đánh giá (Evaluation)

## Hướng dẫn sử dụng (Usage)

Script đánh giá hỗ trợ tính toán hai loại điểm số:
1. **Weighted L2 (MSE)** (mặc định) cho vòng Chung kết.
2. **Exact Match Accuracy** cho vòng Bán kết.

Cài đặt các thư viện cần thiết:
```bash 
pip install -r requirements.txt
```

### Chạy từ Command Line

Sử dụng metric mặc định (weighted_l2_mse):
```bash
python evaluate.py sample_data/Y_gold.csv sample_data/Y_pred.csv
```

Sử dụng metric tuỳ chỉnh bằng tham số `--metric`:
```bash
python evaluate.py sample_data/Y_gold.csv sample_data/Y_pred.csv --metric weighted_l2_mse
python evaluate.py sample_data/Y_gold.csv sample_data/Y_pred.csv --metric exact_match
```

### Chạy trên Python / Jupyter Notebook

```python
from evaluate import score

# Đánh giá file trực tiếp và lấy điểm weighted_l2_mse (mặc định)
score_mse = score('sample_data/Y_gold.csv', 'sample_data/Y_pred.csv')
print("Kết quả Weighted L2 MSE:", score_mse)

# Tính điểm với exact_match
score_em = score('sample_data/Y_gold.csv', 'sample_data/Y_pred.csv', metric="exact_match")
print("Kết quả Exact Match Accuracy:", score_em)
```
