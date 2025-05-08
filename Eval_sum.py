import pandas as pd

CSV_PATH = "eval_report.csv"

def summarize_eval(csv_path):
    df = pd.read_csv(csv_path, index_col=0)  # <- Đặt cột tên class làm index
    
    # Giữ lại macro avg, weighted avg, accuracy
    df_summary = df[df.index.str.contains('macro avg|weighted avg|accuracy', na=False)]

    # Nhóm theo model
    summary = df_summary.groupby("model").agg({
        "precision": "mean",
        "recall": "mean",
        "f1-score": "mean",
        "accuracy": "mean"
    }).reset_index()

    summary = summary.rename(columns={
        "precision": "avg_precision",
        "recall": "avg_recall",
        "f1-score": "avg_f1",
        "accuracy": "accuracy"
    })

    print("\nTổng hợp kết quả theo mô hình:")
    print(summary.to_string(index=False))

    summary.to_csv("summary_eval.csv", index=False)
    print("\nFile summary_eval.csv đã được lưu!")

if __name__ == "__main__":
    summarize_eval(CSV_PATH)
