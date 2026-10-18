"""NV2: prepare -> train (GridSearchCV) -> evaluate held-out test once.

Run: python -m src.nv2 run
The CNN of NV3 can later be scored using its frozen test predictions.
"""
import argparse
import hashlib
import json
import os
import platform
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

if __package__:
    from .nv2_models import model_specs
else:
    from nv2_models import model_specs

CLASSES = np.arange(10)


def log(text):
    print(text, flush=True)


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_csv(path):
    frame = pd.read_csv(path)
    expected = ["label"] + [f"pixel{i}" for i in range(784)]
    if list(frame.columns) != expected or frame.isna().any().any():
        raise ValueError("Can CSV co label, pixel0..pixel783 dung thu tu va khong thieu du lieu.")
    y = frame.label.to_numpy()
    X = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not np.array_equal(np.unique(y), CLASSES):
        raise ValueError("Nhan phai la cac so nguyen 0..9, du 10 lop.")
    if not np.isfinite(X).all() or X.min() < 0 or X.max() > 255:
        raise ValueError("Pixel phai huu han, trong [0, 255].")
    return X / 255.0, y.astype(np.int64)


def make_split(y, seed):
    train, held = train_test_split(np.arange(len(y)), test_size=0.30, stratify=y, random_state=seed)
    val, test = train_test_split(held, test_size=0.5, stratify=y[held], random_state=seed)
    return train, val, test


def prepare(args):
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "config.json").exists():
        raise ValueError("Run da ton tai. Dung train/evaluate de tiep tuc, hoac chon --output moi.")
    X, y = load_csv(args.data)
    source_ids = np.arange(len(y))
    if args.max_samples and args.max_samples < len(y):
        source_ids, _ = train_test_split(source_ids, train_size=args.max_samples,
                                        stratify=y, random_state=args.seed)
        X, y = X[source_ids], y[source_ids]
    train, val, test = make_split(y, args.seed)
    tuning = train
    if 0 < args.tune_samples < len(train):
        tuning, _ = train_test_split(train, train_size=args.tune_samples,
                                     stratify=y[train], random_state=args.seed)
    if min(np.bincount(y[tuning], minlength=10)) < args.folds:
        raise ValueError("Khong du mau moi lop cho CV.")
    config = {"source_csv": str(args.data.resolve()), "source_sha256": sha256(args.data),
              "seed": args.seed, "folds": args.folds, "threads": args.threads,
              "tuning_samples": len(tuning), "train_samples": len(train),
              "validation_samples": len(val), "test_samples": len(test),
              "smoke_test": bool(args.max_samples), "preprocessing": "float32 pixel / 255; no augmentation",
              "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}
    np.savez_compressed(folder / "split_indices.npz", train=source_ids[train], validation=source_ids[val],
                        test=source_ids[test], tuning=source_ids[tuning])
    # Shared flattened arrays; CNN can reshape X to (-1, 28, 28, 1).
    np.savez_compressed(folder / "shared_data.npz", X_train=X[train], y_train=y[train],
                        X_validation=X[val], y_validation=y[val], X_test=X[test], y_test=y[test],
                        train_row_ids=source_ids[train], validation_row_ids=source_ids[val], test_row_ids=source_ids[test])
    save_json(folder / "config.json", config)
    log(f"Prepared: train={len(train)}, validation={len(val)}, test={len(test)}, tuning={len(tuning)}")


def score(y, pred):
    report = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"]}


def timed_fit(model, X, y):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    return time.perf_counter() - start, sorted({str(w.message) for w in caught})


def train(args):
    folder = args.output
    config = read_json(folder / "config.json")
    if (folder / "test_frozen.json").exists():
        raise ValueError("Run da mo tap test. Khong duoc train/tune lai trong run nay.")
    if sha256(Path(config["source_csv"])) != config["source_sha256"]:
        raise ValueError("CSV da thay doi so voi cach chia da luu.")
    with np.load(folder / "shared_data.npz") as data:
        X, y = data["X_train"], data["y_train"]
        Xv, yv = data["X_validation"], data["y_validation"]
    split = np.load(folder / "split_indices.npz")
    positions = {int(row_id): i for i, row_id in enumerate(split["train"])}
    tune_ids = np.array([positions[int(i)] for i in split["tuning"]])
    cv = StratifiedKFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    records = []
    with threadpool_limits(limits=config["threads"]):
        for name, (base, grid) in model_specs(config["seed"]).items():
            target = folder / name
            target.mkdir(exist_ok=True)
            if (target / "training.json").exists():
                records.append(read_json(target / "training.json"))
                log(f"{name}: da hoan tat, giu nguyen.")
                continue
            log(f"\n{name}: GridSearchCV {config['folds']} folds, {len(tune_ids)} anh goc")
            search = GridSearchCV(base, grid, cv=cv, scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                                  refit=False, n_jobs=1, error_score="raise", verbose=1)
            search_seconds, search_warnings = timed_fit(search, X[tune_ids], y[tune_ids])
            cv_table = pd.DataFrame(search.cv_results_)
            cv_table.to_csv(target / "cv_results.csv", index=False)
            best_index = int(np.argmax(search.cv_results_["mean_test_accuracy"]))
            params = search.cv_results_["params"][best_index]
            log(f"{name}: fit toan bo train voi {params}")
            model = clone(base).set_params(**params)
            fit_seconds, fit_warnings = timed_fit(model, X, y)
            start = time.perf_counter()
            pred = model.predict(Xv)
            predict_seconds = time.perf_counter() - start
            joblib.dump(model, target / "model.joblib", compress=3)
            np.testing.assert_array_equal(joblib.load(target / "model.joblib").predict(Xv[:20]), pred[:20])
            record = {"model": name, **score(yv, pred), "best_params": params,
                      "cv_accuracy_mean": float(search.cv_results_["mean_test_accuracy"][best_index]),
                      "cv_accuracy_std": float(search.cv_results_["std_test_accuracy"][best_index]),
                      "search_seconds": search_seconds, "fit_seconds": fit_seconds,
                      "validation_predict_seconds": predict_seconds,
                      "model_bytes": (target / "model.joblib").stat().st_size,
                      "model_sha256": sha256(target / "model.joblib"),
                      "warnings": search_warnings + fit_warnings}
            save_json(target / "training.json", record)
            pd.DataFrame({"row_id": split["validation"], "y_true": yv, "y_pred": pred}).to_csv(
                target / "validation_predictions.csv", index=False)
            records.append(record)
            pd.DataFrame(records).to_csv(folder / "validation_comparison.csv", index=False)
            log(f"{name}: validation accuracy={record['accuracy']:.4%}, fit={fit_seconds:.1f}s")
    winner = max(records, key=lambda r: r["accuracy"])["model"]
    save_json(folder / "selection.json", {"selected_on_validation": winner,
              "rule": "Maximum validation accuracy; order in nv2_models breaks ties", "model_count": len(records)})
    log(f"Selected before opening test: {winner}")


def validate_predictions(frame, ids):
    if not {"row_id", "y_pred"}.issubset(frame.columns) or frame.row_id.duplicated().any():
        raise ValueError("Du doan can row_id duy nhat va y_pred.")
    if len(frame) != len(ids) or set(frame.row_id) != set(ids):
        raise ValueError("row_id khong khop tap test dung chung.")
    if not frame.y_pred.isin(CLASSES).all():
        raise ValueError("Du doan phai la so nguyen 0..9.")
    return frame.set_index("row_id").loc[ids, "y_pred"].to_numpy(dtype=np.int64)


def evaluation_artifacts(target, X, y, pred, ids):
    report = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T
    # One-vs-rest accuracy per class complements overall multiclass accuracy.
    for digit in CLASSES:
        table.loc[str(digit), "ovr_accuracy"] = np.mean((y == digit) == (pred == digit))
    table.to_csv(target / "test_classification_report.csv")
    matrix = confusion_matrix(y, pred, labels=CLASSES)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(target / "test_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{target.name} | held-out test (n={len(y)})")
    fig.tight_layout(); fig.savefig(target / "test_confusion_matrix.png", dpi=140); plt.close(fig)
    wrong = np.flatnonzero(y != pred)
    pd.DataFrame({"row_id": ids[wrong], "true": y[wrong], "predicted": pred[wrong]}).to_csv(
        target / "misclassified.csv", index=False)
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for ax, i in zip(axes.flat, list(wrong[:15]) + [None] * 15):
        ax.axis("off")
        if i is not None:
            ax.imshow(X[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"row {ids[i]}: {y[i]} -> {pred[i]}", fontsize=10)
    fig.suptitle(f"{target.name}: first 15 errors (true -> predicted)")
    fig.tight_layout(); fig.savefig(target / "misclassified.png", dpi=140); plt.close(fig)
    np.fill_diagonal(matrix, 0)
    pairs = sorted(((int(matrix[a, b]), int(a), int(b)) for a in CLASSES for b in CLASSES if matrix[a, b]), reverse=True)[:5]
    save_json(target / "error_summary.json", {"errors": len(wrong), "top_confusions": [
        {"true": a, "predicted": b, "count": count} for count, a, b in pairs]})
    return score(y, pred)


def freeze(folder):
    selection = read_json(folder / "selection.json")
    names = list(model_specs())
    if selection["model_count"] != len(names):
        raise ValueError("Can train du tat ca mo hinh truoc khi mo test.")
    current = {name: sha256(folder / name / "model.joblib") for name in names}
    for name, digest in current.items():
        if digest != read_json(folder / name / "training.json")["model_sha256"]:
            raise ValueError("Model da thay doi sau training.")
    state = {"models": current, "shared_data_sha256": sha256(folder / "shared_data.npz"),
             "selection": selection}
    marker = folder / "test_frozen.json"
    if marker.exists() and read_json(marker) != state:
        raise ValueError("Run da frozen; model/du lieu/selection khong duoc thay doi.")
    save_json(marker, state)


def evaluate(args):
    folder = args.output
    freeze(folder)
    config = read_json(folder / "config.json")
    with np.load(folder / "shared_data.npz") as data:
        X, y, ids = data["X_test"], data["y_test"], data["test_row_ids"]
    with threadpool_limits(limits=config["threads"]):
        for name in model_specs():
            target = folder / name
            metrics_path = target / "test_metrics.json"
            if metrics_path.exists():
                log(f"{name}: dung ket qua test da luu, khong predict lai.")
                continue
            cache = target / "test_predictions.csv"
            timing = target / "test_timing.json"
            if cache.exists() and timing.exists():
                pred = validate_predictions(pd.read_csv(cache), ids)
                seconds = read_json(timing)["predict_seconds"]
            else:
                model = joblib.load(target / "model.joblib")
                start = time.perf_counter(); pred = model.predict(X); seconds = time.perf_counter() - start
                save_json(timing, {"predict_seconds": seconds})
                pd.DataFrame({"row_id": ids, "y_true": y, "y_pred": pred}).to_csv(cache, index=False)
            metrics = evaluation_artifacts(target, X, y, pred, ids)
            record = {**read_json(target / "training.json"), **metrics, "test_samples": len(y),
                      "test_predict_seconds": seconds, "ms_per_image": seconds * 1000 / len(y)}
            save_json(metrics_path, record)
            log(f"{name}: test accuracy={record['accuracy']:.4%}, F1={record['f1_macro']:.4f}")
    report(folder)


def evaluate_cnn(args):
    folder = args.output
    freeze(folder)
    target = folder / "cnn_nv3"
    if (target / "test_metrics.json").exists():
        raise ValueError("CNN da duoc danh gia. Khong ghi de ket qua test.")
    if args.predictions is None or args.cnn_metadata is None:
        raise ValueError("Can --predictions CSV va --cnn-metadata JSON. Xem README_NV2.md.")
    meta = read_json(args.cnn_metadata)
    for key in ["fit_seconds", "predict_seconds", "model_bytes"]:
        if not isinstance(meta.get(key), (int, float)) or not np.isfinite(meta[key]) or meta[key] < 0:
            raise ValueError(f"CNN metadata can {key} huu han, >= 0.")
    if meta.get("shared_data_sha256") != sha256(folder / "shared_data.npz"):
        raise ValueError("CNN metadata phai xac nhan dung bo chia shared_data.npz nay.")
    with np.load(folder / "shared_data.npz") as data:
        X, y, ids = data["X_test"], data["y_test"], data["test_row_ids"]
    pred = validate_predictions(pd.read_csv(args.predictions), ids)
    target.mkdir(exist_ok=True)
    pd.DataFrame({"row_id": ids, "y_true": y, "y_pred": pred}).to_csv(target / "test_predictions.csv", index=False)
    metrics = evaluation_artifacts(target, X, y, pred, ids)
    save_json(target / "test_metrics.json", {"model": "cnn_nv3", **metrics, "fit_seconds": meta["fit_seconds"],
              "test_predict_seconds": meta["predict_seconds"], "model_bytes": meta["model_bytes"],
              "ms_per_image": meta["predict_seconds"] * 1000 / len(y), "test_samples": len(y),
              "source": "Predictions and timing supplied by NV3", "metadata": meta})
    report(folder)


def report(folder):
    records = [read_json(p) for p in sorted(folder.glob("*/test_metrics.json"))]
    order = list(model_specs()) + ["cnn_nv3"]
    records.sort(key=lambda r: order.index(r["model"]))
    pd.DataFrame(records).to_csv(folder / "test_comparison.csv", index=False)
    config = read_json(folder / "config.json")
    selection = read_json(folder / "selection.json")["selected_on_validation"]
    best = max(records, key=lambda r: r["accuracy"])
    lines = ["# NV2 — Kết quả thực nghiệm", "", "Không bao gồm phần lý thuyết.", "",
             f"Dữ liệu: {config['train_samples']:,} train / {config['validation_samples']:,} validation / {config['test_samples']:,} test; seed {config['seed']}.",
             f"Pixel /255, không augmentation. GridSearchCV {config['folds']} fold trên {config['tuning_samples']:,} ảnh train; fit cuối trên toàn bộ train.",
             "CV phục vụ chọn tham số, không phải ước lượng độc lập. Tập test được giữ lại trong lần thực nghiệm này; dữ liệu nguồn đã từng được dùng ở các thí nghiệm cũ.", "",
             f"**Mô hình chọn theo validation trước khi mở test: {selection}.**", "",
             "| Mô hình | Accuracy | Precision macro | Recall macro | F1 macro | Fit (s) | Predict test (s) | File (MiB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['model']} | {r['accuracy']:.2%} | {r['precision_macro']:.4f} | {r['recall_macro']:.4f} | {r['f1_macro']:.4f} | {r['fit_seconds']:.2f} | {r['test_predict_seconds']:.2f} | {r['model_bytes']/1048576:.2f} |")
    lines += ["", "Thời gian fit cuối không bao gồm tìm tham số. Thời gian predict là một lần dự đoán cả tập test, không gồm tải model; phụ thuộc máy và số luồng. Kích thước là file joblib nén mức 3, không phải RAM lúc chạy.",
              "", f"Accuracy test cao nhất trong các mô hình đã có: **{best['model']} ({best['accuracy']:.2%})**. Không dùng kết quả test để tuning lại.",
              "", "## Tham số và cảnh báo", ""]
    for r in records:
        if "best_params" in r:
            lines.append(f"- {r['model']}: `{r['best_params']}`; CV accuracy {r['cv_accuracy_mean']:.2%} ± {r['cv_accuracy_std']:.2%}; tìm tham số {r['search_seconds']:.1f}s.")
            if r["warnings"]:
                lines.append(f"  Cảnh báo: {'; '.join(r['warnings'])}")
    lines += ["", "## Phân tích lỗi", "", "Ảnh sai và mã dòng gốc được lưu riêng cho từng mô hình. true → predicted trong hình là nhãn thật → dự đoán."]
    for r in records:
        name = r["model"]; error = read_json(folder / name / "error_summary.json")
        pairs = "; ".join(f"{p['true']}→{p['predicted']}: {p['count']} ảnh" for p in error["top_confusions"])
        lines += ["", f"### {name}", f"Sai {error['errors']}/{r['test_samples']} ảnh. Các cặp nhầm nhiều nhất: {pairs}.",
                  f"![Ảnh sai]({name}/misclassified.png)"]
    lines += ["", "Nhận xét nguyên nhân theo ảnh mẫu được ghi riêng trong error_analysis.md sau khi kiểm tra trực quan. Các nguyên nhân là giả thuyết, không phải kết luận nhân quả.",
              "", "## Phối hợp NV1 và NV3", "",
              "Dùng shared_data.npz và split_indices.npz để giữ cùng ảnh/cùng thứ tự. NV3 reshape thành (n,28,28,1); chỉ train/validation được dùng để chọn CNN.",
              "CNN: " + ("đã bổ sung kết quả NV3." if any(r["model"] == "cnn_nv3" for r in records) else "chưa có mô hình/dự đoán từ NV3; chưa thể kết luận mô hình tốt nhất toàn nhóm."),
              "", "Per-class precision/recall/F1 và OvR accuracy nằm trong test_classification_report.csv. Recall của một chữ số chính là tỷ lệ dự đoán đúng trong các ảnh thật của chữ số đó."]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([r["model"] for r in records], [100*r["accuracy"] for r in records], color="#286B8C")
    ax.set(ylabel="Test accuracy (%)", ylim=(0, 105), title="NV2: held-out test accuracy")
    for i, r in enumerate(records): ax.text(i, 100*r["accuracy"]+1, f"{r['accuracy']:.2%}", ha="center")
    fig.tight_layout(); fig.savefig(folder / "accuracy_comparison.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "run", "evaluate-cnn", "report"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/digit-recognizer/train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/nv2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--tune-samples", type=int, default=6000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--cnn-metadata", type=Path)
    args = parser.parse_args()
    if args.folds < 2 or args.threads < 1 or min(args.tune_samples, args.max_samples) < 0:
        parser.error("folds>=2, threads>=1, sample limits>=0")
    args.output = args.output.resolve()
    if args.command == "run":
        prepare(args); train(args); evaluate(args)
    elif args.command == "report": report(args.output)
    else: {"prepare": prepare, "train": train, "evaluate": evaluate, "evaluate-cnn": evaluate_cnn}[args.command](args)


if __name__ == "__main__":
    main()
