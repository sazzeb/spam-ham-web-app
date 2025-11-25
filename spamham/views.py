from django.shortcuts import render, redirect
import numpy
import pickle
import string
import math
import numpy as np
import pandas as pd
from pandas.errors import ParserError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base64
from io import BytesIO
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from utils import text_process
from manage import importPipelines



def make_summary_chart(label_counts,
                       acc_per_label,
                       total_emails,
                       avg_accuracy_all,
                       avg_raw_length,
                       avg_clean_length,
                       file_size_kb):
    """
    Build one summary figure:
      - top: class distribution (count per class)
      - bottom: average accuracy per class
      - left text: totals and averages (no model date)
    """
    labels = sorted(label_counts.keys())
    counts = [label_counts[l] for l in labels]
    mean_acc = [
        sum(acc_per_label[l]) / len(acc_per_label[l]) if acc_per_label[l] else 0.0
        for l in labels
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].bar(labels, counts)
    axes[0].set_ylabel("Number of emails")
    axes[0].set_title("Class distribution")

    axes[1].bar(labels, mean_acc)
    axes[1].set_ylabel("Average accuracy (%)")
    axes[1].set_xlabel("Class")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.xticks(rotation=45, ha="right")

    text_lines = [
        f"Total emails: {total_emails}",
        f"File size: {file_size_kb:.1f} KB" if file_size_kb is not None else "File size: N/A",
        f"Overall avg accuracy: {avg_accuracy_all:.1f}%",
        f"Avg raw length: {avg_raw_length:.1f} chars",
        f"Avg clean length: {avg_clean_length:.1f} chars",
    ]
    stats_text = "\n".join(text_lines)

    fig.text(
        0.02,
        0.98,
        stats_text,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey")
    )

    plt.tight_layout(rect=[0.18, 0.0, 1.0, 0.95])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64




def make_bar_chart(labels, values, title, xlabel, ylabel):
    """
    Create a simple bar chart and return it as a base64 string.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def load_uploaded_dataframe(uploaded_file):
    """
    Accepts an uploaded CSV or Excel file and returns a pandas DataFrame.
    Handles slightly broken CSVs by falling back to a tolerant parser.
    """
    filename = uploaded_file.name.lower()

    # CSV files
    if filename.endswith(".csv"):
        try:
            # First try the fast default engine
            df = pd.read_csv(uploaded_file)
        except ParserError as e:
            # Reset file pointer before trying again
            uploaded_file.seek(0)

            # Fallback: more tolerant parser, skip bad lines
            df = pd.read_csv(
                uploaded_file,
                engine="python",
                on_bad_lines="skip"  # skip rows that break parsing
            )
        return df

    # Excel files
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        return df

    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def build_text_column(df):
    possible_subject_cols = ["subject", "Subject", "SUBJECT"]
    possible_message_cols = ["message", "Message", "MESSAGE", "body", "Body"]

    subject_col = None
    message_col = None

    for col in df.columns:
        if col in possible_subject_cols:
            subject_col = col
        if col in possible_message_cols:
            message_col = col

    texts = []

    for _, row in df.iterrows():
        subject = str(row[subject_col]) if subject_col and pd.notna(row[subject_col]) else ""
        message = str(row[message_col]) if message_col and pd.notna(row[message_col]) else ""

        combined = (subject + " " + message).strip()
        if not combined:
            combined = " ".join(str(v) for v in row.values if pd.notna(v))

        texts.append(combined)

    return texts

def home(request):
    if request.method == 'POST':
        # 1) If a message is submitted via textarea (single prediction)
        message_raw = request.POST.get('message', '').strip()
        uploaded_file = request.FILES.get('file')

        # --- Case A: single text prediction ---
        if message_raw:
            message_cp = message_raw  # copy for display
            raw_length = len(message_raw)

            tokens = text_process(message_raw)
            cleaned = ' '.join(tokens)
            clean_length = len(cleaned)

            message_for_model = [cleaned]
            result, accuracy = predict(message_for_model)

            context = {
                'result': result,
                'message': message_cp,
                'accuracy': accuracy,
                'raw_length': raw_length,
                'clean_length': clean_length,
            }
            return render(request, 'home.html', context)

        # --- Case B: file upload for multiple predictions ---
        if uploaded_file:
            try:
                df = load_uploaded_dataframe(uploaded_file)
                texts = build_text_column(df)
                bulk_results = multiple_predict(texts)

                # You can pass the results list directly to the template
                context = {
                    'bulk_results': bulk_results,
                    'file_name': uploaded_file.name,
                }
                return render(request, 'multiple_upload.html', context)
            except Exception as e:
                # basic error reporting back to template
                context = {
                    'error': str(e),
                }
                return render(request, 'multiple_upload.html', context)

    # GET (or no valid POST) – just render empty form
    return render(request, 'home.html')

def multiple_upload(request):
    
    if request.method != "POST":
        return redirect("home")

    upload_list = request.FILES.getlist("files")

    if not upload_list:
        return render(request, "multiple_upload.html", {"error": "No files selected for upload."})

    all_file_results = []

    for uploaded_file in upload_list:
        file_name = uploaded_file.name

        try:
            df = load_uploaded_dataframe(uploaded_file)
            texts = build_text_column(df)
            bulk_results = multiple_predict(texts)

            # If no rows, just record that and skip chart
            if not bulk_results:
                all_file_results.append({
                    "file_name": file_name,
                    "results": [],
                    "error": "No valid rows found in file.",
                    "summary_chart": None,
                    "summary_stats": None,
                })
                continue

            # ---------- aggregate for summary ----------
            label_counts = Counter()
            acc_per_label = defaultdict(list)
            total_accuracy = 0.0
            total_raw_len = 0
            total_clean_len = 0

            for row in bulk_results:
                result_str = row["result"]  # "very likely phishing"
                raw_label = result_str.split()[-1].lower()

                label_counts[raw_label] += 1
                acc_per_label[raw_label].append(row["accuracy"])

                total_accuracy += row["accuracy"]
                total_raw_len += row["raw_length"]
                total_clean_len += row["clean_length"]

            total_emails = len(bulk_results)
            avg_accuracy_all = total_accuracy / total_emails
            avg_raw_length = total_raw_len / total_emails
            avg_clean_length = total_clean_len / total_emails

            file_size_kb = getattr(uploaded_file, "size", None)
            if file_size_kb is not None:
                file_size_kb = file_size_kb / 1024.0

            # Compose one summary chart
            summary_chart = make_summary_chart(
                label_counts=label_counts,
                acc_per_label=acc_per_label,
                total_emails=total_emails,
                avg_accuracy_all=avg_accuracy_all,
                avg_raw_length=avg_raw_length,
                avg_clean_length=avg_clean_length,
                file_size_kb=file_size_kb,
            )

            # Also create a short text summary for display
            if label_counts:
                top_label, top_count = max(label_counts.items(), key=lambda kv: kv[1])
            else:
                top_label, top_count = "N/A", 0

            summary_stats = {
                "total_emails": total_emails,
                "file_size_kb": file_size_kb,
                "avg_accuracy_all": avg_accuracy_all,
                "avg_raw_length": avg_raw_length,
                "avg_clean_length": avg_clean_length,
                "top_label": top_label,
                "top_count": top_count,
            }

            all_file_results.append({
                "file_name": file_name,
                "results": bulk_results,
                "error": None,
                "summary_chart": summary_chart,
                "summary_stats": summary_stats,
            })

        except Exception as e:
            all_file_results.append({
                "file_name": file_name,
                "results": [],
                "error": str(e),
                "summary_chart": None,
                "summary_stats": None,
            })

    return render(request, "multiple_upload.html", {
        "multi_files_results": all_file_results
    })



def predict(message):

    # load both pipelines (they should be trained on the same label set)
    pipeline, pipeline_second = importPipelines()

    # probabilities from first model
    prob1 = pipeline.predict_proba(message)[0]       # shape: [n_classes]
    classes = pipeline.classes_                      # labels, e.g. ['ham','spam','phishing',...]
    # probabilities from second model
    prob2 = pipeline_second.predict_proba(message)[0]

    # average ensemble of both models
    combined = (prob1 + prob2) / 2.0                 # still [n_classes]

    # best label + confidence
    best_idx = int(np.argmax(combined))
    raw_label = classes[best_idx]                    # e.g. 'phishing', 'spam', 'harmful', ...
    confidence = float(combined[best_idx]) * 100.0   # 0–100 %
    confidence = round(confidence, 3)

    # human-friendly text
    if confidence > 80:
        result = f"predicting {raw_label}"
    else:
        result = f"predicting {raw_label}"

    return result, confidence

def multiple(request):
    return render(request, 'multiple_upload.html')    

def multiple_predict(texts):
    results = []

    for raw_text in texts:
        raw_text = raw_text or ""
        raw_length = len(raw_text)

        tokens = text_process(raw_text)
        cleaned = " ".join(tokens)
        clean_length = len(cleaned)

        result, accuracy = predict([cleaned])

        results.append({
            "raw_text": raw_text,
            "result": result,          # e.g. "very likely phishing"
            "accuracy": accuracy,      # numeric %
            "raw_length": raw_length,
            "clean_length": clean_length,
        })

    return results
