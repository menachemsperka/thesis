import importlib
import evaluate
import pandas as pd
import chardet
import th_functions as tf
import os  # Add import for file handling
import numpy as np
from seqeval.metrics import classification_report
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import random


def _resolve_model_name():
    env_value = os.environ.get('THESIS_MODEL_NAME')
    if env_value:
        return env_value
    internal_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'dictabert')
    if os.path.exists(os.path.join(internal_model, 'config.json')):
        return internal_model
    return 'dicta-il/dictabert'


def _local_only_enabled():
    return os.environ.get('THESIS_MODEL_LOCAL_ONLY', '0').strip().lower() in ('1', 'true', 'yes', 'on')

class PrepDataSetNERTraining:
    def __init__(self):
        pass

    def load_and_prepare_data(self, file_path: str):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        data = pd.read_csv(file_path, delimiter=',', encoding=result['encoding'])
        return data

    def run_training_steps(self, data):
        sentences = tf.train_data_fit(data)
        train_data, test_data = tf.split_list(sentences, split_ratio=0.7)
        model_name = _resolve_model_name()
        local_only = _local_only_enabled()
        
        model, tokenizer, data_collator, ds_train, ds_eval, ds_test, label_list = tf.setup_token_classification(
            data=data,
            train_data=train_data,
            test_data=test_data,
            eval_data=test_data,
            model_name=model_name,
            local_files_only=local_only,
        )
        trainer1, evaluation_results1 = tf.train_and_evaluate_model(
            model, ds_train, ds_eval, data_collator, tokenizer, label_list, metric_name="seqeval"
        )

 
        return trainer1, evaluation_results1, label_list, ds_eval  # Include ds_eval in the return

    def run_training_with_presplit(self, data, train_sentences, eval_sentences):
        """Train using pre-computed train/eval sentence lists (from experiment 07)."""
        model_name = _resolve_model_name()
        local_only = _local_only_enabled()

        model, tokenizer, data_collator, ds_train, ds_eval, ds_test, label_list = tf.setup_token_classification(
            data=data,
            train_data=train_sentences,
            test_data=eval_sentences,
            eval_data=eval_sentences,
            model_name=model_name,
            local_files_only=local_only,
        )
        trainer, evaluation_results = tf.train_and_evaluate_model(
            model, ds_train, ds_eval, data_collator, tokenizer, label_list, metric_name="seqeval"
        )
        return trainer, evaluation_results, label_list, ds_eval

def prepare_eval_results(eval_ds, trainer, tokenizer, label_list):
    # Get predictions
    preds, _, _ = trainer.predict(eval_ds)
    preds_labels = np.argmax(preds, axis=2)

    def _safe_label_name(label_idx):
        if label_idx == -100:
            return None
        if isinstance(label_idx, (np.integer, int)) and 0 <= int(label_idx) < len(label_list):
            return label_list[int(label_idx)]
        return None

    # Prepare data for tab 1
    sentences = []
    true_labels = []
    pred_labels = []

    for i, item in enumerate(eval_ds):
        input_ids = item['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        true = [_safe_label_name(l) for l in item['labels']]
        pred = [_safe_label_name(p) for p in preds_labels[i]]

        # Remove special tokens (assuming -100 is used for ignored tokens)
        filtered = [(tok, t, p) for tok, t, p, l in zip(tokens, true, pred, item['labels'])
                    if l != -100 and t is not None and p is not None and not tok.startswith("[")]
        if filtered:
            tokens_f, true_f, pred_f = zip(*filtered)
            sentences.append(" ".join(tokens_f))
            true_labels.append(" ".join(true_f))
            pred_labels.append(" ".join(pred_f))

    df_sentences = pd.DataFrame({
        "Sentence": sentences,
        "True Labels": true_labels,
        "Predicted Labels": pred_labels
    })

    # Prepare data for tab 2 (confusion matrix)
    all_true = []
    all_pred = []
    for i, item in enumerate(eval_ds):
        tokens = tokenizer.convert_ids_to_tokens(item['input_ids'])
        for t, p, l, tok in zip(item['labels'], preds_labels[i], item['labels'], tokens):
            true_label = _safe_label_name(t)
            pred_label = _safe_label_name(p)
            if l != -100 and true_label is not None and pred_label is not None and not tok.startswith("["):
                all_true.append(true_label)
                all_pred.append(pred_label)

    from sklearn.metrics import confusion_matrix, classification_report as sk_classification_report, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, accuracy_score, f1_score

    cm = confusion_matrix(all_true, all_pred, labels=label_list)
    cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
    report_dict = sk_classification_report(
        all_true,
        all_pred,
        labels=label_list,
        output_dict=True,
        zero_division=0
    )
    print(report_dict)
    report_df = pd.DataFrame(report_dict).transpose()
    # Calculate additional metrics (TP, FP, FN, TN) for each label
    cm_array = cm.values if hasattr(cm, 'values') else cm
    metrics_list = []
    for i, lbl in enumerate(label_list):
        TP = cm_array[i, i]
        FP = cm_array[:, i].sum() - TP
        FN = cm_array[i, :].sum() - TP
        TN = cm_array.sum() - (TP + FP + FN)
        metrics_list.append([lbl, TP, FP, FN, TN])

    extra_df = pd.DataFrame(metrics_list, columns=["Label", "TP", "FP", "FN", "TN"]).set_index("Label")
    report_df = report_df.join(extra_df, how="left")
    print(report_df.head(2))

    # Compute global metrics including label "O"
    acc = accuracy_score(all_true, all_pred)
    bal_acc = balanced_accuracy_score(all_true, all_pred)
    kappa = cohen_kappa_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)
    f1_with_o = f1_score(all_true, all_pred, average="micro")
    f1_macro_with_o = f1_score(all_true, all_pred, average="macro")
    f1_weighted_with_o = f1_score(all_true, all_pred, average="weighted")

    # Compute global metrics excluding label "O"
    all_true_no_o = []
    all_pred_no_o = []
    for t, p in zip(all_true, all_pred):
        if t != "O":
            all_true_no_o.append(t)
            all_pred_no_o.append(p)
    print("all_true_no_o, all_true_no_o")
    print([all_true_no_o, all_pred_no_o])

    acc_no_o = accuracy_score(all_true_no_o, all_pred_no_o)
    bal_acc_no_o = balanced_accuracy_score(all_true_no_o, all_pred_no_o)
    kappa_no_o = cohen_kappa_score(all_true_no_o, all_pred_no_o)
    mcc_no_o = matthews_corrcoef(all_true_no_o, all_pred_no_o)
    print(all_true_no_o, all_pred_no_o)
    f1_no_o = f1_score(all_true_no_o, all_pred_no_o, average="micro")
    f1_macro_no_o = f1_score(all_true_no_o, all_pred_no_o, average="macro")
    f1_weighted_no_o = f1_score(all_true_no_o, all_pred_no_o, average="weighted")
    #print all_true_no_o, all_pred_no_
    global_metrics = {
        "accuracy_with_o": acc,
        "balanced_accuracy_with_o": bal_acc,
        "cohen_kappa_with_o": kappa,
        "matthews_corrcoef_with_o": mcc,
        "f1_micro_with_o": f1_with_o,
        "f1_macro_with_o": f1_macro_with_o,
        "f1_weighted_with_o": f1_weighted_with_o,
        "accuracy_no_o": acc_no_o,
        "balanced_accuracy_no_o": bal_acc_no_o,
        "cohen_kappa_no_o": kappa_no_o,
        "matthews_corrcoef_no_o": mcc_no_o,
        "f1_micro_no_o": f1_no_o,
        "f1_macro_no_o": f1_macro_no_o,
        "f1_weighted_no_o": f1_weighted_no_o
    }

    return df_sentences, cm_df, report_df, [cm, label_list], global_metrics

def save_eval_results_to_excel(df_sentences, cm_df, report_df, output_path, eval_overall=None, global_metrics=None):
    # Write DataFrames to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_sentences.to_excel(writer, sheet_name="Sentences", index=False)
        cm_df.to_excel(writer, sheet_name="ConfusionMatrix")
        
        # Add overall results to the Metrics tab
        if eval_overall:
            overall_df = pd.DataFrame([{
                "precision": eval_overall.get("Precision"),
                "recall": eval_overall.get("Recall"),
                "f1-score": eval_overall.get("F1"),
                "support": None
            }], index=["""overall"""])

            
            report_df = pd.concat([report_df, overall_df], ignore_index=False)
        
        # Add label names as a column in the Metrics tab
        #report_df.insert(0, "Label", report_df.index)
        report_df.to_excel(writer, sheet_name="Metrics", index=True)
        
        # Write global metrics on a new sheet
        if global_metrics:
            df_global = pd.DataFrame(list(global_metrics.items()), columns=["Metric", "Value"])
            df_global.to_excel(writer, sheet_name="GlobalMetrics", index=False)

def train_incremental_subsets(data, subset_sizes=[50, 100, 150, 200, 250], model_name='dicta-il/dictabert'):
    """
    Train models on incrementally increasing random subsets of sentences and measure results.
    
    Args:
        data: The full dataset
        subset_sizes: List of subset sizes to train on
        model_name: The model to use for training
    
    Returns:
        results_df: DataFrame with performance metrics for each subset size
    """
    # Prepare all sentences once
    all_sentences = tf.train_data_fit(data)
    
    # Filter subset sizes based on available data
    max_sentences = len(all_sentences)
    valid_subset_sizes = [size for size in subset_sizes if size <= max_sentences]
    
    if not valid_subset_sizes:
        print(f"No valid subset sizes. Maximum available sentences: {max_sentences}")
        return pd.DataFrame()
    
    print(f"Training on subset sizes: {valid_subset_sizes} (max available: {max_sentences})")
    
    results = []
    
    # Use the same test set for all experiments for fair comparison
    _, test_data = tf.split_list(all_sentences, split_ratio=0.8)
    
    for subset_size in valid_subset_sizes:
        print(f"\nTraining on {subset_size} sentences...")
        
        # Randomly sample sentences for training
        random.seed(42)  # For reproducibility
        train_subset = random.sample(all_sentences, subset_size)
        
        # Setup model and training
        model, tokenizer, data_collator, ds_train, ds_eval, ds_test, label_list = tf.setup_token_classification(
            data=data,
            train_data=train_subset,
            test_data=test_data,
            eval_data=test_data,
            model_name=model_name
        )
        
        # Train and evaluate
        trainer, evaluation_results = tf.train_and_evaluate_model(
            model, ds_train, ds_eval, data_collator, tokenizer, label_list, metric_name="seqeval"
        )
        
        # Extract metrics
        f1_score = evaluation_results.get("eval_overall_f1", None)
        precision = evaluation_results.get("eval_overall_precision", None)
        recall = evaluation_results.get("eval_overall_recall", None)
        loss = evaluation_results.get("eval_loss", None)
        
        # Get detailed predictions for additional metrics
        df_sentences, cm_df, report_df, [cm, label_list_used], global_metrics = prepare_eval_results(
            ds_eval, trainer, tokenizer, label_list
        )
        
        # Store results
        result_row = {
            'subset_size': subset_size,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'eval_loss': loss,
            'accuracy_with_o': global_metrics.get('accuracy_with_o'),
            'accuracy_no_o': global_metrics.get('accuracy_no_o'),
            'f1_micro_with_o': global_metrics.get('f1_micro_with_o'),
            'f1_macro_with_o': global_metrics.get('f1_macro_with_o'),
            'f1_micro_no_o': global_metrics.get('f1_micro_no_o'),
            'f1_macro_no_o': global_metrics.get('f1_macro_no_o'),
            'cohen_kappa_with_o': global_metrics.get('cohen_kappa_with_o'),
            'cohen_kappa_no_o': global_metrics.get('cohen_kappa_no_o'),
            'matthews_corrcoef_with_o': global_metrics.get('matthews_corrcoef_with_o'),
            'matthews_corrcoef_no_o': global_metrics.get('matthews_corrcoef_no_o')
        }
        results.append(result_row)
        
        print(f"Results for {subset_size} sentences: F1={f1_score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df

def save_incremental_results(results_df, output_path):
    """
    Save the incremental training results to Excel.
    
    Args:
        results_df: DataFrame with results for each subset size
        output_path: Path to save the Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name="IncrementalResults", index=False)
    
    print(f"Incremental training results saved to {output_path}")

def main():
    results_dir = "training_eval"
    os.makedirs(results_dir, exist_ok=True)
    excel_output = os.path.join(results_dir, "evaluation_full_results1.2.xlsx")

    # Run training and save results if no saved file exists
    worker = PrepDataSetNERTraining()
    file_path = "ner_dataset.csv"
    data = worker.load_and_prepare_data(file_path)
    # trainer, eval_results, label_list, ds_eval = worker.run_training_steps(data)
    # print(eval_results)

    # # Extract eval_overall metrics
    # eval_overall = {
    #     "F1": eval_results.get("eval_overall_f1", None),
    #     "Precision": eval_results.get("eval_overall_precision", None),
    #     "Recall": eval_results.get("eval_overall_recall", None)
    # }

    # # Split function usage
    # df_sentences, cm_df, report_df , [cm,label_list], global_metrics = prepare_eval_results(ds_eval, trainer, trainer.tokenizer, label_list)
    # save_eval_results_to_excel(df_sentences, cm_df, report_df, excel_output, eval_overall, global_metrics)
    # print(f"Evaluation results saved to {excel_output}")

    # Incremental training
    incremental_results_df = train_incremental_subsets(data, subset_sizes=[50, 100, 150, 200, 250], model_name='dicta-il/dictabert')
    if not incremental_results_df.empty:
        incremental_excel_output = os.path.join(results_dir, "incremental_training_results.xlsx")
        save_incremental_results(incremental_results_df, incremental_excel_output)

if __name__ == "__main__":
    main()