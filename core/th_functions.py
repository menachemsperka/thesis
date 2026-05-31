import random
import os

import warnings
import pandas as pd
import evaluate
import torch
import inspect

from transformers import AutoModelForTokenClassification, AutoConfig, Trainer, AutoTokenizer, DataCollatorForTokenClassification, pipeline, TrainingArguments

_INTERNAL_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'dictabert')
DEFAULT_MODEL_NAME = _INTERNAL_MODEL if os.path.exists(os.path.join(_INTERNAL_MODEL, 'config.json')) else 'dicta-il/dictabert'



def balance_with_gai(train_data, data, g_type = 1):
    original_train_data = train_data.copy()
    s_st = generate_label_df(train_data)
    #display(s_st)
    gen_results = []
    
    label_list = s_st['Label'].tolist()
    print(label_list)
    
    for label_tg in label_list:
        print(label_tg)
        temp_list= filter_items_by_label_suffix(original_train_data, label_suffix= label_tg)
        score_list = []
        for sent in temp_list:
            scores, final_score = calculate_label_score(sent, s_st, label_tg)
            score_list.append([sent,final_score])
        score_list.sort(key=lambda x: x[1])
        total_to_generate = s_st.loc[s_st['Label']==label_tg, 'Delta to Max']
        total_to_generate = total_to_generate.iloc[0]
        print(total_to_generate)
        sent_list = distribute_sentence_assignment(score_list, total_to_generate)

        for item in sent_list[0:]:
            sent= item[0]
            to_gen_count  = item[2]
            #print(item)
            #print('')
            #print('')
            #print(f"***Generating for: {[sent, label_tg,  to_gen_count]} ****")
            if g_type == 1:
                temp_result =  generate_sentences(data, sent, label_tg, to_gen_count)
            elif g_type ==2:
                temp_result =  duplicate_sentences(sent, to_gen_count)
            gen_results.append(temp_result)

            train_data = train_data + train_data_fit(assign_ids(pd.concat(gen_results, ignore_index=True), start_id=1000000))
        s_st = generate_label_df(train_data)
        #display(s_st)
        res = pd.concat(gen_results, ignore_index=True)
        res.to_csv('temp_res.csv')
    return res
    
def train_and_evaluate_model(model, ds_train, ds_eval, data_collator, tokenizer, label_list, metric_name="seqeval", output_path=None):
    """
    Train and evaluate a model using the provided datasets and parameters.

    Parameters:
    - model: The model to be trained.
    - ds_train: The training dataset.
    - ds_eval: The evaluation dataset.
    - data_collator: The data collator for batching.
    - tokenizer: The tokenizer used for preprocessing.
    - label_list: The list of labels for evaluation.
    - metric_name: The name of the evaluation metric to load (default is "seqeval").
    - output_path: Custom output directory for the final model (used particularly in Colab workflows).
    """
    # Load the evaluation metric (offline-safe fallback)
    metric = None
    try:
        metric = evaluate.load(metric_name)
    except Exception:
        metric = None

    epochs_raw = (os.environ.get("THESIS_NUM_EPOCHS") or "").strip()
    try:
        num_train_epochs = float(epochs_raw) if epochs_raw else 3.0
    except ValueError:
        num_train_epochs = 3.0

    is_colab = os.environ.get("THESIS_RUN_ENV") == "colab"
    
    if is_colab:
        # Colab-specific training arguments
        out_dir = output_path if output_path else os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "trainer_output", "final_model")
        colab_kwargs = {
            "output_dir": out_dir,
            "num_train_epochs": num_train_epochs,
            "save_strategy": "steps",
            "save_steps": 100,
            "eval_steps": 100,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "overall_f1",
        }
        # Handle recent Transformers versions where evaluation_strategy was renamed to eval_strategy
        sig = inspect.signature(TrainingArguments.__init__)
        if "eval_strategy" in sig.parameters:
            colab_kwargs["eval_strategy"] = "steps"
        else:
            colab_kwargs["evaluation_strategy"] = "steps"
            
        training_args = TrainingArguments(**colab_kwargs)
    else:
        # Default local training arguments
        out_dir = output_path if output_path else os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "tmp_trainer")
        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=num_train_epochs,
            save_strategy="no",
        )
    
    # Initialize the Trainer (Transformers compatibility: tokenizer -> processing_class)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": ds_train,
        "eval_dataset": ds_eval,
        "data_collator": data_collator,
        "compute_metrics": lambda p: compute_metrics(p, label_list, metric),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    
    # Train the model
    trainer.train()
    
    # Save the model if THESIS_SAVE_TRAINED_MODELS is set
    save_models_flag = (os.environ.get("THESIS_SAVE_TRAINED_MODELS") or "").strip() == "1"
    if save_models_flag:
        # Build unique model save path based on experiment context
        model_save_base = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "outputs", "trained_models"
        )
        os.makedirs(model_save_base, exist_ok=True)
        
        # Use environment variables to build a unique identifier
        exp_id = os.environ.get("THESIS_CURRENT_EXP_ID", "unknown_exp")
        model_id = os.environ.get("THESIS_MODEL_NAME", "unknown_model")
        condition_key = os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default")
        seed = os.environ.get("THESIS_SPLIT_SEED", "42")
        
        # Sanitize model_id (remove path components)
        model_short = model_id.replace("/", "_").replace("\\", "_").split("_")[-1]
        save_name = f"{exp_id}_{model_short}_{condition_key}_seed{seed}"
        model_save_path = os.path.join(model_save_base, save_name)
        
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"[Model Saved] {model_save_path}")
    
    # Evaluate the model
    evaluation_results = trainer.evaluate()
    
 
    #print(evaluation_df)
    return trainer, evaluation_results

# Example usage:
# train_and_evaluate_model(model, ds_train, ds_eval, data_collator, tokenizer, label_list)

def setup_token_classification(data, train_data, eval_data,  test_data,  model_name=DEFAULT_MODEL_NAME, local_files_only=False):
    label_list = data.raw_tags.dropna().astype(str).unique().tolist()
    label_to_id = {label: idx for idx, label in enumerate(label_list)}

    # Load config explicitly to strip fields that can leak between models
    # in the same process (e.g. finetuning_task from a prior DictaBERT load).
    config = AutoConfig.from_pretrained(
        model_name,
        id2label={idx: label for label, idx in label_to_id.items()},
        label2id=label_to_id,
        num_labels=len(label_to_id),
        local_files_only=local_files_only,
    )
    for attr in ("finetuning_task",):
        if hasattr(config, attr):
            delattr(config, attr)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        local_files_only=local_files_only,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    class TokenClassificationTorchDataset(torch.utils.data.Dataset):
        def __init__(self, records):
            self.features = []
            for record in records:
                words = str(record["text"]).split()
                labels = [str(label) for label in record["labels"]]
                encoding = tokenizer(words, is_split_into_words=True, truncation=True)
                try:
                    word_ids = encoding.word_ids()
                except Exception:
                    word_ids = [None] + list(range(min(len(words), max(0, len(encoding["input_ids"]) - 2)))) + [None]
                    if len(word_ids) < len(encoding["input_ids"]):
                        word_ids.extend([None] * (len(encoding["input_ids"]) - len(word_ids)))
                    elif len(word_ids) > len(encoding["input_ids"]):
                        word_ids = word_ids[:len(encoding["input_ids"])]

                aligned_labels = []
                prev_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        aligned_labels.append(-100)
                    elif word_idx != prev_word_idx:
                        label_name = labels[word_idx] if word_idx < len(labels) else "O"
                        aligned_labels.append(label_to_id.get(label_name, label_to_id.get("O", 0)))
                    else:
                        aligned_labels.append(-100)
                    prev_word_idx = word_idx

                encoding["labels"] = aligned_labels
                self.features.append(encoding)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx]

    ds_train = TokenClassificationTorchDataset(train_data)
    ds_eval = TokenClassificationTorchDataset(eval_data)
    ds_test = TokenClassificationTorchDataset(test_data)

    return model, tokenizer, data_collator, ds_train, ds_eval, ds_test, label_list

def compute_metrics(p, label_list, metric):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[pred] for (pred, lab) in zip(prediction, label)
         if lab != -100 and not label_list[pred].startswith("[")]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lab] for (pred, lab) in zip(prediction, label)
         if lab != -100 and not label_list[lab].startswith("[")]
        for prediction, label in zip(predictions, labels)
    ]

    if metric is not None:
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return results

    try:
        from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            "overall_precision": precision_score(true_labels, true_predictions),
            "overall_recall": recall_score(true_labels, true_predictions),
            "overall_f1": f1_score(true_labels, true_predictions),
            "overall_accuracy": accuracy_score(true_labels, true_predictions),
        }
    except Exception:
        return {
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "overall_f1": 0.0,
            "overall_accuracy": 0.0,
        }


def calculate_raw_tag_statistics(df):
    # Group by 'raw_tags' and calculate the count of instances and distinct IDs
    result = df.groupby('raw_tags').agg(
        Count_of_Instances=('raw_tags', 'size'),
        Count_of_Distinct_IDs=('id', 'nunique')
    ).reset_index()
    
    return result

def merge_subtokens(df):
    """
    Merge rows with subtokens (starting with ##) into one row and use the first label for the full word.
    """
    print(df.head(5))
    merged_rows = []
    current_token = ""
    current_label = None

    for _, row in df.iterrows():
        token = row['token']
        label = row['raw_tags'] if 'raw_tags' in row else row['label_tag']

        if token.startswith("##"):
            # Append subtoken to the current token
            current_token += token.replace("##", "")
        else:
            # If a new token starts, save the previous token and label
            if current_token:
                merged_rows.append({'token': current_token, 'label': current_label})
            # Start a new token
            current_token = token
            current_label = label

    # Add the last token
    if current_token:
        merged_rows.append({'token': current_token, 'label': current_label})
    print(pd.DataFrame(merged_rows))
    # Create a DataFrame from merged rows
    return pd.DataFrame(merged_rows)


def transform_df(df):
    #print("Columns in DataFrame:", df.columns)
    # Filter out the CLS and SEP tokens
    df.to_csv('before.csv', encoding='utf-8-sig')
    df = merge_subtokens(df)
    df.to_csv('afetr.csv',encoding='utf-8-sig')
    filtered_df = df[~df['token'].isin(['[CLS]', '[SEP]'])]

    # Concatenate all tokens into one text
    text = ' '.join(filtered_df['token'].tolist())
   

    try:
        #print(filtered_df.columns)
        labels = filtered_df['label'].tolist()
        #print("Using 'raw_tags' for labels") 
    except KeyError:
    # Create a list of raw tags
        labels = filtered_df['tags'].tolist()

    # Create the result dictionary
    result = {'text': text, 'labels': labels}

    return result

def filter_df(df, column_name , x,y):
    #filter on rows where row[i] = x but row[i+1] != y
    # Create the boolean masks
    mask_x = df[column_name] == x
    mask_not_y_next = df[column_name].shift(-1) != y

    # Combine the masks
    result_mask = mask_x & mask_not_y_next
    return df[result_mask]

def _extract_non_o_labels(sentence_item):
    labels = sentence_item.get('labels', []) if isinstance(sentence_item, dict) else []
    return {str(label) for label in labels if str(label) != 'O'}


def _extract_non_o_label_counts(sentence_item):
    labels = sentence_item.get('labels', []) if isinstance(sentence_item, dict) else []
    counts = {}
    for label in labels:
        key = str(label)
        if key == 'O':
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _resolve_split_seed(seed):
    if seed is not None:
        return seed
    env_seed = os.environ.get('THESIS_SPLIT_SEED')
    if env_seed is None:
        return None
    try:
        return int(env_seed)
    except ValueError:
        return None


def split_list(items, split_ratio=0.8, seed=None, ensure_label_coverage=True):
    items_list = list(items)
    if not items_list:
        return [], []

    if len(items_list) == 1:
        return items_list, []

    resolved_seed = _resolve_split_seed(seed)
    rng = random.Random(resolved_seed) if resolved_seed is not None else random.Random()
    shuffled_items = list(items_list)
    rng.shuffle(shuffled_items)

    split_index = int(len(shuffled_items) * split_ratio)
    split_index = max(1, min(len(shuffled_items) - 1, split_index))

    label_counts_per_item = [_extract_non_o_label_counts(item) for item in shuffled_items]
    total_counts = {}
    for item_counts in label_counts_per_item:
        for label, count in item_counts.items():
            total_counts[label] = total_counts.get(label, 0) + count

    if not total_counts:
        return shuffled_items[:split_index], shuffled_items[split_index:]

    target_counts = {label: total * split_ratio for label, total in total_counts.items()}
    selected_indices = set()
    current_counts = {label: 0 for label in total_counts}

    while len(selected_indices) < split_index:
        best_idx = None
        best_score = None
        for idx in range(len(shuffled_items)):
            if idx in selected_indices:
                continue

            item_counts = label_counts_per_item[idx]
            score = 0.0
            if item_counts:
                for label, add_count in item_counts.items():
                    before = current_counts[label] - target_counts[label]
                    after = (current_counts[label] + add_count) - target_counts[label]
                    score += (after * after) - (before * before)
                if ensure_label_coverage:
                    newly_covered = sum(1 for label in item_counts if current_counts[label] == 0)
                    score -= 1000.0 * newly_covered

            score += rng.random() * 1e-6
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.add(best_idx)
        for label, add_count in label_counts_per_item[best_idx].items():
            current_counts[label] += add_count

    train_data = [shuffled_items[idx] for idx in range(len(shuffled_items)) if idx in selected_indices]
    eval_data = [shuffled_items[idx] for idx in range(len(shuffled_items)) if idx not in selected_indices]
    return train_data, eval_data

def generate_label_df(data):
    # Initialize a dictionary to store label information
    label_info = {}

    # Iterate over each list in the data
    for item in data:
        text = item['text']
        labels = item['labels']
        
        # Iterate over each label in the labels list
        for label in labels:
            if label != 'O':  # Ignore 'O' labels
                # Split the label by the delimiter and get the second part
                parts = label.split('-')
                if len(parts) > 1:
                    second_part = parts[1]
                else:
                    continue  # Skip if there's no second part

                if second_part not in label_info:
                    label_info[second_part] = {'count': 0, 'sentences': set()}
                
                # Increment the count for the second part of the label
                label_info[second_part]['count'] += 1
                
                # Add the sentence to the set of sentences for the second part of the label
                label_info[second_part]['sentences'].add(text)

    # Prepare data for DataFrame
    df_data = []
    for label, info in label_info.items():
        df_data.append({
            'Label': label,
            'Instance Count': info['count'],
            'Distinct Sentence Count': len(info['sentences'])
        })

    # Create DataFrame
    df = pd.DataFrame(df_data).sort_values(by='Distinct Sentence Count', ascending=True).reset_index(drop=True)
    # Calculate the maximum "Distinct Sentence Count"
    max_distinct_sentence_count = df['Distinct Sentence Count'].max()

    # Add a new column for the delta
    df['Delta to Max'] = max_distinct_sentence_count - df['Distinct Sentence Count']
    return df


def filter_items_by_label_suffix(data, label_suffix):
    """
    Filters items from the list of dictionaries based on the specified label suffix.

    :param data: List of dictionaries, each containing 'text' and 'labels'.
    :param label_suffix: The label suffix to filter by (text after the '-').
    :return: List of dictionaries that contain the specified label suffix.
    """
    filtered_items = []

    for item in data:
        for label in item['labels']:
            # Split the label by '-' and check the suffix
            parts = label.split('-')
            if len(parts) > 1 and parts[1] == label_suffix:
                filtered_items.append(item)
                break  # No need to check further labels in this item

    return filtered_items

    
def calculate_label_score(data, df, exclude_label=None):
    """
    Calculates the score based on the count of instances of each label suffix,
    multiplied by the corresponding Distinct Sentence Count from the DataFrame,
    excluding a specified label.

    :param data: List of dictionaries, each containing 'text' and 'labels'.
    :param df: DataFrame containing label information and Distinct Sentence Count.
    :param exclude_label: The label suffix to exclude from the score calculation.
    :return: A dictionary with label suffixes as keys and their calculated scores as values.
    """
    label_counts = {}

    # Count occurrences of each label suffix

    for label in data['labels']:
        parts = label.split('-')
        if len(parts) > 1:
            suffix = parts[1]
            if suffix in label_counts:
                label_counts[suffix] += 1
            else:
                label_counts[suffix] = 1

    # Calculate scores using Distinct Sentence Count from the DataFrame
    scores = {}
    final_score = 0
    for suffix, count in label_counts.items():
        if suffix != exclude_label and suffix in df['Label'].values:
            distinct_sentence_count = df.loc[df['Label'] == suffix, 'Distinct Sentence Count'].values[0]
            score = count * distinct_sentence_count
            scores[suffix] = score
            final_score+=score

    return scores, final_score



def assign_ids(df, start_id=1000000):
    """
    Assigns a running ID to each row in the DataFrame, starting from start_id.
    The ID is incremented each time a '[SEP]' token is encountered followed by an empty token.
    """
    current_id = start_id
    ids = []
    num_rows = len(df)
    
    for index, row in df.iterrows():
        ids.append(current_id)
        
        # Check if the current row's token is '[SEP]' and the next row's token is ''
        if row['token'] == '' and index + 1 < num_rows and df.iloc[index + 1]['token'] == '[CLS]':
            current_id += 1
    
    df['id'] = ids
    return df

def generate_sentences(train_data, data, lbl_to_generate, n):

    # Suppress specific FutureWarning messages
    warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

    # Ensure n is an integer
    n = int(n)*5+1
    tokens = data['text'].split()
    rows = []
    for i, token in enumerate(tokens):
        row = {
            'token': token,
            #'label': data['labels'][i],
            'label_tag': data['labels'][i],
        }
        rows.append(row)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)
    #lbl_to_generate = "B-BOK"

    #df['model_input'] = df.apply(lambda r: "[MASK]" if r['label_tag'] == 'I-' + lbl_to_generate or r['label_tag'] == 'B-' + lbl_to_generate else r['token'], axis = 1)
    #df['model_input'] = df.apply(lambda r: "[MASK]" if r['label'] == lbl_to_generate else r['token'], axis=1)

    # Initialize the new column with None
    df['model_input'] = None

    # Iterate through the DataFrame to set the appropriate values
    consecutive = False
    mask_set = False  # Flag to track if the first [MASK] has been set

    for index, row in df.iterrows():
        if mask_set:
            # If the mask has been set, just copy the token
            df.at[index, 'model_input'] = row['token']
        elif row['label_tag'] == 'I-' + lbl_to_generate or row['label_tag'] == 'B-' + lbl_to_generate:
            df.at[index, 'model_input'] = '[MASK]'
            mask_set = True  # Set the flag to True after setting the first [MASK]
        else:
            df.at[index, 'model_input'] = row['token']

    #print(df)
    # Join the tokens into one string
    joined_text = ' '.join(df['model_input'])

    # Display the joined text
    if 1==2: print([joined_text,n])


    result_df = filter_df(train_data, 'raw_tags', f"B-{lbl_to_generate}", f"I-{lbl_to_generate}")
    #display(result_df.head())
    distinct_values = result_df['token'].unique().tolist()

    fill_mask = pipeline("fill-mask", model="dicta-il/dictabert", top_k=n)
    predictions = fill_mask(joined_text, targets = distinct_values)

    # Print the predictions
    #for prediction in predictions:
        #print(f"Token: {prediction['token_str']}, Score: {prediction['score']:.4f}")
    generated_dfs = []
    res_list = []


    for prediction in predictions:
        # Create a new DataFrame for the generated sentence
        generated_df = df[df['model_input'] != ''].copy()
        #print( prediction) 
        #print( prediction['token_str']) 
        #print(generated_df)
        # Replace the [MASK] token with the predicted token
        mask_replaced = False
        for index, row in generated_df.iterrows():
            if row['model_input'] == '[MASK]' and not mask_replaced:
                
                generated_df.at[index, 'model_output'] = prediction['token_str']
                mask_replaced = True
            else:
                generated_df.at[index, 'model_output'] = row['token']


            # Remove columns: token and model_input
        generated_df.drop(columns=['token', 'model_input'], inplace=True)

        # Rename model_output to 'token'
        generated_df.rename(columns={'model_output': 'token'}, inplace=True)

        # Add a row at the top
        cls_row = pd.DataFrame([{'token': '[CLS]', 'label': 10, 'label_tag': 'O'}])
        sep_row = pd.DataFrame([{'token': '[SEP]', 'label': 10, 'label_tag': 'O'}])
        empty_row = pd.DataFrame([{'token': '', 'label': 10, 'label_tag': 'O'}])
        generated_df = pd.concat([cls_row, generated_df,sep_row,empty_row ], ignore_index=True)
        # Append the generated DataFrame to the list
        generated_dfs.append(generated_df)
    # Concatenate all generated DataFrames into one
    if generated_dfs:
        final_df = pd.concat(generated_dfs, ignore_index=True)
    else:
        # Handle the case where no DataFrames were generated
        final_df = pd.DataFrame()  # or some other appropriate action

    #final_df = assign_ids(pd.concat(generated_dfs, ignore_index=True), start_id=1000000)
    final_df.rename(columns={'label_tag': 'raw_tags', 'label':'ner_tag'}, inplace=True)
    return final_df


def duplicate_sentences( data, n):
    #print(n)
    #print(data)
    # Ensure n is an integer
    n = 5*int(n)+1
    tokens = data['text'].split()
    rows = []
    for i, token in enumerate(tokens):
        row = {
            'token': token,
            #'label': data['labels'][i],
            'label_tag': data['labels'][i],
        }
        rows.append(row)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)
    # Check if the DataFrame is empty
    if df.empty:
        print("Error: DataFrame is empty. No objects to concatenate.")
        return df  # Return an empty DataFrame or handle the error appropriately

    # Add a row at the top
    cls_row = pd.DataFrame([{'token': '[CLS]', 'label': 10, 'label_tag': 'O'}])
    sep_row = pd.DataFrame([{'token': '[SEP]', 'label': 10, 'label_tag': 'O'}])
    empty_row = pd.DataFrame([{'token': '', 'label': 10, 'label_tag': 'O'}])
    
    # Concatenate the additional rows with the original DataFrame
    modified_df = pd.concat([cls_row, df, sep_row, empty_row], ignore_index=True)

    # Duplicate the modified DataFrame n times
    final_df = pd.concat([modified_df] * n, ignore_index=True)
    
    return final_df



def distribute_sentence_assignment(ids_scores, total_to_generate):
    # Calculate the inverse of scores
    #inverse_scores = [1 / score for _, score in ids_scores]
    inverse_scores = [1 / (score if score != 0 else 1) for _, score in ids_scores]
    # Calculate the total of inverse scores
    total_inverse_score = sum(inverse_scores)
    
    # Calculate the dollar distribution based on inverse scores
    distribution = []
    for (id, score), inverse_score in zip(ids_scores, inverse_scores):
        allocated_dollars = round((inverse_score / total_inverse_score) * total_to_generate,0)
        distribution.append((id, score, allocated_dollars))
    
    return distribution

def train_data_fit(data_df):
    sentences = []



    # Use the detected encoding
    data = data_df
    #display(data.iloc[0:15])
    # Assuming 'text' column contains the text you want to fine-tune on
    # Group the data by sentences (identified by the 'id' column)
    data = data.dropna(subset=['token'])
    grouped_data = data.groupby("id", as_index=False)

    for _, group in grouped_data:
        res_dict_temp = transform_df(group)
        if len(set(res_dict_temp['labels'])) >1: sentences.append(res_dict_temp)
    return sentences