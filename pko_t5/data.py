#필요 라이브러리 임포트
import pandas as pd

# 데이터 불러오기
def load_data(train_path, test_path):    
    df_train = pd.read_csv(train_path)
    df_train = df_train[['Q', 'A']]
    df_train = df_train.reset_index(drop = True)
    
    df_test = pd.read_csv(test_path)
    df_test = df_test[['Q', 'A']]
    df_test = df_test.reset_index(drop = True)
    
    df_train.Q = 'qa question: ' + df_train.Q
    df_test.Q = 'qa question: ' + df_test.Q
    
    print("=========data shape=========")
    print("Number of train Dataset: {}".format(df_train.shape))
    print("Number of test Dataset: {}".format(df_test.shape))
    
    return df_train, df_test

def prepare_data(examples, tokenizer, args):
    all_inputs = [i for i in examples['Q'].tolist()]
    inputs = tokenizer(all_inputs, add_special_tokens=True, max_length=args.max_input_length, truncation=True, padding=True, return_tensors='np')
    input_ids = inputs.input_ids.tolist()
    attention_mask = inputs.attention_mask.tolist()

    all_labels = [i for i in examples['A'].tolist()]
    targets = tokenizer(all_labels, add_special_tokens=True, max_length=args.max_target_length, truncation=True, padding=True, return_tensors='np')
    labels = targets.input_ids
    labels[~targets.attention_mask] = -100
    labels = labels.tolist()
    dataset = [{'input_ids': row[0], 'attention_mask': row[1], 'labels': row[2]} for row in zip(input_ids, attention_mask, labels)]

    return dataset
