import argparse
import torch
from data import load_data, prepare_data
from model import init_model
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter for Training T5 for QA')
    parser.add_argument('--model_checkpoint', default='paust/pko-t5-large', type=str)
    parser.add_argument('--prefix', default='qa question: ', type=str,
                                                help='inference input prefix')
    parser.add_argument('--max_input_length', default=32, type=int,
                                                help='max input length for qa')
    parser.add_argument('--max_target_length', default=32, type=int,
                                                help='max target length for qa')
    parser.add_argument('--train_batch_size', default=8, type=int,
                                                help='train batch size')
    parser.add_argument('--eval_batch_size', default=8, type=int,
                                                help='eval batch size')
    parser.add_argument('--num_train_epochs', default=100, type=int,
                                                help='train epoch size')
    parser.add_argument('--lr', default=7e-4, type=int,
                                                help='learning rate for training')
    parser.add_argument('--wd', default=0.01, type=int,
                                                help='weight decay for training')
    parser.add_argument('--steps', default=30000, type=int,
                                                help='evaluation, logging, saving step for training')                                            
    parser.add_argument('--model_name', default='paust/pko-t5-base', type=str,
                                                help='model name for saving')
    parser.add_argument('--train_path', default='../data/comuchat_train.csv', type=str,
                                                help='dataset path')
    parser.add_argument('--test_path', default='../data/comuchat_valid.csv', type=str,
                                                help='dataset path')
    parser.add_argument('--model_path', default='./model', type=str,
                                                help='model path for saving')
    args = parser.parse_args()

    # Load datset
    df_train, df_test = load_data(args.train_path, args.test_path)
    
    # Load model & tokenizer
    # model_name = 'paust/pko-t5-base'
    tokenizer, model = init_model(args.model_name)
    
    # tokenizer에 넣기
    training_set = prepare_data(df_train, tokenizer, args)
    test_set = prepare_data(df_test, tokenizer, args)
    # set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' #torch.device('cuda:3')
    
    model.to(device)
    
    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.AdamW(params =  model.parameters(), lr=args.lr) 

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir = args.model_path,
        save_total_limit=1,
        early_stopping_patience=5,
        evaluation_strategy = "epoch",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        save_steps=1000,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_steps=3000,
        logging_steps=1000,
        save_strategy="no",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
    )
    
    trainer = Seq2SeqTrainer(
        model = model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=test_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    # Training
    print('Start Training...')  
    
    trainer.train()
    
    # Saving model
    print('Saving Model...')
    trainer.save_model()
    
