import argparse
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration, T5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm

def model_init(args):
        # vanilla model일 경우 arg대신 아래와 같이 모델이름을 넣어주면 됨
        model_name = 'paust/pko-t5-large'
        #model = T5ForConditionalGeneration.from_pretrained(args.model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_name)                
        #tokenizer = T5TokenizerFast.from_pretrained(args.model_path)
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
        # add tokens
        tokenizer.add_tokens(["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "(여자)화자"])
        model.resize_token_embeddings(len(tokenizer))
        model.config.max_length = args.max_target_length
        tokenizer.model_max_length = args.max_target_length
        return model, tokenizer

def sequantially_chat(model, tokenizer, args):
        while True:
                user_inputs = input("user: ")
                if user_inputs == 'false':
                        break
                inputs = [args.prefix + user_inputs]
                inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True, return_tensors="pt")
                output = model.generate(**inputs, num_beams=2, do_sample=True, min_length=10, max_length=args.max_target_length, no_repeat_ngram_size=2) #repetition_penalty=2.5
                result = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                result = str(result)
                result = result.replace('[', '').replace(']', '').replace("'", '').replace("'", '')
                print(f'chatbot: {result}')

def generate_from_file(model, tokenizer, args):
        file = pd.read_csv(args.qpath)
        q_list = file['question']
        result_list = []
        for f in tqdm(q_list):
                inputs = [args.prefix + f]
                inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True, return_tensors="pt")
                output = model.generate(**inputs, num_beams=2, do_sample=True, min_length=10, max_length=args.max_target_length, no_repeat_ngram_size=2) #repetition_penalty=2.5
                result = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                result = str(result)
                result = result.replace('[', '').replace(']', '').replace("'", '').replace("'", '')
                result_list.append(result)
        result_df = pd.DataFrame({
                'Query': q_list,
                'Answer': result_list
        })
        result_df.to_csv(args.outputpath, encoding='utf-8-sig', index=False)
        
if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Hyperparameter for Chat')
        parser.add_argument('--model_path', default='/home/work/deeptext/COMU-ChatBot/model/pko_t5_wel_patience10', type=str,
                                                help='model path for chat') 
        parser.add_argument('--max_input_length', default=64, type=int,
                                                help='user max input length')
        parser.add_argument('--max_target_length', default=64, type=int,
                                                help='max target length')
        parser.add_argument('--prefix', default='qa question: ', type=str,
                                                help='inference input prefix')
        parser.add_argument('--qpath', default='/home/work/deeptext/COMU-ChatBot/data/qlist.csv', type=str,
                                                help='question list')
        parser.add_argument('--outputpath', default='./output.csv', type=str,
                                                help='output file save')
        args = parser.parse_args()

        model, tokenizer = model_init(args)
        generate_from_file(model, tokenizer, args)
        print('done.')
        # sequantially_chat(model, tokenizer, args)