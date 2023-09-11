# Importing the T5 modules from huggingface/transformers
from transformers import T5ForConditionalGeneration ,T5TokenizerFast

def init_model(model_name):
    #model_name = 'paust/pko-t5-large'
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # add tokens
    tokenizer.add_tokens(["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "(여자)화자"])
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model