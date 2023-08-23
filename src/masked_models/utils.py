import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def mask_logprob(masked_tokens, original_tokens, tokenizer, model, diagnose=False):
    """
    Calculate mean logprob for masked tokens.

    1. Make prediction for masked batch encoding `masked_tokens`.
    2. Calculate probabilities for expected ids for masked tokens. Use
       `original_tokens` batch encoding to extract expected token ids.
    3. Return mean of their logprobs.
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    probs = model(**masked_tokens.to(device)).logits.softmax(dim=-1)
    probs_true = torch.gather(  # Probabilities only for the expected token ids
        probs[0],
        dim=1,
        index=torch.t(original_tokens['input_ids'].to(device))
    )
    mask_indices = masked_tokens['input_ids'][0] == tokenizer.mask_token_id
    logprob = torch.mean(torch.log10(probs_true[mask_indices]))
    if diagnose:
        print('Probs:', probs)
        print('Probs for correct tokens:', probs_true)
        print('Probs for masked tokens:', probs_true[mask_indices])
        print('Log of their mean:', logprob)
    logprob = logprob.item()

    return logprob


def sentence_logprob(sen1, sen2, tokenizer, model, diagnose=False):
    """
    Calculate `mask_logprob` for `sentence`. Sentence is expected to have
    a <bracketed> keyword. `lru_cache` is used. Run this cell to clear the cache.
    """
    original_tokens = tokenize(sen1, tokenizer)
    masked_tokens = tokenize_with_mask(sen1, sen2, tokenizer)
    logprob = mask_logprob(masked_tokens, original_tokens, tokenizer, model, diagnose)
    if diagnose:
        print('Original sentence:', sen1)
        print('Token ids:', original_tokens['input_ids'][0])
        print('Token ids (masked):', masked_tokens['input_ids'][0])
        print('Tokens:', ', '.join('`' + tokenizer.decode([t]) + '`' for t in original_tokens['input_ids'][0]))
        print('Decoded token ids:', tokenizer.decode(original_tokens['input_ids'][0]))
        print('Decoded token ids (masked):', tokenizer.decode(original_tokens['input_ids'][0]))
    return logprob


def model_init(model_name):
    model, tokenizer = AutoModelForMaskedLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda:0')
    return model, tokenizer


def tokenize(sen, tokenizer, only_ids=False, **kwargs):
    """
    Use `tokenizer` to parse sentence `sen`.
    
    `only_ids` - Return only token ids if True, `BatchEncoding` otherwise.
    `kwargs` - Are sent to tokenizer.
    """
    batch_encoding = tokenizer(sen, return_tensors="pt", **kwargs)
    if only_ids:
        return batch_encoding['input_ids'][0].tolist()
    else:
        return batch_encoding


def tokenize_with_mask(sen1, sen2, tokenizer, only_ids=False):
    '''
    Use `tokenizer` to parse sentence `sen`. Replace keyword with appropriate
    number of `mask_token` tokens.
  
    We need to use `SequenceMatcher` because simply adding <mask> tokens is not
    realiable enough and weird empty tokens are being injected if the mask touches
    interpunction. E.g. XLM-R will tokenize `<mask>,` as: `['<mask>', '', ',']`
    Note the unexpected `''` token in the middle. Instead, we detect the tokens
    that stay the same in the original sentence using `Sequencematcher` and mask
    all the other tokens.
    
    `only_ids` - Return only token ids if True, `BatchEncoding` otherwise.
    '''
    batch_encoding = tokenize(sen1, tokenizer)
    sen1_tokens, sen2_tokens = tokenize(sen1, tokenizer, only_ids=True), tokenize(sen2, tokenizer, only_ids=True)
    
    for token_id, (sen1_token, sen2_token) in enumerate(zip(sen1_tokens, sen2_tokens)):
        if sen1_token != sen2_token:
            batch_encoding['input_ids'][0][token_id] = tokenizer.mask_token_id
   
    if only_ids:
        return batch_encoding['input_ids'][0].tolist()
    else:
        return batch_encoding