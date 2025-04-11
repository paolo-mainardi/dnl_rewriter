import argparse
import os
import torch
import pandas as pd
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from transformers.modeling_outputs import BaseModelOutput

def balance_neogate(neogate_df:pd.DataFrame):
    """
    Each entry in Neo-GATE has one masculine and one feminine translation. 
    We only want one gendered sentence for each row, but we want the final result to be balanced by gender.
    This function returns a version of Neo-GATE with one gendered sentence for each row, where half the sentences are masculine and half are feminine.

    Args:
        neogate_df (pandas.DataFrame): Neo-GATE in pandas format. 

    Returns:
        pandas.DataFrame: A new balanced version of the dataset in pandas format. 
    """
    drop_m = "REF-F"
    drop_f = "REF-M"

    splitpoint = len(neogate_df) // 2

    m = neogate_df[:splitpoint].drop(columns=drop_m).rename(columns={"REF-M": "REF"})
    f = neogate_df[splitpoint:].drop(columns=drop_f).rename(columns={"REF-F": "REF"})

    m["GENDER"] = "M"
    f["GENDER"] = "F"

    result = pd.concat([m, f])
    
    assert len(result) == len(neogate_df), \
        f"Warning! When calling balance_neogate(), the original dataset has {len(neogate_df)} rows, while the resulting one has {len(result)}"

    return result

def collect_preprocess_data(train_data, val_data):
    train = Dataset.from_pandas(pd.read_csv("./dataset/train.tsv", sep="\t"))
    val = Dataset.from_pandas(pd.read_csv("./dataset/val.tsv", sep="\t"))

    # Add my version of Neo-GATE dev split to training data
    adapted_ref_dev = [l.strip() for l in open("./data/adapted_neogate/adapted_neogate_dev.ref", encoding="utf-8").readlines()] # Neo-GATE references adapted to my paradigm
    neogate_dev_schwa = load_dataset("FBK-MT/Neo-GATE")["dev"].add_column(name="SCHWA", column=adapted_ref_dev).to_pandas()
    neogate_dev_balanced = Dataset.from_pandas(balance_neogate(neogate_dev_schwa))

    # Concatenate training set
    train = concatenate_datasets([train, val, neogate_dev_balanced]) # Shuffling happens when prompts are created

    # Load test set
    neogate = load_dataset("FBK-MT/Neo-GATE")["test"].to_pandas()
    test = Dataset.from_pandas(balance_neogate(neogate)) # Gender-balance this split too

    return train, test

def load_model(model_path, access_token, quantization, workflow):
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}", token=access_token)

    model_args = {
        "pretrained_model_name_or_path": f"{model_path}",
        "token": f"{access_token}"
    }

    # Set quantization parameters
    if quantization is not None:
        if quantization == 4:
            q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
            )
        
        elif quantization == 8:
            q_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        model_args["quantization_config"] = q_config
        model_args["device_map"] = "auto"

    # Select appropriate AutoModel class based on model type
    if workflow == "llm":
        model_class = AutoModelForCausalLM
    elif workflow == "seq2seq":
        model_class = AutoModelForSeq2SeqLM

    model = model_class.from_pretrained(**model_args)

    print("Note: Model takes up {} bytes of memory".format(model.get_memory_footprint()))

    return tokenizer, model

def create_prompts(language, workflow, train_sources, train_targets, test, num_examples):
    """
    This function returns target inputs (requests) and batches of k (num_examples) examples separately. It does NOT add a description of the task: to get that, call add_instructions() on the examples returned by this function. 

    Args:
        train_sources (list): The list of reference sentences (inputs) used to create example pairs.
        train_targets (list): The list of target sentences (labels) used to create example pairs.
        test (list): The list of test (input) sentences used to send requests.
        num_examples (int): The number of examples (shots) for each request. 
    
    Returns:
        example_batches (list): A list containing a batche of k example pairs for each request.
        target_inputs (list): A list containing target inputs, i.e., requests to send to the model. 
    """
    # Create open-ended target input (request)
    target_inputs = []
    for sent in test:
        sent = sent.strip()
        if language == "en":
            target_input = f"Original sentence: <{sent}> Rewritten sentence:"
        elif language == "it":
            target_input = f"Frase originale: <{sent}> Riformulazione:"
        if workflow == "seq2seq":
            target_input = f"{target_input} <extra_id_0>" # Add sentinel token for seq2seq models trained on a denoising objective: see Raffel et al. (2020), Lee et al. (2024).
        target_inputs.append(target_input)

    # Create example set
    train_pairs = list(zip(train_sources, train_targets)) # Examples are made up of a reference traslation + its schwa reformulation

    train_examples = [] # Create example pairs by concatenating input and target
    i=0
    for pair in train_pairs:
        src = pair[0].strip()
        tgt = pair[1].strip()
        if language == "en":
            example_pair = f"Original sentence: <{src}> Rewritten sentence: {tgt}</s>"
        elif language == "it":
            example_pair = f"Frase originale: <{src}> Riformulazione: {tgt}</s>"
        train_examples.append(example_pair)
        i+=1

    # Create n batches of example pairs, where n is the number of requests (sentences in the test set)
    example_batches = []
    for i in target_inputs: # A batch is a set of k example pairs
        batch = random.sample(train_examples, num_examples)
        example_set = "\n".join(batch) # Add a newline between each example pair and the next
        example_batches.append(example_set)

    return example_batches, target_inputs

def create_messages(train_sources, train_targets, test, num_examples) -> list:
    """
    This function returns a list of chat messages containing:
    - instructions and input examples from the user
    - sample assistant outputs
    - a request
    This is meant for "chat" models, whose configuration has a chat_template that the tokenizer can use (this is required).

    Args:
        train_sources (list): The list of reference sentences (inputs) used in the user input.
        train_targets (list): The list of target sentences (labels) used in the sample assistant output.
        test (list): The list of test (input) sentences used to send requests.
        num_examples (int): The number of examples (shots) for each request.
    
    Returns:
        messages (list): A list of chat messages.
    """
    messages = []

    # Create open-ended target input (request) with sentinel token
    target_inputs = []
    for sent in test:
        target_inputs.append(f"Original sentence: <{sent.strip()}> Rewritten sentence: ")

    if num_examples == 0:
        # Create 0-shot prompts
        for i, inpt in enumerate(target_inputs):
            message = [
                {
                    "role": "user",
                    "content": f"Rewrite the following Italian sentence by replacing masculine and feminine endings with a schwa (ə) for human entities.\n{inpt}"
                }
            ]

            messages.append(message)

    else:
        # Create example set
        train = list(zip(train_sources, train_targets)) # Examples are made up of reference traslation + schwa reformulation

        example_pairs = []
        for pair in train: # Create templates for the example pairs
            src = pair[0].strip()
            tgt = pair[1].strip()
            example_source = f"Original sentence: <{src}> Rewritten sentence: "
            example_target = f"<{tgt}>"

            example_pairs.append((example_source, example_target))

        # Create complete messages with batces of k examples, where k = num_examples 
        for i in range(len(target_inputs)):
            example_batch = random.sample(example_pairs, k=num_examples)
            src_batch = []
            tgt_batch = []
            for pair in example_batch:
                src_batch.append(pair[0].strip())
                tgt_batch.append(pair[1].strip())
            target_input = target_inputs[i]
            message = [
                {
                    "role": "user",
                    "content": f"Rewrite the following Italian sentence by replacing masculine and feminine endings with a schwa (ə) for human entities.\n" + "\n".join(src_batch)
                },
                {
                    "role": "assistant",
                    "content": "\n".join(tgt_batch)
                },
                {
                    "role": "user",
                    "content": f"{target_input}"
                }
            ]
            messages.append(message)
    
    return messages

def add_instructions(language, examples:list):
    """
    This function enriches prompts for instruction-tuned models by opening the prompt with a description of the task. This is the default for chat models, so this function can only be called on a dedicated list of examples, not on complete prompts. This is thus meant for standard (non-chat) decoder-only LLMs mainly. It can also be used for prompting encoder-decoder (seq2seq) models, in which case the instructions will be passed to the encoder together with the examples and the request.
    """
    instructions = []
    for batch in examples:
        if language == "en":
            instr = f"Rewrite the following Italian sentence by replacing masculine and feminine endings with a schwa (ə) for human entities.\n{batch}"
        elif language == "it":
            instr = f"Riscrivi la seguente frase italiana utilizzando uno schwa (ə) al posto delle desinenze maschili e femminili per i referenti umani.\n{batch}"
        instructions.append(instr)

    return instructions

def get_encoder_outputs(tokenizer, model, example_set:str, target_input:str):
    """
    This function is used when prompting encoder-decoder (seq2seq) models with a fusion-based approach, where each example is processed separately by the encoder: see Lee et al. (2024). 

    Args:
        example_set (str): A batch of k examples (shots) separated by newline (\n).
        target_input (str): A request.

    Returns:
        encoder outputs: A tuple containing the concatenated encoder last hidden states and attention masks (one of each per example). 
    """
    encoder_outputs = []
    attention_masks = []
    for example in example_set.split("\n"): # Encode each example together with the target input
        encoder_inputs = tokenizer(
            f"{example}\n{target_input}",
            return_tensors = "pt"
        ).to(model.device)

        attention_masks.append(encoder_inputs.attention_mask)

        lhs = model.encoder(**encoder_inputs).last_hidden_state

        encoder_outputs.append(lhs)
    
    concat_lhs = BaseModelOutput(last_hidden_state=torch.cat(encoder_outputs, dim=1)) # Concatenate all hidden states and wrap as BaseModelOutput
    concat_am = torch.cat((attention_masks), dim=1)

    return concat_lhs, concat_am

def kshot_prompt(workflow, tokenizer, model, example_batches:list, target_inputs:list) -> list:
    """
    This function sends requests and collects outputs (predictions). It can work with both decoder-only and encoder-decoder models, but it is only suitable for k-shot prompting. Use no_examples() for 0-shot prompting. In the case of encoder-decoder models, this function adopts the "early fusion" approach described in Lee et al. (2024). 
    The arguments for this function can be obtained by calling create_prompts(). 

    Args:
        example_batches(list): A list containing batches of k newline-separated examples. There has to be one batch of examples for each request to send to the model. 
        target_inputs(list): A list containing requests to send to the model. A request contains an input sentence and it gives the model the prompt to generate its output. 
    
    Returns:
        predictions (list): A list of model outputs (one per request). 
    """
    predictions = []

    zipped = list(zip(example_batches, target_inputs)) # Create tuples containing a set of examples and a request
    for i, item in enumerate(zipped):
        example_set = item[0]  # Examples
        target_input = item[1] # Request

        if workflow == "seq2seq":
            encoder_hidden_states, encoder_attention_masks = get_encoder_outputs(tokenizer, model, example_set, target_input) # Get encoder hidden states for examples and request
            
            input_len = tokenizer(target_input, return_tensors="pt").input_ids.shape[1]

            decoder_inputs = tokenizer(
                "<extra_id_0>",
                return_tensors="pt"
            ).to(model.device).input_ids # Use the sentinel token to prompt the decoder to generate the rewritten sentence

            output = model.generate(
                tokenizer = tokenizer,
                inputs = decoder_inputs,
                attention_mask = encoder_attention_masks,
                encoder_outputs = encoder_hidden_states, # Use concatenated encoder last hidden states for decoder cross-attention
                max_new_tokens = input_len,
                stop_strings = ["."]
                )
        
        elif workflow == "llm":
            prompt = f"{example_set}\n{target_input}" # Concatenate examples and request with a newline in between
            
            prompt_input_ids = tokenizer(
                prompt,
                return_tensors="pt"
            ).to(model.device).input_ids

            prompt_len = prompt_input_ids.shape[1]

            output = model.generate(
                inputs = prompt_input_ids,
                max_new_tokens = prompt_len
            )

        prediction = tokenizer.decode(output[0], skip_special_tokens=True) # Decode the logits
            
        predictions.append(prediction.split("\n")[-1]) # Try to only collect final answer if the model outputs the whole prompt
        print(f"\r{i+1}/{len(zipped)} predictions collected", end="", flush=True)
    print()
    
    return predictions

def complete_chat(tokenizer, model, messages:list) -> list:
    """
    This function is used to send requests to chat models. The arguments for this function can be obtained by calling create_messages().
    It is only suitable for k-shot prompting; use no_examples() for 0-shot prompting.     

    Args:
        messages (list): The list of prompts containing a set of k examples and a request. 

    Returns:
        predictions (list): A list of model outputs (one per request). 
    """
    predictions = []

    for i, message in enumerate(messages): # Tokenize by applying model-specific chat template
        message_tokenized = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        message_len = message_tokenized.input_ids.shape[1]

        outputs = model.generate(
            inputs=message_tokenized.input_ids,
            attention_mask=message_tokenized.attention_mask,
            max_new_tokens=message_len,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction) # NOTE: The collected prediction will need post-processing
        print(f"\r{i+1}/{len(messages)} predictions collected", end="", flush=True)
    print()

    return predictions

def zeroshot_prompt(tokenizer, model, language, test):
    """
    This function is used for 0-shot prompting; it compiles requests and adds a task description to each request sent. It is only meant for standard (non-chat) decoder-only models when no examples (training data) are provided. It should not be called in combination with add_instructions().

    Args:
        inputs (list): A list of requests to send to the model. This can be obtained by calling create_prompts().
    
    Returns:
        predictions (list): A list of model outputs (one per request).
    """
    target_inputs = []
    for sent in test:
        if language == "en":
            target_inputs.append(f"Original sentence: <{sent.strip()}> Rewritten sentence:")
        elif language == "it":
            target_inputs.append(f"Frase originale: <{sent.strip()}> Riformulazione:")

    predictions = []
    
    for i, inpt in enumerate(target_inputs):
        prompt = "Rewrite the following Italian sentence by replacing masculine and feminine endings with a schwa (ə) for human entities.\n" + inpt

        input_ids = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device).input_ids
    
        prompt_len = input_ids.shape[1]

        outputs = model.generate(
            inputs = input_ids,
            max_new_tokens = prompt_len
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction.split("\n")[-1]) # Try to only collect final answer
        print(f"\r{i+1}/{len(target_inputs)} predictions collected", end="", flush=True)
    print()

    return predictions

def write_predictions(save_model_name, logging_dir, preds:list) -> None:
    """
    This function is used to write model outputs to a file, where each output will be on one line. No post-processing is applied at this stage. 

    Args:
        preds (list): The list of model predictions obtained by calling one of get_predictions(), complete_chat(), or no_examples(). 
    """
    i = 1
    with open(f"{logging_dir}/{save_model_name}_predictions.txt", "a+", encoding="utf-8") as wf:
        for pred in preds:
            wf.write(f"{pred}\n")
            print(f"\r{i}/{len(preds)} model outputs written", end="", flush=True)
            i+=1
        print()
        print(f"Find outputs at {logging_dir}/{save_model_name}_predictions.txt")
    
    return None

def main(args):
    # Get model name
    model_name = args.model_path.split("/")[1]

    # Set language
    if args.lang == None:
        language = "it" if "it" in model_name else "en"
    else:
        language = args.lang

    # Set file name for logging
    os.makedirs(f"{args.logging_dir}", exist_ok=True)
    
    save_model_name = f"{model_name}_{language}"

    if args.quantization is not None:
        if args.quantization == 4:
            save_model_name = f"{save_model_name}_4bit"
        elif args.quantization == 8:
            save_model_name = f"{save_model_name}_8bit"
    else:
        save_model_name = f"{save_model_name}_full"
    
    save_model_name = f"{save_model_name}_{args.num_examples}shot"

    # Load data
    train, test = collect_preprocess_data(args.train_data, args.val_data)

    # Load tokenizer and model
    tokenizer, model = load_model(f"{args.model_path}", args.hf_token, args.quantization, args.workflow)

    # Extract inputs and labels
    inputs = train["REF"]
    labels = train["SCHWA"]

    # Shorten each individual input/label that will be used to create example if necessary to save memory
    # This way, only examples will be shortened, leaving the prompt template intact
    if args.example_maxlen != None:
        inputs = [" ".join(inpt.split(" ")[:args.example_maxlen]) for inpt in inputs]
        labels = [" ".join(labl.split(" ")[:args.example_maxlen]) for labl in labels]

    if args.chat == True:
        messages = create_messages(train["REF"], train["SCHWA"], test["REF"], args.num_examples)
        predictions = complete_chat(tokenizer, model, messages)
    else:
        if args.num_examples == 0:
            predictions = zeroshot_prompt(tokenizer, model, language, test["REF"])
        else:
            examples, target_inputs = create_prompts(language, args.workflow, train["REF"], train["SCHWA"], test["REF"], args.num_examples)
            if args.instructions == True:
                examples = add_instructions(language, examples)
                save_model_name = f"{save_model_name}_instructions"
            predictions = kshot_prompt(args.workflow, tokenizer, model, examples, target_inputs)
    
    # Write model outputs to file
    write_predictions(save_model_name, args.logging_dir, predictions)
    print()

    return None

def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        help="The path to the Hugging Face model to prompt, formatted as 'author/model'"
    )
    parser.add_argument(
        "framework",
        choices=["llm", "seq2seq"],
        help="The type of model to train; either 'llm' or 'seq2seq'"
    )
    parser.add_argument(
        "--train_path", 
        default="./dataset/train.tsv",
        help="The path to the training data"
    )
    parser.add_argument(
        "--val_path", 
        default="./dataset/val.tsv",
        help="The path to the validation data"
    )
    parser.add_argument(
        "--logging_dir",
        default="./results/prompting", 
        help="The path to the folder to use for logging"
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token to access gated repositories, if applicable. Defaults to None")
    parser.add_argument(
        "--lang",
        default=None,
        help="The language to use for prompts. Defaults to 'it' if 'it' is in model name, 'en' otherwise"
    )
    parser.add_argument(
        "--quantization",
        type=int,
        default=None,
        help="Specify the precision to quantize the model before calling it, if necessary. Either 4 or 8 (for 4-bit or 8-bit precision)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="The number of examples to include in each prompt. If set to 0, switches to 0-zhot prompting. Defaults to 2"
    )
    parser.add_argument(
        "--example_maxlen",
        type=int,
        default=None,
        help=""
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        default=False,
        help="Add this flag to add task instructions at the beginning of each prompt. Defaults to False"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=False,
        help="Add this flag if calling a chat model"
    )

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli()