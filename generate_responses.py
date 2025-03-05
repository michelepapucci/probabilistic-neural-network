from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sae_lens import SAE
from tqdm import tqdm
import argparse
import torch
import json
import csv
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils

"""
Output structure of 'data/activations/generations_and_judgments_with_sae_{sae_target_layer}.json' the following:
{
    "idx": { # idx is an integer that indicates the index of the question in TruthfulQA
        "prompt": ... # Original Question from TruthfulQA
        "generations": [..., ..., ...] # A list of n_samples response from the model to the question
        "truth_label": ["yes", "yes", "no"] a list of n_samples truth judgments from the judge one for each of the sample.
    }
}

Output structure of "data/activations/output_{sae_target_layer}_activations.jsonl" is a jsonl where each line is a 
json object representing one of the original TruthfulQA question. For each of them the objes is as follows:
[ # A list of n_samples, one for each generation + 1, the last one is the one with temp = 0.0
    [ # A list for each generation step. Model has max_new_tokens = 50, so this have a maximum length of 50. The first step is the prompt.
        [ # A list of dictionary  each representing a token seen by the model at that step. 
          # Except for the first step where the model sees the prompt this list always has 1 token.
            {"sae_latent_id": ..., "sae_latent_id": ...} # A dictionary of non-zero activation of Sae Latent for that token    
        ],
        [],
        ...
    ],
    [],
    []
],
"""

device_gemma = "cuda:0"
device_judge = "cuda:1"
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False) # avoid blowing up mem
model_name = "google/gemma-2-2b"

def write_model_predictions(output_path, judge_predictions):
    with open(output_path, "w") as output_csv: 
        writer = csv.writer(output_csv)
        writer.writerow(["index", "prediction"])
        for prediction in judge_predictions:
            writer.writerow(prediction)

def SAE_on_layer_hook(value, hook):
    sae_acts = sae.encode(value.to(torch.float32))
    recon = sae.decode(sae_acts)

    if(len(sae_acts.shape) > 2):
        sae_acts = sae_acts.squeeze()
    if len(sae_acts.shape) == 1:
        sae_acts = sae_acts.unsqueeze(0) # uniforms to n_tokens, d_sae for all inputs (prompt included)
    run_activations = []
    for index, token in enumerate(sae_acts):
        indices = torch.nonzero(token, as_tuple=True)[0]
        values = token[indices]
        index_value_dict = dict(zip(indices.tolist(), values.tolist()))
        run_activations.append(index_value_dict)
    activations.append(run_activations)
    

def sample_responses_and_evaluate(dataset, model, truth_judge, tokenizer_judge, target_layer):
    global i, result, smpl_idx, activations
    result = {}
    output_jsonl = open(f"data/activations/output_{target_layer}_activations.jsonl", "w")
    output_jsonl.close()

    for i, example in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
        result[f"{i}"] = {
        "prompt": example['Question'], 
        "generations": [], 
        "truth_label": [], 
        "most_probable_answer": "",
        "most_probable_answer_label": ""
        }
        samples_activations = []
        with torch.no_grad():
            for smpl_idx in range(n_samples):
                activations = []
                outputs = model.generate(example['Question'], max_new_tokens=50)
                result[f"{i}"]['generations'].append(outputs)
                samples_activations.append(activations)
            result[f"{i}"]['most_probable_answer'] = model.generate(example['Question'], max_new_tokens=50, temperature=0.0)
        with open(f"data/activations/output_{target_layer}_activations.jsonl", "a") as output_sae:
            output_sae.write(json.dumps(samples_activations) + "\n")
            
        for outputs in result[f"{i}"]['generations']:
            prompt = f"Q: {example['Question']}?\nA: {outputs}\nTrue:"
            judge_inputs = tokenizer_judge.encode(prompt, return_tensors="pt").to(device_judge)
            outputs = truth_judge.generate(judge_inputs)
            pred_truth_label = tokenizer_judge.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
            result[f"{i}"]['truth_label'].append(pred_truth_label)
        
        prompt = f"Q: {example['Question']}?\nA: {result[f'{i}']['most_probable_answer']}\nTrue:"
        judge_inputs = tokenizer_judge.encode(prompt, return_tensors="pt").to(device_judge)
        outputs = truth_judge.generate(judge_inputs)
        pred_truth_label = tokenizer_judge.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        result[f"{i}"]['most_probable_answer_label'] = pred_truth_label
            
        with open(f"ongoing_{target_layer}.json", "w") as test:
            test.write(json.dumps(result))

    with open(f"data/activations/generations_and_judgments_with_sae_{target_layer}.json", "w") as output_json:
        output_json.write(json.dumps(result))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--sae_target_layer', type=int, required=True)
    parser.add_argument('-n', '--n_samples', type=int, default=5)
    parser.add_argument('-t', '--temperature', type=float, default=1.0)
    args = parser.parse_args()

    n_samples = args.n_samples
    temperature = args.temperature
    target_layer = args.sae_target_layer
    
    model = HookedTransformer.from_pretrained_no_processing(
        "google/gemma-2-2b",
        device_map=device_gemma,
        torch_dtype=torch.float16
    ).to(device_gemma)
    print(model)
    
    #tokenizer_gemma =  AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-2b-pt-res-canonical",
        sae_id = f"layer_{args.sae_target_layer}/width_16k/canonical",
    )
    sae.to(device_gemma)
    
    model.add_perma_hook(utils.get_act_name("resid_post", args.sae_target_layer), SAE_on_layer_hook)
    
    truth_judge = AutoModelForCausalLM.from_pretrained(
        "allenai/truthfulqa-truth-judge-llama2-7B", 
        torch_dtype=torch.float16
        ).to(device_judge)
    tokenizer_judge = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B")
    
    ds = load_dataset("domenicrosati/TruthfulQA")
    
    sample_responses_and_evaluate(ds, model, truth_judge, tokenizer_judge, args.sae_target_layer)