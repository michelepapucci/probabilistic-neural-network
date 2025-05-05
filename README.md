# Probabilistic Neural Network Exam
The scripts are an adaptation of [Uncertainty in LLMs](https://github.com/DhairyaKarna/uncertainity_in_LLMs/) and adapted to [TruthfulQA Dataset](https://huggingface.co/datasets/domenicrosati/TruthfulQA).

# Models

The experiments are done on [Gemma 2](https://huggingface.co/google/gemma-2-2b) on its 2B parameters version. 
Other model that have been used for various purposes are:
- [TruthfulQA Truth Judge](https://huggingface.co/allenai/truthfulqa-truth-judge-llama2-7B) is a 7b Llama model trained on analyzing the responses to the TruthfulQA questions and giving a "truth" score (binary 'yes' or 'no'). 
- [DeBerta MNLI](https://huggingface.co/microsoft/deberta-large-mnli) is a DeBerta model trained on the [Multi-Genre Natural Language Inference](https://cims.nyu.edu/~sbowman/multinli/) task, giving a binary classification of wheter the first part of the sequence, before the \[SEP\] entails the last part of the sequence. 
- [Sentence Transformers](https://sbert.net/) are a collection of encoder model trained for calculating similarity scores between sentences. I've used [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) pre-trained sentence transformer model as an alternative of DeBerta MNLI for calculating answer distances.
- [Gemma Scope Res SAE](https://huggingface.co/google/gemma-scope-2b-pt-res) is a Sparse Autoencoder pre-trained on Gemma 2 residual stream activation. It's a Multi-layer perceptron with a hidden size 10x the hidden size of Gemma, uses a sparsity constraint in the loss and a modified activation function called JumpReLU. It's trained in reconstructing the input after projecting it into this sparse and larger latent space. It's used to disentangle the super-positioned feature inside the dense latent space of the Transformer. 

# Code

## Generating Samples

`generate_responses.py` takes an argument `-l` which represents the layer of Gemma at which the Gemma Scope SAE is hooked. The script loops trough TruthfulQA and generates 5 answers per-question at temperature 1.0 and one at temperature 0.0. Each of these answers are judged by the TruthfulQA Truth Judge, and for each of these generations the SAE latent space activation are saved for further analysis. 

Then, `clean_responses.py` is launched to clean the generations from the prompt if the model repeated it in its answer.


## Generating Similarity Scores

`generate_similarities_score.ipynb` is a Jupyter notebook that, given the previously generated answers, calculates a bunch of similarity scores. 
In particular it calculates:
- Rouge Scores (rouge1, rouge2, rougeL) as a form of *syntactic similarity*;
- Semantic Sets trough DeBerta MNLI predictions. For each possible couple $(a, b)$ in the generation set of each questions (aka 10 couples), we predict wheter $a$ entails $b$ or $b$ entails $a$. If neither is the case, the two answers are placed in different "semantic sets", otherwise they join the same set. The hypothesys is that if a model is more certain about its answers they should belong to the same semantic set. 

Thes Similarity scores are then used to calculate both **uncertainty** estimate $U(x)$ or a confidence score $C(x)$. 

### Sentence Transformer Variant
`generate_similarities_scores_sentence_transformer.ipynb` is a variant of `generate_similarities_scores.ipynb` where instead of using DeBerta trained on MNLI, it uses a pre-trained Sentence Transformer which is a model specifically tailored to calculate similarity between sentence. All experiments have been done on both variant of similarity scores. 

## Uncertainty vs Confidence
Uncertainty $U$ typically depends only on the input $x$. So, e.g., $P(Y|x) = \mathcal{N}(\mu, \sigma^2)$ the variance $\sigma^2$ is an uncertainty measure. 
However Confidence measure $C$ is generally associated with both the input and the prediction $C(x,y)$, in the context of classification, one of the simplest confidence measures is just the predicted probabilty. 

## Generating an Uncertainty Estimate

`generate_uncertainty.ipynb` is a Jupyter notebook that given the generated answers calculates a measure of *total uncertainty* which should be indicative of epistemic uncertainty. 
The way it works is by creating a prompt as: *Question: {} \n Here are some ideas that were brainstormed:{}\n Possible answer:{}\n Is the possible answer:\n (A) True\n (B) False\n The possible answer is: True*. 
This is done in a few shot settings, before this is appended a couples of examples with the correct prediction. Then, this is all masked, with the exception of the "True" token, so that we can see the loss of the model w.r.t this injected prediction (i.e. we always inject True, wheter or not the answer is true). 
This loss is calculated for a good sample of the dataset 800/816 (the others are used as few-shot examples), and then transformed to *probability estimates* of the model uncertainty towards that prediction by the means of $exp(-loss)$. 
These values are used, togheter with the true labels for calculating an AUROC score as a total uncertainty measure of the model prediction of the "True" label. 

## Generating Confidence Measures

`generate_log_likelihoods.ipynb` is a Jupyter notebook used to do a bunch of calculation before actually generating confidence measures with `generate_confidence.ipynb`. 

In particular, `generate_log_likelihoods.ipynb` calculates:
- Average Negative Log Likelihood (NLI) calculated for each generations. Basically it calculates the loss for any of the answers by doing a forward pass trough the model (ignoring the prompt for the purpose of calculating the loss). This gives a measure of how likely is the model to generate that sentence given the prompt. 
- Average Unconditioned NLL is the same as above, but without providing the prompt. Aka how likely is the model to generate that sentence without the context of the prompt. 
- Pointwise Mutual Information (PMI) between the NLL and the Uncoditioned NLL. It gives a measure of how much more informative the prompt is to generate the sequence compared to generating it without it.
- Hidden state of the last layer are also saved. 

This calculations are done for all the generations. `generate_confidence.ipynb` calculates a bunch more measure starting from these ones, in particular: 
- Average PMI;
- Mutual Information;
- Variance of NLL;
- Mean of NLL;
- Predictive Entropy;
- Predictive Entropy across Concepts; 
- Margin Probability Uncertainty; 

This is calculated on the whole dataset and also on different subsets. Then, all the calculated measures are aggregated and reported on a json file using `result.ipynb` and `result_SAE.ipynb`. 

# Results and Comments

I divided the results into three categories: the uncertainty measure results, the confidence measures results, and the SAE activations analysis. 

## Uncertainty Measure
uncertainty

## Confidence Measure
confidence

## SAE activations analysis
SAE