import os
import random
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from src.trainer import TrainingManager  # Assuming TrainingManager is part of your project
from src.utils import load_config  # Assuming this utility is available
import json
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

MIN_NUM_TOKENS = 5
class InferenceManager:
    def __init__(self, config_path, model,tokenizer, test_file_path, sample_fraction=0.005):
        # Load configuration
        self.config = load_config(config_path)

        # Setup paths and load model and tokenizer
        self.test_file_path = test_file_path
        self.sample_fraction = sample_fraction
        
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_test_data(self):
        """Load and sample the test dataset."""
        with open(self.test_file_path, "r", encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line.split()) > MIN_NUM_TOKENS and not line.isspace())
            ]
    
        # Extract "text" field and filter based on token count
        valid_lines = []
        for line in lines:
            try:
                data = json.loads(line)  # Parse the line as JSON
                if data.get("n_words", 0) > MIN_NUM_TOKENS:  # Check "n_words"
                    valid_lines.append(data["text"])  # Access the "text" field
            except (json.JSONDecodeError, KeyError):
                # Skip lines that cannot be parsed or are missing required keys
                continue
    
        # Sample the valid lines
        sample_size = max(1, int(len(valid_lines) * self.sample_fraction))
        sampled_sentences = random.sample(valid_lines, sample_size)
        return sampled_sentences



    def mask_tokens(self, sentence, mask_prob=0.1):
        """
        Randomly mask tokens in a sentence based on the tokenizer.
        """
        tokens = self.tokenizer.tokenize(sentence)
        original_tokens = {}
    
        for i, token in enumerate(tokens):
            # Mask only non-special tokens
            if not token.startswith(self.tokenizer.special_tokens_map.get("cls_token", "[CLS]")) and \
               not token.startswith(self.tokenizer.special_tokens_map.get("sep_token", "[SEP]")) and \
               random.random() < mask_prob:
                original_tokens[i] = token
                tokens[i] = self.tokenizer.mask_token or "[MASK]"
    
        masked_sentence = self.tokenizer.convert_tokens_to_string(tokens)
        return masked_sentence, original_tokens



    def evaluate(self, sentences, mask_prob=0.15, results_file="evaluation_results.json"):
        """Evaluate the model on masked sentences using BLEU and ROUGE metrics."""
        results = []
        all_references = []  # For BLEU: list of lists of reference strings
        all_hypotheses = []  # For BLEU: list of hypothesis strings
        
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
        batch_size = 16  # Adjust based on available GPU memory
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            masked_sentences = []
            original_tokens_list = []
    
            # Mask tokens in the sentences
            for sentence in batch_sentences:
                masked_sentence, original_tokens = self.mask_tokens(sentence, mask_prob)
                if original_tokens:
                    masked_sentences.append(masked_sentence)
                    original_tokens_list.append(original_tokens)
    
            if not masked_sentences:
                continue
    
            # Tokenize and pass to the model
            inputs = self.tokenizer(masked_sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
            logits = outputs.logits
    
            # Locate masked token indices
            mask_token_indices = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
    
            for idx, original_tokens in enumerate(original_tokens_list):
                correct = 0
                predicted_tokens = []
    
                # Generate predictions for the entire sequence
                for token_idx in range(logits.size(1)):
                    token_logits = logits[idx, token_idx, :]
                    predicted_token_id = token_logits.argmax(dim=-1).item()
                    predicted_token = self.tokenizer.decode([predicted_token_id]).strip()
                    predicted_tokens.append(predicted_token)
    
                # Check correctness only for masked tokens
                for token_idx in mask_token_indices[1][mask_token_indices[0] == idx]:
                    token_idx = token_idx.item()
                    if token_idx in original_tokens:
                        original_token = original_tokens[token_idx]
                        if predicted_tokens[token_idx] == original_token:
                            correct += 1
    
                # Calculate accuracy for masked tokens
                accuracy = correct / len(original_tokens) if original_tokens else 0
                original_sentence = batch_sentences[idx]
                masked_sentence = masked_sentences[idx]
                predicted_sentence = self.tokenizer.convert_tokens_to_string(predicted_tokens)
    
                results.append({
                    "original_sentence": original_sentence,
                    "masked_sentence": masked_sentence,
                    "predicted_sentence": predicted_sentence,
                    "accuracy": accuracy
                })
    
                # For BLEU, add references and hypothesis as strings
                all_references.append([original_sentence])  # List of reference strings for each hypothesis
                all_hypotheses.append(predicted_sentence)   # Single hypothesis string
    
                # Compute ROUGE scores
                rouge_score = rouge_scorer_instance.score(original_sentence, predicted_sentence)
                rouge_scores["rouge1"].append(rouge_score["rouge1"].fmeasure)
                rouge_scores["rouge2"].append(rouge_score["rouge2"].fmeasure)
                rouge_scores["rougeL"].append(rouge_score["rougeL"].fmeasure)
    
        # Debug prints
        if all_hypotheses:
            print("Sample hypothesis:", all_hypotheses[0])
            print("Sample reference:", all_references[0])
    
        # Compute BLEU score
        bleu_score = corpus_bleu(all_hypotheses, all_references).score
        print(f"BLEU Score: {bleu_score}")
    
        # Compute average ROUGE scores
        avg_rouge_scores = {key: sum(values) / len(values) for key, values in rouge_scores.items()}
        print(f"ROUGE Scores: {avg_rouge_scores}")
    
        # Save results to a file
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({"results": results, "bleu_score": bleu_score, "rouge_scores": avg_rouge_scores}, f, indent=4)
    
        return results, bleu_score, avg_rouge_scores
            
        
        
        
    
    
