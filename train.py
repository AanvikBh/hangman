import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import json
import argparse
import re
import collections

from utils import (fetch_word_list, generate_token_maps, persist_checkpoint)
from data_loader import HangmanDataset, batching_processor
from model import HangmanGuesserNetwork

# --- English Language Semantics ---
ENGLISH_LETTER_FREQ = {'e': 12.0, 't': 9.1, 'a': 8.1, 'o': 7.7, 'i': 7.3, 'n': 6.9, 's': 6.3, 'r': 6.0, 'h': 5.9, 'd': 4.3, 'l': 4.0, 'u': 2.9, 'c': 2.7, 'm': 2.6, 'f': 2.3, 'y': 2.1, 'w': 2.1, 'g': 2.0, 'p': 1.8, 'b': 1.5, 'v': 1.1, 'k': 0.7, 'x': 0.2, 'q': 0.1, 'j': 0.1, 'z': 0.1}

def model_guess(network, token_map, current_word, guessed_letters, device):
    """Generates a guess using the trained neural network."""
    network.eval()
    int_to_token = {i: c for c, i in token_map.items()}
    
    clean_word_list = [char for char in current_word if char != ' ']
    
    input_ints = [token_map.get(c, 0) for c in clean_word_list]
    input_seq = torch.tensor([input_ints], dtype=torch.long).to(device)
    seq_dim = torch.tensor([len(clean_word_list)], dtype=torch.long)
    
    with torch.no_grad():
        prediction_logits = network(input_seq, seq_dim)[0]

    unknown_indices = [i for i, c in enumerate(clean_word_list) if c == '_']
    if not unknown_indices:
        return None # No letters to guess

    aggregated_scores = torch.sum(prediction_logits[unknown_indices], dim=0)
    
    # Simple semantic boost for 'q' -> 'u'
    if 'q' in clean_word_list and 'u' not in guessed_letters:
        q_pos = clean_word_list.index('q')
        if q_pos + 1 < len(clean_word_list) and clean_word_list[q_pos+1] == '_':
             u_index = token_map.get('u', -1)
             if u_index != -1:
                aggregated_scores[u_index] += 5.0
    
    sorted_scores, sorted_indices = torch.sort(aggregated_scores, descending=True)

    for _, char_idx in zip(sorted_scores, sorted_indices):
        char = int_to_token.get(char_idx.item())
        if char and char.isalpha() and char not in guessed_letters:
            return char
            
    return None # Should not happen if there are still letters to guess

def dictionary_guess(current_word, guessed_letters, current_dictionary, full_freq_sorted):
    """Fallback guess method using dictionary frequency."""
    clean_word = current_word.replace(" ", "")
    regex_word = clean_word.replace("_", ".")
    len_word = len(clean_word)

    new_dictionary = [
        dict_word for dict_word in current_dictionary
        if len(dict_word) == len_word and re.fullmatch(regex_word, dict_word)
    ]
    
    dict_string = "".join(new_dictionary)
    if dict_string:
        counts = collections.Counter(dict_string)
        sorted_letter_count = counts.most_common()
    else: # Fallback to full dictionary frequency
        sorted_letter_count = full_freq_sorted
        
    for letter, _ in sorted_letter_count:
        if letter not in guessed_letters:
            return letter, new_dictionary
            
    return None, new_dictionary # Should not happen

def run_game_simulation(network, secret_term, token_to_int, compute_device, full_dictionary, full_freq_sorted, max_mistakes=6):
    """Simulates a game using a hybrid model and dictionary approach."""
    network.eval()
    
    current_display_list = ['_'] * len(secret_term)
    attempted_chars = set()
    mistakes = 0
    current_dictionary = full_dictionary

    while '_' in current_display_list and mistakes < max_mistakes:
        current_word_str = " ".join(current_display_list)
        
        # Primary guess from the model
        guess = model_guess(network, token_to_int, current_word_str, attempted_chars, compute_device)
        
        # Fallback to dictionary if model fails or gives a repeated letter
        if guess is None or guess in attempted_chars:
            guess, current_dictionary = dictionary_guess(current_word_str, attempted_chars, current_dictionary, full_freq_sorted)
            if guess is None: break # No valid guesses left

        attempted_chars.add(guess)

        if guess in secret_term:
            for i, letter in enumerate(secret_term):
                if letter == guess:
                    current_display_list[i] = letter
        else:
            mistakes += 1
            
    is_success = '_' not in current_display_list
    return {'term': secret_term, 'is_win': is_success, 'errors': mistakes}


def assess_network_performance(network, vocabulary, token_map, exec_device, full_dict, full_freq, game_count=1000, assessment_name="Test"):
    """Runs multiple game simulations to assess model performance."""
    print(f"\n--- Initiating {assessment_name} ---")
    print(f"Running assessment over {game_count} sample games...")
    test_terms = random.sample(vocabulary, min(game_count, len(vocabulary)))
    successes = 0
    total_errors = 0

    for term in tqdm(test_terms, desc=f"Simulating {assessment_name} Games"):
        outcome = run_game_simulation(network, term, token_map, exec_device, full_dict, full_freq)
        if outcome['is_win']:
            successes += 1
        total_errors += outcome['errors']

    success_ratio = successes / len(test_terms)
    avg_errors = total_errors / len(test_terms)
    
    print(f"--- {assessment_name} Complete ---")
    print(f"Success Ratio: {success_ratio:.2%}")
    print(f"Average Errors per Game: {avg_errors:.2f}")
    return success_ratio, avg_errors

def execute_training_cycle(network, data_iterator, loss_calculator, adam_opt, hardware):
    """Runs a single cycle of training over the dataset."""
    network.train()
    cumulative_loss = 0
    for input_data, target_data, seq_dims in tqdm(data_iterator, desc="Training Cycle"):
        input_data, target_data = input_data.to(hardware), target_data.to(hardware)
        
        adam_opt.zero_grad()
        predictions = network(input_data, seq_dims)
        
        batch_loss = loss_calculator(predictions.view(-1, predictions.size(-1)), target_data.view(-1))
        
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        adam_opt.step()
        cumulative_loss += batch_loss.item()
        
    return cumulative_loss / len(data_iterator)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def main(args):
    config = load_config(args.config_path)
    paths_cfg = config['paths']
    model_cfg = config['model_params']
    train_cfg = config['training_params']
    eval_cfg = config['evaluation_params']

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {compute_device}")

    vocabulary = fetch_word_list(paths_cfg['word_lexicon'])
    full_freq_sorted = collections.Counter("".join(vocabulary)).most_common()
    token_to_int, _ = generate_token_maps(vocabulary)
    char_count = len(token_to_int)
    
    # --- Modified Data Split: 90% train, 10% test ---
    random.shuffle(vocabulary)
    train_test_split = int(0.9 * len(vocabulary))
    
    training_set = vocabulary[:train_test_split]
    testing_set = vocabulary[train_test_split:]
    print(f"Data split: {len(training_set)} train, {len(testing_set)} test words.")

    training_data = HangmanDataset(training_set, token_to_int)
    training_loader = DataLoader(training_data, batch_size=train_cfg['chunk_size'], shuffle=True, collate_fn=batching_processor)

    guesser_network = HangmanGuesserNetwork(
        vocab_size=char_count, 
        hidden_dim=model_cfg['recurrent_units'], 
        vector_dim=model_cfg['vector_dimension'],
        layer_count=model_cfg['recurrent_layers'],
        dropout_prob=model_cfg['dropout_rate']
    ).to(compute_device)
    
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    adam_optimizer = optim.Adam(guesser_network.parameters(), lr=train_cfg['initial_learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=5, gamma=0.1)

    execution_log = []
    
    print("\nBeginning network training...")
    for cycle_num in range(train_cfg['training_cycles']):
        print(f"\n--- Epoch {cycle_num + 1}/{train_cfg['training_cycles']} ---")
        
        avg_cycle_loss = execute_training_cycle(guesser_network, training_loader, loss_function, adam_optimizer, compute_device)
        print(f"Mean Training Loss: {avg_cycle_loss:.4f}")
        
        # Prevent LR from decaying too low
        if scheduler.get_last_lr()[0] > train_cfg['min_learning_rate']:
             scheduler.step()
        print(f"Current learning rate: {adam_optimizer.param_groups[0]['lr']}")
        
        run_summary = {'cycle': cycle_num + 1, 'mean_loss': avg_cycle_loss}
        execution_log.append(run_summary)
    
    print("\n--- Training Complete ---")
    print("Saving final model checkpoint...")
    persist_checkpoint(guesser_network, adam_optimizer, execution_log, token_to_int, paths_cfg['model_checkpoint'])


    if args.run_test:
        print("\n--- Loading final model for testing ---")
        assess_network_performance(
            guesser_network, testing_set, token_to_int, compute_device,
            vocabulary, full_freq_sorted,
            game_count=len(testing_set),
            assessment_name="Final Test"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Evaluate the Hangman Guesser Network.")
    parser.add_argument('--config-path', type=str, default='config.json', help='Path to the configuration JSON file.')
    parser.add_argument('--run-test', action='store_true', help='Run evaluation on the test set after training.')
    parser.set_defaults(use_pretrained_embeddings=True)
    
    parsed_args = parser.parse_args()
    main(parsed_args)

