import torch
from tqdm import tqdm

def fetch_word_list(lexicon_path="words_250000_train.txt"):
    """Retrieves and filters a word list from a file."""
    print(f"Loading lexicon from {lexicon_path}...")
    with open(lexicon_path, 'r') as file_handle:
        entries = [line.strip().lower() for line in file_handle if line.strip()]
    print(f"Loaded {len(entries)} entries.")
    
    filtered_entries = [e for e in entries if 3 <= len(e) <= 15 and e.isalpha()]
    print(f"Retaining {len(filtered_entries)} valid entries after filtering.")
    return filtered_entries

def generate_token_maps(vocabulary_list):
    """Builds mappings from characters to integers and vice versa."""
    distinct_chars = sorted(list(set("".join(vocabulary_list))))
    token_to_int = {'_': 0}  # MASK token
    token_to_int.update({char: i + 1 for i, char in enumerate(distinct_chars)})
    int_to_token = {i: char for char, i in token_to_int.items()}
    return token_to_int, int_to_token

def persist_checkpoint(neural_net, sgd_optimizer, run_log, char_map, destination_path):
    """Saves the network state and training progress."""
    torch.save({
        'network_state_dict': neural_net.state_dict(),
        'optimizer_state_dict': sgd_optimizer.state_dict(),
        'training_log': run_log,
        'token_integer_map': char_map,
    }, destination_path)
    print(f"Checkpoint saved to {destination_path}")

def restore_checkpoint(destination_path, network_class, *constructor_args, **constructor_kwargs):
    """Loads a saved network checkpoint."""
    compute_target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot = torch.load(destination_path, map_location=compute_target)
    
    token_integer_map = snapshot['token_integer_map']
    char_count = len(token_integer_map)
    
    instance = network_class(vocab_size=char_count, *constructor_args, **constructor_kwargs)
    instance.load_state_dict(snapshot['network_state_dict'])
    instance.to(compute_target)
    
    print(f"Restored network from {destination_path}")
    return instance, token_integer_map, snapshot['training_log']
