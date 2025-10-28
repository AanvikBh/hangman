import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class HangmanDataset(Dataset):
    """Generates training items by masking characters in terms."""
    def __init__(self, term_list, token_map, conceal_ratio=0.4, variations_per_term=5):
        self.term_collection = term_list
        self.token_integer_map = token_map
        self.mask_int_val = token_map['_']
        self.conceal_probability = conceal_ratio
        self.num_variations = variations_per_term
        self.data_samples = self._generate_samples()

    def _generate_samples(self):
        generated_items = []
        for term in self.term_collection:
            for _ in range(self.num_variations):
                original_ints = [self.token_integer_map[c] for c in term]
                concealed_ints = list(original_ints)
                ground_truth_ints = [-1] * len(term)
                
                num_to_conceal = max(1, int(len(term) * self.conceal_probability))
                conceal_locs = random.sample(range(len(term)), num_to_conceal)
                
                for loc in conceal_locs:
                    ground_truth_ints[loc] = concealed_ints[loc]
                    concealed_ints[loc] = self.mask_int_val
                
                generated_items.append((concealed_ints, ground_truth_ints))
        return generated_items

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, item_idx):
        concealed_seq, truth_seq = self.data_samples[item_idx]
        return torch.tensor(concealed_seq, dtype=torch.long), torch.tensor(truth_seq, dtype=torch.long)

def batching_processor(data_batch):
    """Pads sequences within a batch to equal length."""
    input_sequences, target_sequences = zip(*data_batch)
    
    sequence_lengths = torch.tensor([len(seq) for seq in input_sequences], dtype=torch.long)
    
    padded_inputs = pad_sequence(input_sequences, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(target_sequences, batch_first=True, padding_value=-1)
    
    return padded_inputs, padded_targets, sequence_lengths
