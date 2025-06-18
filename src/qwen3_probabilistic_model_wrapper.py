from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
import torch

class Qwen3ProbabilisticModelWrapper(ProbabilisticModel):

    def __init__(self, max_seq_length: int, alphabet: Alphabet, device: str, model, tokenizer, prompt: Sequence = Sequence()):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self._alphabet = alphabet
        self._prompt = prompt
        self._max_seq_length = max_seq_length

    @property
    def name(self) -> str:
        return "Qwen3"

    @property
    def terminal_symbol(self) -> Symbol:
        return SymbolStr(self.tokenizer.eos_token)

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet

    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def last_token_probability(self, sequence: Sequence, symbols=None):
        if symbols is None:
            symbols = set(self._alphabet.symbols)
            symbols.add(self.terminal_symbol)
        
        if len(sequence) == 0:  # Handle empty sequence
            if not symbols:
                return {}
            # Return uniform distribution over all symbols
            return {symbol: 1.0/len(symbols) for symbol in symbols}
        
        return self._get_probability(sequence, symbols)

    def get_last_token_weights(self, sequence, required_suffixes):
        symbol_probs = self.last_token_probability(sequence, required_suffixes)
        return [symbol_probs[suffix] for suffix in required_suffixes]

    def get_last_token_weights_batch(self, sequences, required_suffixes):
        return [self.get_last_token_weights(seq, required_suffixes) for seq in sequences]

    def _build_input_ids(self, sequence: Sequence):
        prompt_str = "".join(str(x) for x in self._prompt)
        sequence_str = "".join(str(x) for x in sequence)
        full_input = prompt_str + sequence_str
        
        if not full_input.strip():  # Handle empty input
            return torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)
            
        input_ids = self.tokenizer(full_input, return_tensors="pt").input_ids.to(self.device)
        return input_ids.long()

    def _get_probability(self, sequence: Sequence, symbols):
        input_ids = self._build_input_ids(sequence)
        if input_ids.numel() == 0:  # Check for empty input
            return {symbol: 0.0 for symbol in symbols}  # Return uniform or zero probabilities
        
        with torch.no_grad():
            output = self.model(input_ids)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        return self._get_symbols_probabilities_dict(probs[0], symbols)

    def _get_symbols_probabilities_dict(self, last_token_probs, symbols):
        symbol_probs = {}
        for symbol in symbols:
            tokens = self.tokenizer.tokenize(str(symbol))
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(token_ids) == 0:
                symbol_probs[symbol] = 0.0
                continue
            prob = last_token_probs[token_ids[0]]
            # Probabilidad para símbolos multitérmino (subwords)
            for prev_token_id in token_ids[1:]:
                prob = prob * 1e-6  # Penalización: Qwen3 no permite evaluar directamente multi-token sin cadena
            symbol_probs[symbol] = prob.item()
        return symbol_probs
