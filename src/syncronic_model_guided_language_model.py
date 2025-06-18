import numpy as np

from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet

class SyncronicModelGuidedLanguageModel(ProbabilisticModel):

    def __init__(self, model:ProbabilisticModel, guiding_model:ProbabilisticModel, max_seq_length: int = None,  model_name: str = None, normalize_outputs = False, top_k = None, check_is_defined = False, undefined_ouput = None ):
        super().__init__()
        if model_name is None:
            self._model_name = model.name+"_"+guiding_model.name
        else:
            self._model_name = model_name
        if max_seq_length is None:
            if guiding_model is not None:
                self._max_seq_length = min(model._max_seq_length, guiding_model._max_seq_length)
            else: 
                self._max_seq_length = model._max_seq_length
        self._model = model
        self._guiding_model = guiding_model
        self._normalize_outputs = normalize_outputs
        self.query_cache = dict()
        if guiding_model is not None:
            assert model.alphabet == guiding_model.alphabet
            assert model.terminal_symbol == guiding_model.terminal_symbol
        self._alphabet = model.alphabet
        self._terminal_symbol = model.terminal_symbol
        self._top_k = top_k
        self.check_is_defined = check_is_defined
        self.undefined_output = undefined_ouput
        if self.check_is_defined:
            assert undefined_ouput is not None, "There should be an undefined output specified if checking if the sequence is defined"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def alphabet(self):
        return self._alphabet

    @property    
    def name(self) -> str:
        return self.model_name

    def _next_symbol_probas(self, sequence):
        if sequence not in self.query_cache:
            self.query_cache[sequence] = self.next_symbol_probas(sequence)
        return self.query_cache[sequence] 

    @property            
    def terminal_symbol(self) -> Symbol:
        return self._terminal_symbol

    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def last_token_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def _get_symbol_index(self, symbol):
        return self._model._get_symbol_index(symbol)
    
    def get_last_token_weights_batch(self, sequences, required_suffixes):
        final_results = []
        for sequence in sequences:
            final_results.append(self.get_last_token_weights(sequence, required_suffixes))
        return final_results
    

    def _compose_probas(self, probability_vector1, probability_vector2):
        assert len(probability_vector1) ==len(probability_vector2) 
        result = np.array(probability_vector1)*np.array(probability_vector2)  
        assert len(result) ==len(probability_vector1)      
        return result        
    
    def check_sequence_is_defined(self, sequence):
        symbols = list(self.alphabet.symbols)
        symbols.sort()
        symbols = [self.terminal_symbol] + symbols
        for i in range(len(sequence)):
            v_prob =  self._raw_last_token_weights(Sequence(sequence.value[:i]),required_suffixes=symbols)
            is_valid_transition = v_prob[symbols.index(sequence.value[i])]
            if not is_valid_transition:
                return False
        return True

    def _raw_last_token_weights(self, sequence, required_suffixes):
        if self._guiding_model is not None:
            guiding_results = self._guiding_model.get_last_token_weights(sequence, required_suffixes)
            projected_required_suffixes = [required_suffixes[i] for i in range(len(required_suffixes)) if guiding_results[i]]
        else:
            guiding_results =  [1] * len(required_suffixes)
            projected_required_suffixes = required_suffixes    
        model_results = self._model.get_last_token_weights(sequence, projected_required_suffixes)
        model_results_full = []
        j=0
        for i in range(len(guiding_results)):
            if guiding_results[i]>0:
                model_results_full.append(model_results[j])
                j+=1
            else:
                model_results_full.append(0)
        final_probas = self._compose_probas(model_results_full, guiding_results)
        if self._top_k is not None:
            final_probas = self._mask_elements_below_topk(final_probas, self._top_k)
        
        if self._normalize_outputs:
            final_probas = self.normalize(final_probas)
        
        return final_probas

    def get_last_token_weights(self, sequence, required_suffixes):
        assert len(required_suffixes)==len(self.alphabet)+1, 'required_suffixes should only be the alphabet'
        if self.check_is_defined:
            if not self.check_sequence_is_defined(sequence):
                return self.undefined_output
        
        return self._raw_last_token_weights(sequence, required_suffixes)
        
    
    def normalize(self, probas):
        if np.sum(probas)> 0:
            a = np.sum(np.array(probas)/np.sum(probas))
            return list(np.array(probas)/np.sum(probas))
        else:
            return probas

    def last_token_probabilities_batch(self, sequences: list[Sequence], required_suffixes: list[Sequence]) -> \
            list[list[float]]:
        return self.get_last_token_weights_batch(sequences, required_suffixes)
    
        
    def _mask_elements_below_topk(self, arr, topk):
        sorted_indices = np.argsort(arr, axis=None)
        mask = np.zeros_like(arr, dtype=bool)
        
        topk_indices = np.unravel_index(sorted_indices[-topk:], arr.shape)
        mask[topk_indices] = True
        
        return np.where(mask, arr, 0)
        