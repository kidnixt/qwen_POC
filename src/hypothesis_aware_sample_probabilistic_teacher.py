from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pymodelextractor.teachers.sample_probabilistic_teacher import SampleProbabilisticTeacher
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from pythautomata.utilities.guiding_wfa_sequence_generator import GuidingWDFASequenceGenerator
from typing import Union, Sized


class HypothesisAwareSampleProbabilisticTeacher(SampleProbabilisticTeacher):
    def __init__(self, model: ProbabilisticModel, comparator: FiniteAutomataComparator, sample_size: float = None, max_seq_length: int = 128, parallel_cache = False, max_query_elements = 1_000_000, batch_size = 10_000, cache_from_dataloader = None):
        super().__init__(model, comparator, sample_size, None,max_seq_length, False, parallel_cache, max_query_elements, batch_size, cache_from_dataloader)
        self._max_seq_length = max_seq_length

    def generate_words(self, aut):
        sequence_generator = GuidingWDFASequenceGenerator(aut, self._max_seq_length)
        rand_words = sequence_generator.generate_words(self._sample_size)
        rand_words.sort(key=len)
        return rand_words

    def equivalence_query(self, aut: WeightedAutomaton) -> Union[tuple[bool, Sized], tuple[bool, None]]:
        self._equivalence_queries_count += 1
        tried = set()
        suffixes = list()
        suffixes.append(self.terminal_symbol)
        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence([symbol]))
        rand_words = self.generate_words(aut)
        for word in rand_words:
            prefixes = sorted(word.get_prefixes(), key=len)
            for prefix in prefixes:
                if prefix not in tried:
                    obs1 = self.last_token_weights(prefix, suffixes)
                    obs2 = aut.get_last_token_weights(prefix, suffixes)
                    if not self._comparator.next_tokens_equivalent_output(obs1, obs2):
                        return False, prefix
                    tried.add(prefix)        
        return True, None