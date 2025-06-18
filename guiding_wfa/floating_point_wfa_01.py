from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.state import State
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.model_comparators.dfa_comparison_strategy import \
    DFAComparisonStrategy as DFAComparator
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton



dot = SymbolStr(".")
zero = SymbolStr("0")
one = SymbolStr("1")
two = SymbolStr("2")
three = SymbolStr("3")
four = SymbolStr("4")
five = SymbolStr("5")
six = SymbolStr("6")
seven = SymbolStr("7")
eight = SymbolStr("8")
nine = SymbolStr("9")

numbers = [zero, one, two, three, four, five, six, seven, eight, nine]
symbols = numbers.copy()
symbols.append(dot)
symbols.sort()
alphabet = Alphabet(frozenset(symbols))

def _get_symbol_index(symbol: SymbolStr):
    return symbols.index(symbol)

def get_floating_point_wfa_01(terminal_symbol):
    stateInitial = WeightedState("initial", 1,0, terminal_symbol)
    stateNumbers = WeightedState("numbers", 0,1, terminal_symbol)    
    stateDot = WeightedState("dot", 0,0, terminal_symbol)
    hole = WeightedState("hole", 0, 0, terminal_symbol)

   
    stateInitial.add_transition(dot, stateDot, 1) 
    for number in numbers:
        stateDot.add_transition(number, stateNumbers, 1)
        stateNumbers.add_transition(number, stateNumbers, 1)
    
    states = frozenset({stateInitial, stateNumbers, stateDot, hole})

    for state in states:
        for symbol in alphabet.symbols:
            if symbol not in state.transitions_set:
                state.add_transition(symbol, hole, 0)


    comparator = None
    return ProbabilisticDeterministicFiniteAutomaton(alphabet, states, terminal_symbol, comparator, "Floating_Point_WFA", check_is_probabilistic = False)    