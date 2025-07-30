import torch
import time
import pandas as pd
import datetime
import os
import argparse
import numpy as np
import json

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from src.qwen3_probabilistic_model_wrapper import Qwen3ProbabilisticModelWrapper
# Assuming these are available or will be provided, adapting from your first script
from guiding_wfa.floating_point_wfa_01 import alphabet, get_floating_point_wfa_01
from src.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel
from src.hypothesis_aware_sample_probabilistic_teacher import HypothesisAwareSampleProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.pdfa_operations import get_representative_sample
from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitionerPlus
from pythautomata.model_exporters.dot_exporters.wfa_dot_exporting_strategy import WFADotExportingStrategy

# Variables globales para controlar la verbosidad
VERBOSE = False
SILENT = False

def print_if_not_silent(message):
    """Auxiliary function to print only if SILENT is not True."""
    if not SILENT:
        print(message)

def print_if_verbose(message):
    """Auxiliary function to print only if VERBOSE is True and SILENT is not True."""
    if VERBOSE and not SILENT:
        print(message)


def get_pdfa_summary_stats(pdfa) -> dict:
    num_states = len(pdfa.weighted_states)
    num_transitions = sum(
        sum(len(transitions) for transitions in state.transitions_list.values())
        for state in pdfa.weighted_states
    )
    alphabet_size = len(pdfa.alphabet.symbols)
    num_final_states = sum(1 for state in pdfa.weighted_states if state.final_weight > 0)
    initial_state = pdfa.get_first_state().name if pdfa.get_first_state() else "None"

    # Convertir el terminal_symbol a string simple
    terminal_symbol_str = str(pdfa.terminal_symbol)
    # Si terminal_symbol es tipo Sequence, podrías necesitar extraer la representación correcta, por ejemplo:
    # terminal_symbol_str = ''.join(map(str, pdfa.terminal_symbol.value)) if hasattr(pdfa.terminal_symbol, 'value') else str(pdfa.terminal_symbol)

    return {
        "name": str(pdfa.name),
        "num_states": num_states,
        "num_final_states": num_final_states,
        "alphabet_size": alphabet_size,
        "num_transitions": num_transitions,
        "initial_state": str(initial_state),
        "terminal_symbol": terminal_symbol_str
    }



def get_qwen_model_and_tokenizer():
    torch.manual_seed(42)

    model_id = "Qwen/Qwen3-1.7B"
    device_used = ""
    model_dtype_used = ""

    if torch.cuda.is_available():
        device = "cuda"
        device_used = "cuda"
        print_if_not_silent("CUDA is available. Using GPU.")
        print_if_not_silent(f"Number of available GPUs: {torch.cuda.device_count()}")
        print_if_not_silent(f"Name of the current GPU: {torch.cuda.get_device_name(0)}")
        if VERBOSE:
            print_if_verbose(f"Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB allocated")
    else:
        device = "cpu"
        device_used = "cpu"
        print_if_not_silent("CUDA is NOT available. Using CPU (execution will be much slower).")

    print_if_not_silent(f"\nLoading tokenizer for model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    print_if_not_silent("Tokenizer loaded successfully.")

    print_if_not_silent(f"Loading model: {model_id} on {device} with torch_dtype=torch.bfloat16...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     pad_token_id=tokenizer.eos_token_id,
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).to(device)
        model_dtype_used = "torch.bfloat16"
        print_if_not_silent("Model loaded successfully on GPU (bfloat16).")
        if VERBOSE:
            print_if_verbose(f"GPU memory after loading model: {torch.cuda.memory_allocated() / (1024**3):.2f} GB allocated")
    except Exception as e:
        print_if_not_silent(f"ERROR loading model in bfloat16! {e}")
        print_if_not_silent("Attempting to load on CPU as fallback if CUDA failed or another issue occurred.")
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     pad_token_id=tokenizer.eos_token_id,
                                                     trust_remote_code=True).to("cpu")
        model_dtype_used = "torch.float32 (CPU fallback)" # Assuming default float32 on CPU
        print_if_not_silent("Model loaded on CPU.")
    
    return model_id, model, tokenizer, device, device_used, model_dtype_used

def sample():
    global VERBOSE, SILENT

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output_pdfa_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print_if_not_silent(f"Creating output directory: '{output_dir}' to save results.")

    model_id, model, tokenizer, device, device_used, model_dtype_used = get_qwen_model_and_tokenizer()

    wrapper_terminal_symbol = tokenizer.eos_token if tokenizer.eos_token_id is not None else " "
    wrapper = Qwen3ProbabilisticModelWrapper(50, alphabet, device, model, tokenizer)

    property_model = get_floating_point_wfa_01(wrapper.terminal_symbol)
    synch_max_seq_length = 50
    synch_normalize_outputs = True
    synchronic_model = SyncronicModelGuidedLanguageModel(wrapper,
                                                         property_model,
                                                         model_name="GUIDED_QWEN",
                                                         max_seq_length=synch_max_seq_length,
                                                         normalize_outputs=synch_normalize_outputs)

    kappa = 1000
    partitioner = QuantizationProbabilityPartitionerPlus(kappa)
    comparator = WFAPartitionComparator(partitioner)

    teacher_sample_size = 100
    teacher_max_seq_length = 5
    teacher = HypothesisAwareSampleProbabilisticTeacher(synchronic_model,
                                                        comparator=comparator,
                                                        max_seq_length=teacher_max_seq_length,
                                                        sample_size=teacher_sample_size)

    learner_max_states = 15
    learner_max_query_length = 10
    learner_max_seconds_run = 30
    learner_generate_partial_hipothesis = True
    learner_pre_cache_queries_for_building_hipothesis = True
    learner_check_probabilistic_hipothesis = True
    learner_mean_distribution_for_partial_hipothesis = True
    learner_omit_zero_transitions = True

    learner = BoundedPDFAQuantizationNAryTreeLearner(
        partitioner=partitioner,
        max_states=learner_max_states,
        max_query_length=learner_max_query_length,
        max_seconds_run=learner_max_seconds_run,
        generate_partial_hipothesis=learner_generate_partial_hipothesis,
        pre_cache_queries_for_building_hipothesis=learner_pre_cache_queries_for_building_hipothesis,
        check_probabilistic_hipothesis=learner_check_probabilistic_hipothesis,
        mean_distribution_for_partial_hipothesis=learner_mean_distribution_for_partial_hipothesis,
        omit_zero_transitions=learner_omit_zero_transitions)

    print_if_not_silent("\nStarting the PDFA learning process...")
    learning_start_time = time.time()
    learning_result = learner.learn(teacher, verbose=True)
    learning_end_time = time.time()
    pdfa = learning_result.model
    print_if_not_silent(f"PDFA learning completed in {learning_end_time - learning_start_time:.2f} seconds.")



    pdfa_summary = get_pdfa_summary_stats(pdfa)
    summary_filename = os.path.join(output_dir, f"{timestamp}_pdfa_summary.json")
    
    with open(summary_filename, "w") as f:
        json.dump(pdfa_summary, f, indent=4)


    print_if_not_silent("\nExporting learned PDFA to DOT file...")
    exporter = WFADotExportingStrategy()
    pdfa_filename = f"pdfa_{timestamp}"
    exporter.export(pdfa, output_dir, pdfa_filename)
    print_if_not_silent(f"PDFA exported to '{os.path.join(output_dir, pdfa_filename)}.dot'")

    floating_points = []
    num_samples = 10_0
    total_samples_completed = 0
    total_time_for_completed_samples = 0.0
    eta_data = []

    print_if_not_silent(f"\nStarting the sampling process from the learned PDFA for {num_samples} floating-point numbers...")

    for i in range(num_samples):
        sample_start_time = time.time()
        number = get_representative_sample(pdfa, sample_size=1, max_length=10)
        result = str(number).replace('[', '').replace(']', '').replace(',', '')
        floating_points.append(result)
        sample_end_time = time.time()

        total_samples_completed += 1
        total_time_for_completed_samples += sample_end_time - sample_start_time

        # ETA every 1000 samples
        if total_samples_completed % 1000 == 0 or total_samples_completed == 1 or total_samples_completed == num_samples:
            avg_time_per_number = total_time_for_completed_samples / total_samples_completed
            remaining = num_samples - total_samples_completed
            eta_sec = remaining * avg_time_per_number
            print(f"[ETA] {total_samples_completed}/{num_samples} - Remaining: {eta_sec:.2f}s - Avg: {avg_time_per_number:.4f}s")
            eta_data.append([i + 1, total_samples_completed, avg_time_per_number, eta_sec])

    print_if_not_silent(f"\nAll {num_samples} floating-point numbers generated from PDFA.")
    print_if_not_silent(f"Total generation time: {total_time_for_completed_samples:.2f} seconds.")

    avg_time = total_time_for_completed_samples / total_samples_completed if total_samples_completed else 0.0

    # Save generated floating points
    df = pd.DataFrame(floating_points, columns=["floating-point"])
    output_filename = os.path.join(output_dir, f"{timestamp}_pdfa_v2_floating_points.csv")
    df.to_csv(output_filename, index=False)
    print_if_not_silent(f"Results saved to '{output_filename}'.")

    # Save ETA data
    eta_filename = os.path.join(output_dir, f"{timestamp}_eta_data.csv")
    eta_df = pd.DataFrame(eta_data, columns=['Floating Point Number', 'Total Completed', 'Time/Number (s)', 'ETA (s)'])
    eta_df.to_csv(eta_filename, index=False)
    print_if_not_silent(f"ETA data saved to '{eta_filename}'.")

    # Save summary
    summary_filename = os.path.join(output_dir, f"{timestamp}_generation_summary_metrics.txt")
    with open(summary_filename, "w") as f:
        f.write(f"--- Execution Parameters ---\n")
        f.write(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Execution Device: {device_used}\n")
        f.write(f"Model Dtype: {model_dtype_used}\n")
        f.write(f"Tokenizer EOS Token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})\n")
        f.write(f"Random Seed: 42\n")
        f.write(f"--------------------------------\n")
        f.write(f"Number of Floating Points: {num_samples}\n")
        f.write(f"Total Time: {total_time_for_completed_samples:.2f}s\n")
        f.write(f"Average Time per Number: {avg_time:.4f}s\n")
        f.write(f"PDFA Learning Time: {learning_end_time - learning_start_time:.2f}s\n")
        f.write(f"--------------------------------\n")
        f.write(f"[SyncronicModelGuidedLanguageModel]\n")
        f.write(f"  model_name: {model_id}\n")
        f.write(f"  max_seq_length: {synch_max_seq_length}\n")
        f.write(f"  normalize_outputs: {synch_normalize_outputs}\n\n")
        f.write(f"[HypothesisAwareSampleProbabilisticTeacher]\n")
        f.write(f"  sample_size: {teacher_sample_size}\n")
        f.write(f"  max_seq_length: {teacher_max_seq_length}\n\n")
        f.write(f"[QuantizationProbabilityPartitionerPlus]\n")
        f.write(f"  kappa: {kappa}\n\n")
        f.write(f"[BoundedPDFAQuantizationNAryTreeLearner]\n")
        f.write(f"  max_states: {learner_max_states}\n")
        f.write(f"  max_query_length: {learner_max_query_length}\n")
        f.write(f"  max_seconds_run: {learner_max_seconds_run}\n")
        f.write(f"  generate_partial_hipothesis: {learner_generate_partial_hipothesis}\n")
        f.write(f"  pre_cache_queries_for_building_hipothesis: {learner_pre_cache_queries_for_building_hipothesis}\n")
        f.write(f"  check_probabilistic_hipothesis: {learner_check_probabilistic_hipothesis}\n")
        f.write(f"  mean_distribution_for_partial_hipothesis: {learner_mean_distribution_for_partial_hipothesis}\n")
        f.write(f"  omit_zero_transitions: {learner_omit_zero_transitions}\n")


    print_if_not_silent(f"Summary metrics saved to '{summary_filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for sampling from a PDFA learned with a Qwen model.")
    parser.add_argument('--verbose', action='store_true', help='Enables detailed log printing.')
    parser.add_argument('--silent', action='store_true', help='Disables all log printing, except critical errors and ETA.')
    args = parser.parse_args()

    VERBOSE = args.verbose
    SILENT = args.silent
    if SILENT:
        VERBOSE = False

    sample()
