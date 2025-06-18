import torch
import time
import pandas as pd
import datetime
import os
import argparse
import numpy as np

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

    # --- Initial setup and timestamped folder creation ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output_pdfa_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print_if_not_silent(f"Creating output directory: '{output_dir}' to save results.")

    model_id, model, tokenizer, device, device_used, model_dtype_used = get_qwen_model_and_tokenizer()

    # Model configuration for the wrapper (adjust max_seq_length and terminal_symbol as needed for Qwen)
    # The alphabet here needs to be consistent with what Qwen can generate for floating points.
    # Assuming 'alphabet' is defined in guiding_wfas.floating_point_wfa_01 and includes '.', '0'-'9'.
    
    # Qwen tokenizer's eos_token might not be directly relevant for the wrapper's terminal symbol,
    # as the wrapper expects a specific 'terminal_symbol' for its internal logic.
    # For numeric generation, it might be a space or a specific end-of-number token.
    # For now, let's assume the wrapper's 'terminal_symbol' is meant to represent the end of a generated number.
    # If not, adjust `wrapper.terminal_symbol` below based on how your wrapper is designed to identify the end of a sequence.
    wrapper_terminal_symbol = tokenizer.eos_token if tokenizer.eos_token_id is not None else " " # Fallback if EOS is not a single token
    
    wrapper = Qwen3ProbabilisticModelWrapper(50, alphabet, device, model, tokenizer)
    
    property_model = get_floating_point_wfa_01(wrapper.terminal_symbol)
    synchronic_model = SyncronicModelGuidedLanguageModel(wrapper, 
                                                         property_model, 
                                                         model_name="GUIDED_QWEN", 
                                                         max_seq_length=50, # Increased max_seq_length for Qwen context
                                                         normalize_outputs=True)
    
    partitioner = QuantizationProbabilityPartitionerPlus(100)
    comparator = WFAPartitionComparator(partitioner)
    max_states = 15
    max_query_length = 6 # This is for the learning process, not for sample generation
    teacher = HypothesisAwareSampleProbabilisticTeacher(synchronic_model, 
                                                        comparator = comparator, 
                                                        max_seq_length = 4, 
                                                        sample_size = 100)
    
    # Time bound is set to 30 seconds for the learner
    learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner = partitioner, 
                                                     max_states = max_states, 
                                                     max_query_length = max_query_length, 
                                                     max_seconds_run = 30, 
                                                     generate_partial_hipothesis = True, 
                                                     pre_cache_queries_for_building_hipothesis = True,  
                                                     check_probabilistic_hipothesis = True, 
                                                     mean_distribution_for_partial_hipothesis = True, 
                                                     omit_zero_transitions = True)

    print_if_not_silent("\nStarting the PDFA learning process...")
    learning_start_time = time.time()
    learning_result = learner.learn(teacher, verbose=False)
    learning_end_time = time.time()
    pdfa = learning_result.model
    print_if_not_silent(f"PDFA learning completed in {learning_end_time - learning_start_time:.2f} seconds.")

    print_if_not_silent("\nExporting learned PDFA to DOT file...")
    exporter = WFADotExportingStrategy()
    pdfa_filename = f"pdfa_{timestamp}"
    exporter.export(pdfa, output_dir, pdfa_filename)
    print_if_not_silent(f"PDFA exported to '{os.path.join(output_dir, pdfa_filename)}.dot'")

    floating_points = []
    num_samples = 10_000 # Number of floating point numbers to generate
    
    total_samples_completed = 0
    total_time_for_completed_samples = 0.0
    eta_data = []

    print_if_not_silent(f"\nStarting the sampling process from the learned PDFA for {num_samples} floating-point numbers...")

    for i in range(num_samples):        
        sample_start_time = time.time()
        
        
        number = get_representative_sample(pdfa, sample_size = 1, max_length = 10)
        number_string = str(number)
        
        result = number_string.replace('[', '').replace(']', '').replace(',', '')

        floating_points.append(result)

        sample_end_time = time.time()
        sample_duration = sample_end_time - sample_start_time

        total_samples_completed += 1
        total_time_for_completed_samples += sample_duration

        print_if_not_silent(f"--- Generated Floating Point Number {i + 1}/{num_samples} ---")
        print_if_not_silent(f"Number {i + 1} completed. Result: '{result}' (Length: {len(result)})")
        print_if_not_silent(f"Generation time for this number: {sample_duration:.4f} seconds.")

        # --- ETA calculation (printed every 10 numbers, but calculated and saved for each) ---
        if total_samples_completed > 0:
            avg_time_per_number = total_time_for_completed_samples / total_samples_completed
            remaining_numbers = num_samples - total_samples_completed
            total_estimated_remaining_time = remaining_numbers * avg_time_per_number
            
            if total_samples_completed == 1 or total_samples_completed % 10 == 0 or total_samples_completed == num_samples:
                print(f"  ETA: {total_estimated_remaining_time:.2f} seconds remaining (approx. {avg_time_per_number:.4f} s/number)")
            
            eta_data.append([i + 1, total_samples_completed, avg_time_per_number, total_estimated_remaining_time])

    print_if_not_silent(f"\nAll {num_samples} floating-point numbers generated from PDFA.")
    print_if_not_silent(f"Total generation time for {num_samples} numbers: {total_time_for_completed_samples:.2f} seconds.")
    
    final_avg_time_per_number = 0.0
    if total_samples_completed > 0:
        final_avg_time_per_number = total_time_for_completed_samples / total_samples_completed
    print_if_not_silent(f"Average time per floating-point number: {final_avg_time_per_number:.4f} seconds/number.")

    # --- Save outputs to the timestamped folder ---
    # CSV of generated numbers
    output_filename = os.path.join(output_dir, f"{timestamp}_pdfa_floating_points.csv")
    df = pd.DataFrame(floating_points, columns=["floating-point"])
    df.to_csv(output_filename, index=False)
    print_if_not_silent(f"Results of generated numbers saved to '{output_filename}'.")

    # Intermediate ETA data
    eta_filename = os.path.join(output_dir, f"{timestamp}_eta_data.csv")
    eta_df = pd.DataFrame(eta_data, columns=['Floating Point Number', 'Total Completed', 'Time/Number (s)', 'ETA (s)'])
    eta_df.to_csv(eta_filename, index=False)
    print_if_not_silent(f"Intermediate ETA data saved to '{eta_filename}'.")

    # Summary generation metrics
    summary_metrics_filename = os.path.join(output_dir, f"{timestamp}_generation_summary_metrics.txt")
    with open(summary_metrics_filename, "w") as f:
        f.write(f"--- Execution Parameters ---\n")
        f.write(f"Date and Time of Execution: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Execution Device: {device_used}\n")
        f.write(f"Model Data Type (dtype): {model_dtype_used}\n")
        f.write(f"Tokenizer EOS Token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})\n")
        f.write(f"Random Seed: 42 (fixed for reproducibility)\n")
        f.write(f"--------------------------------\n")
        f.write(f"Number of Floating Points to Generate: {num_samples}\n")
        f.write(f"--------------------------------\n")
        f.write(f"--- Generation Metrics ---\n")
        f.write(f"Total floating points generated: {total_samples_completed}\n")
        f.write(f"Total generation time: {total_time_for_completed_samples:.2f} seconds\n")
        f.write(f"Average time per floating point: {final_avg_time_per_number:.4f} seconds/number\n")
        f.write(f"PDFA Learning Time: {learning_end_time - learning_start_time:.2f} seconds\n")
        f.write(f"--------------------------------\n")
    print_if_not_silent(f"Summary generation metrics and parameters saved to '{summary_metrics_filename}'.")


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