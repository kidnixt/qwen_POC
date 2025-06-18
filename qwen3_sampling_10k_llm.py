import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import os
import argparse
import time
import datetime

# Variables globales para controlar la verbosidad
VERBOSE = False
SILENT = False

def calculate_probs(prompt, eos, model, tokenizer, device):
    tokens = tokenizer.tokenize("".join(prompt))
    prompt_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([prompt_ids], device=device)

    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]

    numbers = ["0", "1", "2","3","4","5","6","7","8","9"]
    indexes = [tokenizer.encode(number, add_special_tokens=False) for number in numbers]

    if eos:
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            indexes.append([eos_id])

    word_probs = {}
    for idx_list in indexes:
        token_id = idx_list[0]
        prob = probs[token_id]
        word = tokenizer.decode([token_id]).strip()
        word_probs[word] = prob.item()

    total = sum(word_probs.values())
    normalized_word_probs = {word: p / total for word, p in word_probs.items()}
    return normalized_word_probs

def print_if_not_silent(message):
    """Función auxiliar para imprimir solo si SILENT no es True."""
    if not SILENT:
        print(message)

def print_if_verbose(message):
    """Función auxiliar para imprimir solo si VERBOSE es True y SILENT no es True."""
    if VERBOSE and not SILENT:
        print(message)

def sample():
    global VERBOSE, SILENT

    torch.manual_seed(42) # Semilla fija para reproducibilidad

    # --- Configuración inicial y creación de carpeta con timestamp ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print_if_not_silent(f"Creando directorio de salida: '{output_dir}' para guardar los resultados.")

    # --- Parámetros que queremos registrar ---
    model_id = "Qwen/Qwen3-1.7B"
    num_samples = 10_000     # Cantidad de números flotantes a generar
    min_digits = 1       # Mínimo de dígitos por número flotante
    max_digits = 10      # Máximo de dígitos por número flotante (ajustado de 50 a 10 para pruebas rápidas)
    
    device_used = ""
    model_dtype_used = ""

    if torch.cuda.is_available():
        device = "cuda"
        device_used = "cuda"
        print_if_not_silent("CUDA está disponible. Usando GPU.")
        print_if_not_silent(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
        print_if_not_silent(f"Nombre de la GPU actual: {torch.cuda.get_device_name(0)}")
        if VERBOSE:
            print_if_verbose(f"Memoria de la GPU inicial: {torch.cuda.memory_allocated() / (1024**3):.2f} GB asignados")
    else:
        device = "cpu"
        device_used = "cpu"
        print_if_not_silent("CUDA NO está disponible. Usando CPU (la ejecución será mucho más lenta).")

    print_if_not_silent(f"\nCargando el tokenizer para el modelo: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    print_if_not_silent("Tokenizer cargado exitosamente.")

    print_if_not_silent(f"Cargando el modelo: {model_id} en {device} con torch_dtype=torch.bfloat16...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     pad_token_id=tokenizer.eos_token_id,
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).to(device)
        model_dtype_used = "torch.bfloat16"
        print_if_not_silent("Modelo cargado exitosamente en la GPU (bfloat16).")
        if VERBOSE:
            print_if_verbose(f"Memoria de la GPU después de cargar el modelo: {torch.cuda.memory_allocated() / (1024**3):.2f} GB asignados")
    except Exception as e:
        print_if_not_silent(f"¡ERROR al cargar el modelo en bfloat16! {e}")
        print_if_not_silent("Intentando cargar en CPU como fallback si CUDA falló o hay otro problema.")
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     pad_token_id=tokenizer.eos_token_id,
                                                     trust_remote_code=True).to("cpu")
        model_dtype_used = "torch.float32 (CPU fallback)" # Asumiendo default float32 en CPU
        print_if_not_silent("Modelo cargado en CPU.")


    results = []
    total_samples_completed = 0
    total_time_for_completed_samples = 0.0 # Acumula el tiempo de las muestras COMPLETADAS
    
    # Para el registro del ETA (ahora basado en números generados)
    eta_data = [] 

    print_if_not_silent(f"\nComenzando el proceso de muestreo para {num_samples} números flotantes...")

    for i in range(num_samples):
        prompt = ["."]
        next_token = ""
        digit_count = 0
        
        print_if_not_silent(f"--- Generando Número Flotante {i + 1}/{num_samples} ---")
        current_step = 0 # Reiniciar el contador de pasos de dígitos para cada número
        
        sample_start_time = time.time() # Iniciar el tiempo para esta muestra/número flotante
        
        while next_token != tokenizer.eos_token and digit_count < max_digits:
            eos = digit_count >= min_digits
            probs = calculate_probs(prompt, eos, model, tokenizer, device)
            
            if VERBOSE and tokenizer.eos_token in probs:
                print_if_verbose(f"  Paso de dígito {current_step}: Proba EOS = {probs[tokenizer.eos_token]:.4f}")
            
            next_token = np.random.choice(list(probs.keys()), p=list(probs.values()))
            
            if next_token != tokenizer.eos_token:
                prompt.append(next_token)
                digit_count += 1
            
            current_step += 1 # Cuenta los dígitos generados para la muestra actual
            
            if VERBOSE and current_step % 5 == 0:
                print_if_verbose(f"  Paso de dígito {current_step}: {digit_count} dígitos generados para el número actual. Último token: '{next_token}'")
            
        sample_end_time = time.time() # Finalizar el tiempo para esta muestra/número flotante
        sample_duration = sample_end_time - sample_start_time
        
        final_result = ''.join(prompt[1:])
        results.append(final_result) # Agrega el número flotante completo a los resultados

        total_samples_completed += 1 # Un número flotante más completado
        total_time_for_completed_samples += sample_duration # Acumular el tiempo de esta muestra

        if next_token != tokenizer.eos_token:
            print_if_not_silent(f"  ⚠️  No se alcanzó EOS en {max_digits} dígitos para este número. Resultado incompleto.")
        else:
            print_if_not_silent(f"  ✅ EOS alcanzado en {digit_count} dígitos para este número.")
        
        print_if_not_silent(f"Número Flotante {i + 1} completado. Resultado: '{final_result}' (Longitud de dígitos: {len(final_result)})")
        print_if_not_silent(f"Tiempo de generación para este número: {sample_duration:.2f} segundos.")
        
        # --- Cálculo de ETA (se imprime cada 10 números, pero se calcula y guarda para cada uno) ---
        if total_samples_completed > 0:
            avg_time_per_number = total_time_for_completed_samples / total_samples_completed
            remaining_numbers = num_samples - total_samples_completed
            total_estimated_remaining_time = remaining_numbers * avg_time_per_number
            
            # Solo imprimir si es el primer número, el último, o un múltiplo de 10
            if total_samples_completed == 1 or total_samples_completed % 10 == 0 or total_samples_completed == num_samples:
                print(f"  ETA: {total_estimated_remaining_time:.2f} segundos restantes (aprox. {avg_time_per_number:.2f} s/número)")
            
            # Registrar los datos de ETA en el CSV para cada número generado
            eta_data.append([i + 1, total_samples_completed, avg_time_per_number, total_estimated_remaining_time])


    print_if_not_silent(f"\nTodas las {num_samples} números flotantes generados.")
    print_if_not_silent(f"Tiempo total de generación para {num_samples} números: {total_time_for_completed_samples:.2f} segundos.")
    
    # Calcular el tiempo promedio por número final
    final_avg_time_per_number = 0.0
    if total_samples_completed > 0:
        final_avg_time_per_number = total_time_for_completed_samples / total_samples_completed
    print_if_not_silent(f"Tiempo promedio por número flotante: {final_avg_time_per_number:.2f} segundos/número.")

    # --- Guardar los outputs en la carpeta con timestamp ---
    # CSV de números generados
    output_filename = os.path.join(output_dir, f"{timestamp}_llm_floating_points.csv")
    df = pd.DataFrame(results, columns=["floating-point"])
    df.to_csv(output_filename, index=False)
    print_if_not_silent(f"Resultados de los números generados guardados en '{output_filename}'.")

    # Datos de ETA intermedios
    eta_filename = os.path.join(output_dir, f"{timestamp}_eta_data.csv")
    eta_df = pd.DataFrame(eta_data, columns=['Número Flotante', 'Total Completados', 'Tiempo/Número (s)', 'ETA (s)'])
    eta_df.to_csv(eta_filename, index=False)
    print_if_not_silent(f"Datos de ETA intermedios guardados en '{eta_filename}'.")

    # Métricas de resumen de generación
    summary_metrics_filename = os.path.join(output_dir, f"{timestamp}_generation_summary_metrics.txt")
    with open(summary_metrics_filename, "w") as f:
        f.write(f"--- Parámetros de Ejecución ---\n")
        f.write(f"Fecha y Hora de Ejecución: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio de Salida: {output_dir}\n")
        f.write(f"ID del Modelo: {model_id}\n")
        f.write(f"Dispositivo de Ejecución: {device_used}\n")
        f.write(f"Tipo de Datos del Modelo (dtype): {model_dtype_used}\n")
        f.write(f"Token EOS del Tokenizer: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})\n")
        f.write(f"Semilla Aleatoria: 42 (fija para reproducibilidad)\n")
        f.write(f"--------------------------------\n")
        f.write(f"Cantidad de Números Flotantes a Generar: {num_samples}\n")
        f.write(f"Dígitos Mínimos por Número: {min_digits}\n")
        f.write(f"Dígitos Máximos por Número: {max_digits}\n")
        f.write(f"--------------------------------\n")
        f.write(f"--- Métricas de Generación ---\n")
        f.write(f"Total de números flotantes generados: {total_samples_completed}\n")
        f.write(f"Tiempo total de generación: {total_time_for_completed_samples:.2f} segundos\n")
        f.write(f"Tiempo promedio por número flotante: {final_avg_time_per_number:.2f} segundos/número\n")
        f.write(f"--------------------------------\n")
    print_if_not_silent(f"Métricas de generación resumidas y parámetros guardados en '{summary_metrics_filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para muestreo de un modelo Qwen.")
    parser.add_argument('--verbose', action='store_true', help='Activa la impresión de logs detallados.')
    parser.add_argument('--silent', action='store_true', help='Desactiva toda la impresión de logs, excepto errores críticos y el ETA.')
    args = parser.parse_args()

    VERBOSE = args.verbose
    SILENT = args.silent

    if SILENT:
        VERBOSE = False
    
    sample()