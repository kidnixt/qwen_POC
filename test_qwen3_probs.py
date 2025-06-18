import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import sys 

# Función para obtener probabilidades de tokens específicos
def get_specific_token_probs(prompt, model, tokenizer, device, tokens_to_check):
    tokens = tokenizer.tokenize("".join(prompt))
    prompt_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([prompt_ids], device=device)

    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]

    specific_probs = {}
    for token_str in tokens_to_check:
        if token_str == tokenizer.eos_token:
            token_id = tokenizer.eos_token_id
        elif token_str == tokenizer.pad_token:
            token_id = tokenizer.pad_token_id
        else:
            encoded = tokenizer.encode(token_str, add_special_tokens=False)
            token_id = encoded[0] if len(encoded) == 1 else None
        
        if token_id is not None and token_id < len(probs):
            specific_probs[token_str] = probs[token_id].item()
        else:
            specific_probs[token_str] = 0.0
            
    return specific_probs

# Función que ejecuta una única iteración de la prueba de probabilidades
def _run_single_prob_test_iteration(
    model, tokenizer, device, 
    im_end_token, endoftext_token, 
    max_test_steps, 
    verbose_steps=False # Controla si se imprimen los pasos individuales
):
    prompt = ["."] 
    
    # Lista para almacenar las probabilidades en cada paso de ESTA iteración
    prob_history = [] 

    for step in range(max_test_steps):
        current_prompt_str = "".join(prompt)
        
        if verbose_steps:
            print(f"\n--- Paso {step + 1} ---")
            print(f"Prompt actual: '{current_prompt_str}'")

        tokens_to_monitor = [im_end_token, endoftext_token]
        specific_probs = get_specific_token_probs(prompt, model, tokenizer, device, tokens_to_monitor)
        
        prob_history.append({
            'step': step + 1,
            'prompt': current_prompt_str,
            im_end_token: specific_probs.get(im_end_token, 0.0),
            endoftext_token: specific_probs.get(endoftext_token, 0.0)
        })

        if verbose_steps:
            for token, prob in specific_probs.items():
                print(f"  Probabilidad de '{token}': {prob:.8f}")

        # Lógica para muestrear el siguiente dígito
        numbers_for_sampling = ["0", "1", "2","3","4","5","6","7","8","9"]
        sampling_ids = []
        for num_str in numbers_for_sampling:
            encoded = tokenizer.encode(num_str, add_special_tokens=False)
            if len(encoded) == 1:
                sampling_ids.append(encoded[0])
        
        with torch.no_grad():
            output = model(torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(current_prompt_str))], device=device))
            logits = output.logits[:, -1, :]
            probs_all = torch.softmax(logits, dim=-1)[0]
        
        digit_probs = {tokenizer.decode([idx]).strip(): probs_all[idx].item() for idx in sampling_ids}
        total_digit_prob = sum(digit_probs.values())

        if total_digit_prob == 0:
            if verbose_steps:
                print("  No hay probabilidades para dígitos válidos. Deteniendo esta iteración.")
            break

        normalized_digit_probs = {word: p / total_digit_prob for word, p in digit_probs.items()}
        
        next_digit = np.random.choice(list(normalized_digit_probs.keys()), 
                                      p=list(normalized_digit_probs.values()))
        prompt.append(next_digit)
        if verbose_steps:
            print(f"  Dígito muestreado (para continuar la secuencia): '{next_digit}'")
            
    # Calcular los resultados de esta iteración
    im_end_wins = 0
    endoftext_wins = 0
    ties = 0
    
    for entry in prob_history:
        prob_im_end = entry[im_end_token]
        prob_endoftext = entry[endoftext_token]

        if prob_im_end > prob_endoftext:
            im_end_wins += 1
        elif prob_endoftext > prob_im_end:
            endoftext_wins += 1
        else:
            ties += 1
            
    return im_end_wins, endoftext_wins, ties, len(prob_history) # Retorna los resultados de esta iteración

# Función principal para ejecutar y promediar múltiples iteraciones
def run_averaged_prob_test():
    # --- Configuración de Dispositivo ---
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA está disponible. Usando GPU.")
    else:
        device = "cpu"
        print("CUDA NO está disponible. Usando CPU.")

    # --- Carga de Modelo y Tokenizer ---
    model_id = "Qwen/Qwen3-1.7B"
    print(f"\nCargando el tokenizer para el modelo: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    print("Tokenizer cargado exitosamente.")

    print(f"Cargando el modelo: {model_id} en {device} con torch_dtype=torch.bfloat16...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     pad_token_id=tokenizer.eos_token_id,
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).to(device)
        print("Modelo cargado exitosamente en la GPU (bfloat16).")
    except Exception as e:
        print(f"¡ERROR al cargar el modelo en bfloat16! {e}")
        print("Intentando cargar en CPU como fallback.")
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     pad_token_id=tokenizer.eos_token_id,
                                                     trust_remote_code=True).to("cpu")
        print("Modelo cargado en CPU.")

    # --- Tokens a Monitorear ---
    im_end_token = tokenizer.eos_token    # Esto es '<|im_end|>'
    endoftext_token = tokenizer.pad_token # Esto es '<|endoftext|>'
    
    print(f"\nTokens cuyas probabilidades serán monitoreadas: ['{im_end_token}', '{endoftext_token}']")

    # --- Parámetros para la prueba de iteraciones ---
    num_iterations = 10   # Cantidad de corridas de la prueba
    max_test_steps = 20   # Pasos de generación por cada corrida

    print(f"\n--- Iniciando {num_iterations} Iteraciones de Prueba (con {max_test_steps} pasos cada una) ---")

    total_im_end_wins = 0
    total_endoftext_wins = 0
    total_ties = 0
    overall_total_steps = 0 # Acumula el total de pasos efectivos de todas las iteraciones

    for i in range(num_iterations):
        print(f"\n***** Ejecutando Iteración {i + 1}/{num_iterations} *****")
        # verbose_steps=True solo para la primera iteración, si deseas ver el detalle
        # Si quieres ver el detalle de todas, cambia a True. Si no quieres ver ninguna, déjalo en False.
        im_end_wins, endoftext_wins, ties, current_steps = \
            _run_single_prob_test_iteration(
                model, tokenizer, device, 
                im_end_token, endoftext_token, 
                max_test_steps, 
                verbose_steps=(i == 0) # Imprime detalles solo para la primera iteración
            )
        
        total_im_end_wins += im_end_wins
        total_endoftext_wins += endoftext_wins
        total_ties += ties
        overall_total_steps += current_steps
        print(f"Iteración {i + 1} completada. Pasos analizados: {current_steps}")


    # --- Análisis Final Promediado ---
    print("\n\n--- Análisis de Probabilidades Promediado sobre "
          f"{num_iterations} Iteraciones ({overall_total_steps} pasos en total) ---")
    
    if overall_total_steps > 0:
        avg_percent_im_end_wins = (total_im_end_wins / overall_total_steps) * 100
        avg_percent_endoftext_wins = (total_endoftext_wins / overall_total_steps) * 100
        avg_percent_ties = (total_ties / overall_total_steps) * 100

        print(f"'{im_end_token}' (EOS oficial) fue mayor en un promedio del {avg_percent_im_end_wins:.2f}% de los pasos.")
        print(f"'{endoftext_token}' (Pad token) fue mayor en un promedio del {avg_percent_endoftext_wins:.2f}% de los pasos.")
        print(f"Empates en un promedio del {avg_percent_ties:.2f}% de los pasos.")

        if avg_percent_im_end_wins > avg_percent_endoftext_wins:
            print(f"\nConclusión General: En promedio, '{im_end_token}' mostró una probabilidad de ocurrencia consistentemente mayor, reforzando su rol como EOS del modelo.")
        elif avg_percent_endoftext_wins > avg_percent_im_end_wins:
            print(f"\nConclusión General: En promedio, '{endoftext_token}' mostró una probabilidad de ocurrencia consistentemente mayor.")
        else:
            print(f"\nConclusión General: En promedio, no hay una diferencia clara en la probabilidad de ocurrencia entre los tokens.")
    else:
        print("No se generaron pasos para analizar en ninguna de las iteraciones.")
    print("-------------------------------------------------------------------")


if __name__ == "__main__":
    run_averaged_prob_test()