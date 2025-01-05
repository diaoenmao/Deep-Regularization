def choose_score(score_function, score_name, model):
    score_function = score_function(model)
    if score_name == 'wanda':
        return score_function.compute_wanda_scores()
    elif score_name == 'lora':
        return score_function.compute_lora_scores()
    elif score_name == 'magnitude':
        return score_function.compute_magnitude_scores()
    else:
        raise ValueError(f"Invalid score name: {score_name}")