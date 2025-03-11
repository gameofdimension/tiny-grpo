from data import extract_answer_from_model_output
from eval import extract_single_number


def correctness_reward(prompts, completions, answer, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.

    Args:
        prompts (list): List of input prompts.
        completions (list): List of model completions, each containing content.
        answer (list): List of expected answers.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of numerical rewards for each completion.

    Explanation:
        1. Extracts the content from each completion.
        2. Extracts the answer portion from each response using extract_answer_from_model_output.
        3. Assigns rewards based on matching criteria:
           - 2.0 points for an exact match
           - 1.5 points for numeric equivalence (when values match but format differs)
           - 0.0 points for incorrect answers
        4. Tracks completion lengths for analysis.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        if r == a:  # Exact match case
            rewards.append(2.0)
        else:
            # Try numeric equivalence
            r_num = extract_single_number(str(r))
            a_num = extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
    # Log completion lengths
    completion_lengths = [len(response.split()) for response in responses]
    return rewards


def format_reward(completions, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.

    Args:
        completions (list): List of model completions, each containing content.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of format compliance scores for each completion.

    Explanation:
        1. Extracts the content from each completion.
        2. Evaluates format compliance by checking for required XML tags:
           - 0.2 points for each tag present (<reasoning>, </reasoning>, <answer>, </answer>)
           - Maximum score of 0.8 for perfect format compliance
        3. Stores and returns the format compliance scores.
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    format_scores = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2
        if "<answer>" in response:
            score += 0.2
        if "</answer>" in response:
            score += 0.2
        rewards.append(score)
        format_scores.append(score)
    return rewards


def combined_reward(prompts, completions, answer):
    """
    Combines correctness and format rewards.

    Args:
        prompts (list[str]): List of prompt texts
        completions (list[list[dict]]): List of completion dictionaries
        answer (list[str]): List of expected answers

    Returns:
        list[float]: Combined rewards for each prompt-completion pair

    Explanation:
        1. Calculates separate rewards for correctness and format compliance.
        2. Combines the rewards with the following weights:
           - Correctness score range: 0.0 to 2.0
           - Format score range: 0.0 to 0.8
           - Total possible range: 0.0 to 2.8
        3. Returns the combined reward for each example.
    """
    # Get individual rewards
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)

    # Combine rewards - correctness is weighted more heavily
    combined_rewards = []
    for c_score, f_score in zip(correctness_scores, format_scores):
        # Correctness score range: 0.0 to 2.0
        # Format score range: 0.0 to 0.8
        # Total range: 0.0 to 2.8
        combined_rewards.append(c_score + f_score)

    return combined_rewards
