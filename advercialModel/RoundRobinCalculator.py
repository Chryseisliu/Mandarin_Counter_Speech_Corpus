from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from advercialModel.JudgeLMinterface import JudgeLMEvaluator

# Initialize the evaluator once to improve performance
evaluator = JudgeLMEvaluator()


def prettyPrintResults(results: list[tuple[str, float]], response_length=20):
    # ANSI color codes
    green = "\033[92m"  # Green for tags
    red = "\033[91m"  # Red for score values
    reset = "\033[0m"  # Reset color to default

    for response, score in results:
        # Truncate and add "..." if longer than response_length
        if len(response) > response_length:
            formatted_response = response[:response_length - 3] + "..."
        else:
            # Pad with spaces to make it response_length characters
            formatted_response = response.ljust(response_length)

        # Print formatted output with colors
        print(f"{green}答案:{reset} {formatted_response} | {green}平均分:{reset} {red}{score}{reset}")


def roundRobinCalculator(hateSpeech: str, previousAnswers: list[str]) -> list[tuple[str, float]]:
    """
    Performs round-robin tournament scoring of answers using JudgeLM evaluator.

    Args:
        hateSpeech (str): The hate speech content to evaluate against
        previousAnswers (list): List of unique answer strings to evaluate

    Returns:
        list: List of [answer, average_score] pairs
    """

    # Initialize scores and counts dictionaries
    scores = {}
    counts = {}

    if len(previousAnswers) == 1:
        # Evaluate the single answer against an empty string
        answer = previousAnswers[0]
        question = {
            "question_body": f"Provide a brief counter-narrative in response to the following hate speech: {hateSpeech}. It is very important for the counter-narrative to be relevant to the hate speech.",
            "answer1_body": answer,
            "answer2_body": ""
        }
        normal_results = evaluator.get_model_answers([question], if_reverse_answers=False)[0]
        reversed_results = evaluator.get_model_answers([question], if_reverse_answers=True)[0]

        score1 = normal_results["output1"] + reversed_results["output2"]
        scores[answer] = score1
        counts[answer] = 2  # Scored twice
    else:
        answer_pairs = list(combinations(previousAnswers, 2))
        # For each pair, create a question format that JudgeLM expects
        for answer1, answer2 in answer_pairs:
            # Create question format
            question = {
                "question_body": f"Provide a brief counter-narrative in response to the following hate speech: {hateSpeech}. It is very important for the counter-narrative to be relevant to the hate speech.",
                "answer1_body": answer1,
                "answer2_body": answer2
            }

            # Get normal and reversed evaluations
            normal_results = evaluator.get_model_answers([question], if_reverse_answers=False)[0]
            reversed_results = evaluator.get_model_answers([question], if_reverse_answers=True)[0]

            # Add the scores from both evaluations
            score1 = normal_results["output1"] + reversed_results["output2"]
            score2 = normal_results["output2"] + reversed_results["output1"]

            # Update scores and counts
            scores[answer1] = scores.get(answer1, 0) + score1
            counts[answer1] = counts.get(answer1, 0) + 2
            scores[answer2] = scores.get(answer2, 0) + score2
            counts[answer2] = counts.get(answer2, 0) + 2

    # Compute average scores
    average_scores = []
    for answer in previousAnswers:
        avg_score = scores[answer] / counts[answer]
        average_scores.append((answer, avg_score))

    # Sort by average score in descending order
    average_scores.sort(key=lambda x: x[1], reverse=True)

    return average_scores


def process_csv_file(input_file: str, output_file: str):
    """
    Process a CSV file containing hate speech responses and evaluate them using round-robin tournament.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Group by ID to get all responses for each hate speech
    grouped = df.groupby('ID')

    # Initialize a flag to write header only once
    first_write = True

    # Process each group
    for id_val, group in tqdm(grouped, desc="Processing groups"):
        # Get hate speech and other static information
        hate_speech = group['HateSpeech'].iloc[0]
        language = group['Language'].iloc[0]
        KN = group['KN'].iloc[0]

        # Create DataFrame with 'Response', 'original_score'
        responses_df = group[['Response', 'Score', 'KN', 'Language']].copy()
        responses_df = responses_df.rename(columns={'Score': 'original_score'})

        # Remove duplicate responses
        unique_responses_df = responses_df.drop_duplicates(subset='Response')

        # Get list of unique responses
        unique_responses = unique_responses_df['Response'].tolist()

        # Ensure that previousAnswers doesn't have any duplicate answers
        previousAnswers = unique_responses  # These are unique already

        # Call roundRobinCalculator with unique responses
        scored_responses = roundRobinCalculator(hate_speech, previousAnswers)

        # Print results for debugging
        prettyPrintResults(scored_responses)

        # Convert to DataFrame
        comparative_scores_df = pd.DataFrame(scored_responses, columns=['Response', 'comparative_score'])

        # Merge comparative scores back to unique_responses_df
        merged_df = pd.merge(unique_responses_df, comparative_scores_df, on='Response', how='left')

        # Sort by comparative_score descending
        merged_df = merged_df.sort_values(by='comparative_score', ascending=False)

        # Select top 4 responses
        merged_df = merged_df.head(4)

        # If less than 4 unique responses, duplicate the responses until we have 4
        num_responses = len(merged_df)
        # If less than 4 unique responses, duplicate the responses until we have 4
        num_responses = len(merged_df)
        if num_responses < 4:
            if num_responses > 0:
                repeats_needed = int(np.ceil(4 / num_responses))
                merged_df = pd.concat([merged_df] * repeats_needed, ignore_index=True)
                merged_df = merged_df.iloc[:4]
            else:
                # If there are no responses, create dummy entries
                merged_df = pd.DataFrame({
                    'Response': [''] * 4,
                    'original_score': [0] * 4,
                    'comparative_score': [0] * 4
                })

        # Assign ranking from 1 to 4
        merged_df['ranking'] = range(1, 5)

        # Add 'ID', 'HateSpeech', 'KN', 'Language' columns
        merged_df['ID'] = id_val
        merged_df['HateSpeech'] = hate_speech
        merged_df['KN'] = KN
        merged_df['Language'] = language

        # Reorder columns to match output specification
        merged_df = merged_df[['ID', 'KN', 'Response', 'original_score', 'Language', 'HateSpeech', 'ranking', 'comparative_score']]

        # Write to CSV incrementally
        if first_write:
            merged_df.to_csv(output_file, mode='w', index=False, encoding='utf-8')
            first_write = False
        else:
            merged_df.to_csv(output_file, mode='a', index=False, header=False, encoding='utf-8')

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    input_file = "../processedOutput/combined_high_output_v1_EU.csv"
    output_file = "../optimizedResultsByLanguage/EU_Final_Results.csv"

    process_csv_file(input_file, output_file)