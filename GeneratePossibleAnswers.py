import os
import json
import math
import nltk
from collections import Counter
from typing import List
import jieba
import pandas as pd
from tqdm import tqdm

from advercialModel.GreedyAlgorithm import findBestCounterSpeech


def GenerateAnswersFromCSV(input_csv_path: str, output_csv_path: str, sampleSize: int = 25, iterations: int = 13, numAICallsPerAILoop: int = 5):
    """
    Generate counter-speech responses from an input CSV file containing hate speech.

    Args:
        input_csv_path (str): Path to the input CSV file with columns 'text' and 'hate_score'.
        output_csv_path (str): Path to save the output CSV file with columns 'response', 'score', 'hateSpeech'.
        sampleSize (int): Number of words to sample for generating responses.
        iterations (int): Number of iterations for simulated annealing.
        numAICallsPerAILoop (int): Number of AI calls per loop.
    """
    # Read input data
    try:
        input_data = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Input CSV file not found at: {input_csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Input CSV file at {input_csv_path} is empty.")
        return
    except Exception as e:
        print(f"An error occurred while reading the input CSV: {e}")
        return

    # Check if required columns exist
    if not {'text', 'hate_score'}.issubset(input_data.columns):
        print("Input CSV must contain 'text' and 'hate_score' columns.")
        return

    # Initialize output file with headers
    output_headers = ["response", "score", "hateSpeech"]
    pd.DataFrame(columns=output_headers).to_csv(output_csv_path, index=False)

    # Iterate over each row and generate counter-speech with tqdm for progress tracking
    for _, row in tqdm(input_data.iterrows(), total=len(input_data), desc="Generating counter-speech"):
        hateSpeech = row['text']
        hate_score = row['hate_score']

        # Generate counter-speech for each entry
        responses = findBestCounterSpeech(
            hateSpeech=hateSpeech,
            sampleSize=sampleSize,
            iterations=iterations,
            numAICallsPerAILoop=numAICallsPerAILoop,
            generateAiAnswersPeriod=2
        )

        # Prepare data for output
        output_rows = []
        for response, score, hs in responses:
            output_rows.append({
                "response": response,
                "score": score,
                "hateSpeech": hs
            })

        # Append responses to the output CSV file
        if output_rows:
            output_df = pd.DataFrame(output_rows)
            output_df.to_csv(output_csv_path, mode='a', header=False, index=False)
        else:
            print(f"No valid responses generated for hate speech: {hateSpeech}")

    print(f"Counter-speech generation completed. Results saved to {output_csv_path}")


if __name__ == "__main__":
    # Define paths for testing

    input_csv_path = "split_input/filtered_hate_score_results_chunk_3.csv"
    output_csv_path = "split_output/output_filtered_hate_score_results_chunk_3.csv"

    # Run the function on the test data
    GenerateAnswersFromCSV(
        input_csv_path=input_csv_path,
        output_csv_path=output_csv_path,
        sampleSize=20,
        iterations=16,
        numAICallsPerAILoop=7
    )
