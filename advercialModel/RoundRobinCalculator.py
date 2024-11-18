from itertools import combinations
import numpy as np
import pandas as pd
from tqdm import tqdm
from advercialModel.JudgeLMinterface import JudgeLMEvaluator

class EvaluatorClass:
    def __init__(self):
        self.evaluator = JudgeLMEvaluator()
        # Any other initialization

    def prettyPrintResults(self, results: list[tuple[str, float]], response_length=20):
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

    def roundRobinCalculator(self, hateSpeech: str, previousAnswers: list[str]) -> list[tuple[str, float]]:
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
            normal_results = self.evaluator.get_model_answers([question], if_reverse_answers=False)[0]
            reversed_results = self.evaluator.get_model_answers([question], if_reverse_answers=True)[0]

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
                if len(question["question_body"]) > 260:
                    question["question_body"] = question["question_body"][:260]
                if len(question["answer1_body"] + question["answer2_body"]) > 260:
                    question["answer1_body"] = question["answer1_body"][:260 - len(question["answer2_body"])]
                # Get normal and reversed evaluations
                normal_results = self.evaluator.get_model_answers([question], if_reverse_answers=False)[0]
                reversed_results = self.evaluator.get_model_answers([question], if_reverse_answers=True)[0]

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

    def process_csv_files(self, input_files: list[str], output_file: str):
        """
        Process multiple CSV files containing hate speech responses and evaluate them using round-robin tournament.

        Args:
            input_files (list[str]): List of paths to input CSV files
            output_file (str): Path to output CSV file
        """
        # Read all the CSV files and concatenate them
        dfs = [pd.read_csv(file) for file in input_files]
        df = pd.concat(dfs, ignore_index=True)

        # Ensure the columns are 'response', 'score', 'hateSpeech'
        # Adjust column names if necessary
        df = df.rename(columns={
            'Response': 'response',
            'Score': 'score',
            'HateSpeech': 'hateSpeech'
        })

        # Check if required columns are present
        required_columns = {'response', 'score', 'hateSpeech'}
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        # Group by 'hateSpeech' to get all responses for each hate speech
        grouped = df.groupby('hateSpeech')

        # Initialize a flag to write header only once
        first_write = True

        # Process each group
        for hate_speech, group in tqdm(grouped, desc="Processing groups"):
            print("Processing hate speech:", hate_speech)
            # Create DataFrame with 'response', 'score'
            responses_df = group[['response', 'score']].copy()

            # Remove duplicate responses
            unique_responses_df = responses_df.drop_duplicates(subset='response')

            # Get list of unique responses
            unique_responses = unique_responses_df['response'].tolist()

            # Ensure that previousAnswers doesn't have any duplicate answers
            previousAnswers = unique_responses  # These are unique already

            if len(previousAnswers) > 5:
                previousAnswers = previousAnswers[:5]
            # Call roundRobinCalculator with unique responses
            try:
                scored_responses = self.roundRobinCalculator(hate_speech, previousAnswers)
            except Exception as e:
                print(f"Error occurred during roundRobinCalculator: {e}")
                print("Deleting evaluator and recreating...")
                del self.evaluator
                self.evaluator = JudgeLMEvaluator()
                try:
                    scored_responses = self.roundRobinCalculator(hate_speech, previousAnswers)
                except Exception as e:
                    print(f"Error occurred again during roundRobinCalculator: {e}")
                    print("Skipping this group.")
                    continue

            # Convert to DataFrame
            comparative_scores_df = pd.DataFrame(scored_responses, columns=['response', 'rrScore'])

            # Merge comparative scores back to unique_responses_df
            merged_df = pd.merge(unique_responses_df, comparative_scores_df, on='response', how='left')

            # Sort by rrScore descending
            merged_df = merged_df.sort_values(by='rrScore', ascending=False)

            # Select top 4 responses
            merged_df = merged_df.head(4)

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
                        'response': [''] * 4,
                        'score': [0] * 4,
                        'rrScore': [0] * 4
                    })

            # Add 'hateSpeech' column
            merged_df['hateSpeech'] = hate_speech

            # Reorder columns to match output specification
            merged_df = merged_df[['hateSpeech', 'response', 'score', 'rrScore']]

            # Write to CSV incrementally
            if first_write:
                merged_df.to_csv(output_file, mode='w', index=False, encoding='utf-8')
                first_write = False
            else:
                merged_df.to_csv(output_file, mode='a', index=False, header=False, encoding='utf-8')

        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    input_files = [
        "../split_output/output_filtered_hate_score_results_chunk_13.csv",
        "../split_output/output_filtered_hate_score_results_chunk_14.csv",
    ]
    output_file = "rankedOutputToBeGraded/13-14.csv"

    evaluator_class = EvaluatorClass()
    evaluator_class.process_csv_files(input_files, output_file)

    input_files = [
        "../split_output/output_filtered_hate_score_results_chunk_3.csv",
        "../split_output/output_filtered_hate_score_results_chunk_6.csv",
    ]
    output_file = "rankedOutputToBeGraded/3-6.csv"
    evaluator_class.process_csv_files(input_files, output_file)


    exit()
