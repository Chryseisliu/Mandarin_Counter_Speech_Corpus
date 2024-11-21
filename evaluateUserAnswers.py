import pandas as pd
from itertools import combinations
from typing import List, Tuple


from advercialModel.JudgeLMinterface import JudgeLMEvaluator

class EvaluatorClass:
    def __init__(self):
        self.evaluator = JudgeLMEvaluator()

    def roundRobinCalculator(self, hateSpeech: str, previousAnswers: List[str]) -> List[Tuple[str, float]]:
        """
        Performs round-robin tournament scoring of answers using JudgeLM evaluator.

        Args:
            hateSpeech (str): The hate speech content to evaluate against.
            previousAnswers (List[str]): List of unique answer strings to evaluate.

        Returns:
            List[Tuple[str, float]]: List of (answer, average_score) pairs.
        """
        # Initialize scores and counts dictionaries
        scores = {}
        counts = {}

        if len(previousAnswers) == 1:
            # Evaluate the single answer against an empty string
            answer = previousAnswers[0]
            question = {
                "question_body": f"Provide a brief counter-narrative in response to the following content: {hateSpeech}. It is very important for the counter-narrative to be relevant.",
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
                question = {
                    "question_body": f"Provide a brief counter-narrative in response to the following content: {hateSpeech}. It is very important for the counter-narrative to be relevant.",
                    "answer1_body": answer1,
                    "answer2_body": answer2
                }
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


def read_and_filter_data(input_file: str) -> pd.DataFrame:
    """
    Reads the .xlsx file and filters rows where hate score is 1 and user entered response is not null.

    Args:
        input_file (str): Path to the input .xlsx file.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = pd.read_excel(input_file)
    # Rename columns for consistency
    df = df.rename(columns={
        'hatespeech': 'hateSpeech',
        'userEnteredResponse': 'userEnteredResponse',
        'hateScore': 'hateScore'
    })
    # Filter rows
    df_filtered = df[(df['hateScore'] == 1) & (df['userEnteredResponse'].notnull())]
    return df_filtered


def group_responses(df: pd.DataFrame) -> dict:
    """
    Groups responses by hate speech.

    Args:
        df (pd.DataFrame): Filtered DataFrame.

    Returns:
        dict: Dictionary with hate speech as keys and list of responses as values.
    """
    grouped = df.groupby('hateSpeech')
    response_dict = {}
    for hate_speech, group in grouped:
        responses = group[['userEnteredResponse', 'generatedResponse1', 'generatedResponse2', 'generatedResponse3',
                           'generatedResponse4']]
        # Flatten the responses into a list and remove duplicates
        response_list = responses.values.flatten().tolist()
        unique_responses = list(set(filter(lambda x: pd.notnull(x), response_list)))
        response_dict[hate_speech] = unique_responses
    return response_dict


def rank_responses(evaluator: EvaluatorClass, hate_speech: str, responses: List[str], user_response: str) -> Tuple[
    List[str], int]:
    """
    Ranks the responses using roundRobinCalculator.

    Args:
        evaluator (EvaluatorClass): Instance of EvaluatorClass.
        hate_speech (str): The hate speech content.
        responses (List[str]): List of responses to rank.
        user_response (str): The user's response.

    Returns:
        Tuple[List[str], int]: Responses sorted from highest to lowest score and user response rank.
    """
    # Check if user response is in generated responses

    try:
        scored_responses = evaluator.roundRobinCalculator(hate_speech, responses)
        # Extract only the responses in sorted order
        ranked_responses = [resp for resp, score in scored_responses]
        # Determine user response rank
        if user_response in ranked_responses:
            user_rank = ranked_responses.index(user_response) + 1
        else:
            # Include user response in ranking
            responses_with_user = responses + [user_response]
            # Remove duplicates
            responses_with_user = list(set(responses_with_user))
            # Re-run ranking including user response
            scored_responses = evaluator.roundRobinCalculator(hate_speech, responses_with_user)
            ranked_responses = [resp for resp, score in scored_responses]
            user_rank = ranked_responses.index(user_response) + 1 if user_response in ranked_responses else None
        return ranked_responses, user_rank
    except Exception as e:
        print(f"Error ranking responses for hate speech '{hate_speech}': {e}")
        return responses, None  # Return original order if ranking fails


def create_output_dataframe(hate_speech: str, user_response: str, user_rank: int,
                            ranked_responses: List[str]) -> pd.DataFrame:
    """
    Creates a DataFrame for the output CSV.

    Args:
        hate_speech (str): The hate speech content.
        user_response (str): The user's response.
        user_rank (int): The rank of the user's response.
        ranked_responses (List[str]): Ranked list of responses.

    Returns:
        pd.DataFrame: DataFrame with the required columns.
    """
    data = {
        'hateSpeech': [hate_speech],
        'userEnteredResponse': [user_response],
        'userEnteredResponseRank': [user_rank],
        'GeneratedResponse1': [ranked_responses[0] if len(ranked_responses) > 0 else None],
        'GeneratedResponse2': [ranked_responses[1] if len(ranked_responses) > 1 else None],
        'GeneratedResponse3': [ranked_responses[2] if len(ranked_responses) > 2 else None],
        'GeneratedResponse4': [ranked_responses[3] if len(ranked_responses) > 3 else None],
    }
    return pd.DataFrame(data)


def rank_scored_answers(input_file: str, output_file: str):
    """
    Main function to rank scored answers and output a CSV file.

    Args:
        input_file (str): Path to the input .xlsx file.
        output_file (str): Path to the output .csv file.
    """
    df_filtered = read_and_filter_data(input_file)
    response_dict = group_responses(df_filtered)
    evaluator = EvaluatorClass()
    output_dfs = []

    for hate_speech, responses in response_dict.items():
        print(f"Processing hate speech: {hate_speech}")
        user_response = df_filtered[df_filtered['hateSpeech'] == hate_speech]['userEnteredResponse'].iloc[0]
        # Limit to a maximum of 5 responses for performance


        if len(responses) < 5 and user_response != '':
            # Get the rank of the user response in the generated responses
            user_rank = responses.index(user_response) + 1
            ranked_responses = responses
            print(f"User response is in generated responses. Assigned rank {user_rank}.")
        else:
            # Include user response in the list if not already present
            if user_response not in responses and user_response != '':
                responses_with_user = responses + [user_response]
            else:
                responses_with_user = responses
            # Remove duplicates
            responses_with_user = list(set(responses_with_user))
            # Rank the responses
            ranked_responses, user_rank = rank_responses(evaluator, hate_speech, responses_with_user, user_response)
        output_df = create_output_dataframe(hate_speech, user_response, user_rank, ranked_responses)
        output_dfs.append(output_df)

    # Concatenate all DataFrames and save to CSV
    final_df = pd.concat(output_dfs, ignore_index=True)
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    input_file = "./humanProcessed/4-5_scored.xlsx"
    output_file = "./comparedHumanAnswers/4-5_scored.csv"
    rank_scored_answers(input_file, output_file)
