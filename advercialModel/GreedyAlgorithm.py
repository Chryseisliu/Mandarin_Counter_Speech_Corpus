import random
import math
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from advercialModel.JudgeLMinterface import JudgeLMEvaluator
from advercialModel.LLMAnswerGenerator import LLMAnswerGenerator
from advercialModel.WordList import generate_frequent_word_list

evaluator = JudgeLMEvaluator()
llmAnswerGenerator = LLMAnswerGenerator()


def sample_words_from_text(text: str, sample_size: int = 1000000) -> List[str]:
    """
    Sample a specified number of words from the given string 'text' list.
    """
    words = text.split()
    unique_words = list(set(words))
    sample = random.sample(unique_words, min(sample_size, len(unique_words)))
    return sample


def sample_words(wordList: List[str], sample_size: int) -> List[str]:
    """
    Sample a specified number of words from the word list.
    """
    return random.sample(wordList, min(sample_size, len(wordList)))


def calculate_scores(words: List[str], question: str, answer: str) -> Dict[str, float]:
    """
    Calculate scores for each word pair based on the question and current answer.
    """
    scores = {}

    # If the list has an odd number of words, duplicate the first word at the end
    if len(words) % 2 != 0:
        words.append(words[0])

    # Iterate over words in pairs
    for i in range(0, len(words), 2):
        word1 = words[i]
        word2 = words[i + 1]

        answer1 = answer + " " + word1
        answer2 = answer + " " + word2
        score1, score2 = score_answer2(answer1, answer2, question)
        scores[answer1] = score1
        scores[answer2] = score2

    return scores


def convert_to_probabilities(weighted_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Convert weighted scores into probabilities.
    """
    total = sum(weighted_scores.values())
    if total == 0:
        return {k: 1 / len(weighted_scores) for k in weighted_scores}
    return {k: v / total for k, v in weighted_scores.items()}


def probabilistic_selection(scores: Dict[str, float], num_selections: int) -> Dict[str, float]:
    """
    Select a specified number of unique items based on their probabilities,
    ensuring that one of the top items is always selected first.
    If num_selections is greater than the number of items, it selects all items.
    """
    num_selections = min(num_selections, len(scores))
    items = list(scores.items())
    weights = list(scores.values())
    selected = []

    # Select the top item first
    max_index = weights.index(max(weights))
    top_item = items.pop(max_index)
    weights.pop(max_index)
    selected.append(top_item)

    remaining_selections = num_selections - 1

    try:
        for _ in range(remaining_selections):
            chosen = random.choices(items, weights=weights, k=1)[0]
            selected.append(chosen)
            index = items.index(chosen)
            items.pop(index)
            weights.pop(index)
    except Exception as e:
        print(e)

    return dict(selected)


def generate_new_answers(answer: str, prompt: str) -> List[str]:
    """
    Generate two new possible answers using a Language Model (LLM).
    """
    formatted_prompt = llmAnswerGenerator.generate_formatted_prompt(prompt, answer)
    random_response = llmAnswerGenerator.generate_random_response(formatted_prompt=formatted_prompt)
    parsed_answers_random = llmAnswerGenerator.parse_answers(random_response)
    return parsed_answers_random


def score_answer2(answer1: str, answer2: str, question: str) -> Tuple[float, float]:
    """
    Score a single answer based on the question.
    """
    questions_test = [
        {
            "question_body": question,
            "answer1_body": answer1,
            "answer2_body": answer2,
        },
    ]

    resultsNormal = evaluator.get_model_answers(questions_test)[0]
    resultsInverted = evaluator.get_model_answers(questions_test, if_reverse_answers=True)[0]
    try:
        averageOutput1 = (resultsNormal["output1"] + resultsInverted["output1"]) / 2
        averageOutput2 = (resultsNormal["output2"] + resultsInverted["output2"]) / 2
    except Exception:
        averageOutput1, averageOutput2 = 0, 0
    return averageOutput1, averageOutput2


def generateExponentialWeightedScores(answers: Dict[str, float], exponentBase: float) -> Dict[str, float]:
    """
    Generate exponential weighted scores.
    """
    weighted_scores = {ans: exponentBase ** score for ans, score in answers.items()}
    return weighted_scores


def simulatedAnnealingSearchStep(
    question: str,
    answer: str,
    wordList: List[str],
    sampleSize: int = 10,
    sampleSizeScoreRatio: int = 1,
    numAnswersToGenerate: int = 3,
    scoreWeighting: float = 100,
    generateAIAnswers: bool = False,
) -> Dict[str, float]:
    """
    Perform a single search step in the simulated annealing process.
    """
    # Step 1: Sample words
    sampled_words = sample_words(wordList, sampleSize)
    sampled_words.append("")

    # Step 2: Calculate scores of each answer
    answer_scores = calculate_scores(sampled_words, question, answer)

    # Step 3: Weight scores with exponential function
    weighted_scores = generateExponentialWeightedScores(answer_scores, scoreWeighting)

    # Step 4: Select answer-score pairs probabilistically
    selected_answers = probabilistic_selection(weighted_scores, numAnswersToGenerate)

    if not generateAIAnswers:
        return selected_answers

    # Step 5: Generate new answers concurrently using threading
    generated_answers = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ans = {
            executor.submit(generate_new_answers, ans, prompt=question): ans
            for ans in selected_answers
        }
        for future in as_completed(future_to_ans):
            ans = future_to_ans[future]
            try:
                new_answers = future.result()
                generated_answers.extend(new_answers)
            except Exception as exc:
                print(f'Answer generation for "{ans}" generated an exception: {exc}')

    # Step 6: Score the new answers
    generated_answers_scores = calculate_scores(generated_answers, question, "")

    # Step 7: Weight the new scores exponentially
    generated_answers_scores_weighted = generateExponentialWeightedScores(generated_answers_scores, scoreWeighting)

    # Step 8: Combine and select final answers
    combined_dict = {**selected_answers, **generated_answers_scores_weighted}
    final_selected_answers = probabilistic_selection(combined_dict, numAnswersToGenerate)

    return final_selected_answers


def print_words(exponential_scores: Dict[str, float], base: float) -> None:
    """
    Print answers with their exponential scores converted back to regular scores.
    Displays only the first 20 characters of each answer followed by ellipses if longer than 20 characters.
    Ensures the commas align by using consistent formatting.
    """
    number_of_letters = 20
    max_display_length = number_of_letters + 3
    for answer, exp_score in exponential_scores.items():
        regular_score = math.log(exp_score, base) if exp_score > 0 else float('-inf')
        display_answer = (answer[:number_of_letters] + "...") if len(answer) > number_of_letters else answer
        formatted_answer = f"{display_answer:<{max_display_length}}"
        print(f"Answer: {formatted_answer}, Score: {regular_score:.2f}")


def simulatedAnnealing(
    question: str,
    answer: str,
    wordList: List[str],
    targetScore: float = 10,
    startingPossibleAffixes: Dict[str, float] = {"": 0.1},
    maxIterations: int = 31,
    sampleSize: int = 20,
    numAnswersToGenerateForEachLoop: int = 4,
    generateAiAnswersPeriod: int = 5,
) -> Tuple[str, Dict[str, float], str]:
    """
    Perform simulated annealing to find the best counter-speech.
    """
    currentPossibleAffixes = startingPossibleAffixes
    scoreWeighting = sampleSize

    for i in range(maxIterations):
        newPossibleAffixes = currentPossibleAffixes.copy()

        for ans, score in currentPossibleAffixes.items():
            if score >= scoreWeighting ** targetScore:
                return ans, currentPossibleAffixes, "Success"

            aiAnswers: bool = not (i % generateAiAnswersPeriod)

            current_ans = answer if i == 0 else ans
            step_results = simulatedAnnealingSearchStep(
                question=question,
                answer=current_ans,
                wordList=wordList,
                sampleSize=sampleSize,
                scoreWeighting=scoreWeighting,
                numAnswersToGenerate=numAnswersToGenerateForEachLoop,
                generateAIAnswers=aiAnswers,
            )
            newPossibleAffixes.update(step_results)

            print(newPossibleAffixes)
            print_words(newPossibleAffixes, scoreWeighting)

        number_to_save = numAnswersToGenerateForEachLoop if i + 1 < maxIterations else 20
        currentPossibleAffixes = probabilistic_selection(newPossibleAffixes, number_to_save)
        print(f"FINISHED ITERATION #{i}")
        print_words(currentPossibleAffixes, scoreWeighting)
        print("\n--------------------------------\n\n")

    best_answer = max(currentPossibleAffixes, key=currentPossibleAffixes.get)
    return best_answer, currentPossibleAffixes, "Failure"


def findBestCounterSpeech(
    ID: int,
    hateSpeech: str,
    KN: str,
    language: str,
    sampleSize: int = 10,
    iterations: int = 20,
    numAICallsPerAILoop: int = 5,
    generateAiAnswersPeriod: int = 5,
) -> List[List]:
    """
    Find the best counter-speech response based on the provided hate speech and language.

    Parameters:
        ID (int): Identifier for the input.
        hateSpeech (str): The hate speech input.
        KN (str): Keyword or key phrase related to the counter-speech.
        language (str): Language for generating the frequent word list.
        sampleSize (int): Number of words to sample for generating responses.
        iterations (int): Number of iterations for simulated annealing.
        numAICallsPerAILoop (int): Number of AI calls per loop.
        generateAiAnswersPeriod (int): Period for generating AI answers.

    Returns:
        List[List]: A list of lists containing [ID, KN, response, score, language, hateSpeech].
    """
    prompt = (
        "Provide a brief counter-narrative in response to the following hate speech: "
        + hateSpeech
        + ". It is very important for the counter-narrative to be relevant to the hate speech."
    )
    word_list = generate_frequent_word_list(language=language) + sample_words_from_text(hateSpeech) * 500
    result = simulatedAnnealing(
        question=prompt,
        answer=hateSpeech,
        wordList=word_list,
        sampleSize=sampleSize,
        maxIterations=iterations,
        numAnswersToGenerateForEachLoop=numAICallsPerAILoop,
        generateAiAnswersPeriod=generateAiAnswersPeriod,
    )

    best_answer, scores_dict, status = result
    exponent_base = sampleSize

    # Convert exponential scores back to normal scores
    normal_scores = {}
    for answer, exp_score in scores_dict.items():
        try:
            regular_score = math.log(exp_score, exponent_base) if exp_score > 0 else float('-inf')
        except:
            regular_score = float('-inf')
        normal_scores[answer] = regular_score

    # Create a list of lists in the format [ID, KN, response, score, language, hateSpeech]
    counter_speech_results = [
        [ID, KN, response, score, language, hateSpeech]
        for response, score in normal_scores.items()
    ]

    return counter_speech_results


if __name__ == "__main__":
    hate_speech = "Dovremmo espellere tutti gli omosessuali."
    sentence_example = "Quand'è che l'amore prospererà e l'odio inizierà a dissiparsi? Non solo rispetterò i miei compagni LGBT+, ma promuoverò i loro diritti."
    result = findBestCounterSpeech(69, hate_speech, sentence_example, "english",iterations=2)
    print(result)
