import random
import math
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from advercialModel.JudgeLMinterface import JudgeLMEvaluator
from advercialModel.LLMAnswerGenerator import LLMAnswerGenerator
from advercialModel.WordList import generate_frequent_word_list

# Instantiate the evaluator globally
evaluator_lock = threading.Lock()
evaluator = JudgeLMEvaluator(lock=evaluator_lock)
evaluator.initModelTokenizer()

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

def calculate_scores(words: List[str], question: str, answer: str, defaultScore=None) -> Dict[str, float]:
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
        if defaultScore is None:
            score1, score2 = score_answer2(answer1, answer2, question)
        else:
            score1, score2 = defaultScore, defaultScore
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
    Only include answers longer than " {Answer}".

    Parameters:
        answer (str): The original answer string.
        prompt (str): The prompt to generate new answers.

    Returns:
        List[str]: A list of filtered new answers.
    """
    # 生成格式化的提示
    formatted_prompt = llmAnswerGenerator.generate_formatted_prompt(prompt, answer)

    # 使用 LLM 生成隨機響應
    random_response = llmAnswerGenerator.generate_random_response(formatted_prompt=formatted_prompt)

    # 解析生成的答案
    parsed_answers_random = llmAnswerGenerator.parse_answers(random_response)

    min_length = len("<ANSWER>")*2
    max_length = 100

    filtered_answers = [s for s in parsed_answers_random if min_length < len(s) < max_length]

    if not filtered_answers and len(parsed_answers_random) > 0:
        filtered_answers = parsed_answers_random

    return filtered_answers

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

    resultsNormal = evaluator.threaded_get_model_answers(questions_test)[0]
    resultsInverted = evaluator.threaded_get_model_answers(questions_test, if_reverse_answers=True)[0]
    try:
        averageOutput1 = (resultsNormal["output1"] + resultsInverted["output1"]) / 2
        averageOutput2 = (resultsNormal["output2"] + resultsInverted["output2"]) / 2
    except Exception:
        averageOutput1, averageOutput2 = 0, 0
    return averageOutput1, averageOutput2


def generateExponentialWeightedScores(answers: Dict[str, float], exponentBase: float) -> Dict[str, float]:
    """
    Generate exponential weighted scores.
    Caps the exponentiation at exponentBase^10 to prevent overflow.

    Args:
        answers (Dict[str, float]): A dictionary where keys are answer identifiers and values are their scores.
        exponentBase (float): The base to be exponentiated with the scores.

    Returns:
        Dict[str, float]: A dictionary with the same keys as `answers` and the exponentiated scores as values.
    """
    max_exponent = 10  # Maximum exponent to prevent overflow
    weighted_scores = {}

    for ans, score in answers.items():
        # Clamp the score to the maximum exponent
        clamped_score = min(score, max_exponent)

        try:
            weighted_score = exponentBase ** clamped_score
        except OverflowError:
            # If overflow occurs, set the score to exponentBase^10
            weighted_score = exponentBase ** max_exponent

        weighted_scores[ans] = weighted_score

    return weighted_scores

def simulatedAnnealingSearchStep(
    args
) -> Dict[str, float]:
    """
    Perform a single search step in the simulated annealing process.
    """
    # Unpack arguments
    (
        question,
        answer,
        wordList,
        sampleSize,
        sampleSizeScoreRatio,
        numAnswersToGenerate,
        scoreWeighting,
        generateAIAnswers,
        defaultScore
    ) = args

    # Step 1: Sample words
    sampled_words = sample_words(wordList, sampleSize)
    sampled_words.append("")

    # Step 2: Calculate scores of each answer
    answer_scores = calculate_scores(sampled_words, question, answer, defaultScore)

    # Step 3: Weight scores with exponential function
    weighted_scores = generateExponentialWeightedScores(answer_scores, scoreWeighting)

    # Step 4: Select answer-score pairs probabilistically
    selected_answers = probabilistic_selection(weighted_scores, numAnswersToGenerate)

    if not generateAIAnswers:
        return selected_answers

    # Step 5: Generate new answers concurrently using threading
    generated_answers = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ans = {
            executor.submit(generate_new_answers, ans, prompt=question): ans
            for ans in selected_answers
        }

        for future in as_completed(future_to_ans):
            ans = future_to_ans[future]
            try:
                new_answers = future.result()
                filtered=[]
                for answer in list(set(new_answers)):
                    if len(answer)<150 and len(answer)>len(" <ANSWER> "):
                        filtered.append(answer)
                generated_answers.extend(filtered)
            except Exception as exc:
                print(f'Answer generation for "{ans}" generated an exception: {exc}')

    # Step 6: Score the new answers
    answer_scores_new = calculate_scores(generated_answers, question, "")

    # Step 7: Weight the new scores exponentially
    weighted_scores_new = generateExponentialWeightedScores(answer_scores_new, scoreWeighting)

    # Step 8: Combine and select final answers
    combined_dict = {**selected_answers, **weighted_scores_new}
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
    targetScore: float = 8.9,
    startingPossibleAffixes: Dict[str, float] = {"": 0.1},
    maxIterations: int = 31,
    sampleSize: int = 20,
    numAnswersToGenerateForEachLoop: int = 4,
    generateAiAnswersPeriod: int = 5,
    threads = None  # Number of threads
) -> Tuple[str, Dict[str, float], str]:
    """
    Perform simulated annealing to find the best counter-speech.
    """
    currentPossibleAffixes = startingPossibleAffixes
    scoreWeighting = sampleSize

    for i in range(maxIterations):
        newPossibleAffixes = currentPossibleAffixes.copy()

        tasks = []
        for ans, score in currentPossibleAffixes.items():
            if score >= scoreWeighting ** targetScore:
                return ans, currentPossibleAffixes, "Success"

            aiAnswers = not (i % generateAiAnswersPeriod)

            current_ans = answer if i == 0 else ans
            current_score = 1 if i == 0 else score

            if current_score<1:
                current_score=1
            current_score=math.log(current_score, scoreWeighting)-0.5
            # Prepare arguments as a tuple
            args = (
                question,
                current_ans,
                wordList,
                sampleSize,
                1,  # sampleSizeScoreRatio (assuming 1 as default)
                numAnswersToGenerateForEachLoop,
                scoreWeighting,
                aiAnswers,
                current_score
            )
            tasks.append(args)

        # Execute simulatedAnnealingSearchStep in parallel using threads
        results = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_task = {executor.submit(simulatedAnnealingSearchStep, task): task for task in tasks}
            for future in as_completed(future_to_task):
                try:
                    step_results = future.result()
                    results.append(step_results)
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        # Update newPossibleAffixes with results
        for step_results in results:
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
    hateSpeech: str,
    sampleSize: int = 10,
    iterations: int = 20,
    numAICallsPerAILoop: int = 5,
    generateAiAnswersPeriod: int = 2,
    threads = None  # Number of threads
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
        threads (int): Number of threads to use.

    Returns:
        List[List]: A list of lists containing [ID, KN, response, score, language, hateSpeech].
    """
    prompt = (
        "Provide a brief counter-narrative in response to the following hate speech: "
        + hateSpeech
        + ". It is very important for the counter-narrative to be relevant to the hate speech."
    )
    word_list = generate_frequent_word_list(language="chinese") + sample_words_from_text(hateSpeech) * 500

    result = simulatedAnnealing(
        question=prompt,
        answer=hateSpeech,
        wordList=word_list,
        sampleSize=sampleSize,
        maxIterations=iterations,
        numAnswersToGenerateForEachLoop=numAICallsPerAILoop,
        generateAiAnswersPeriod=generateAiAnswersPeriod,
        threads=threads
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
        [response, score, hateSpeech]
        for response, score in normal_scores.items()
    ]

    return counter_speech_results

if __name__ == "__main__":
    hate_speech = ("现在广大男性同胞清醒了，反击了，然后她们就开始在这讲道理了，"+
                   "当时女权癌大肆攻击男性的时候，声称杀光男性的时候，她们人呢？"+
                   "怎么没见她们出来跟女权癌解释现实中男人其实也很好，不要伤及无辜？？？")
    result = findBestCounterSpeech(hate_speech, iterations=40, threads=100)
    print(result)
