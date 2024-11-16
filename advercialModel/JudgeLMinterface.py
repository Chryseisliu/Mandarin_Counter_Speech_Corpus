import argparse
import json
import os
import time
from pprint import pprint

import shortuuid
import torch
from tqdm import tqdm
from pathlib import Path
import sys
import threading  # 使用 threading 代替 multiprocessing
from transformers import AutoTokenizer

from advercialModel.judgelm.llm_judge.common import (
    load_questions,
    reorg_answer_file,
    conv_judge_pair,
    conv_judge_pair_w_reference,
    KeywordsStoppingCriteria,
    parse_score,
    translate_score_to_win_list
)

from advercialModel.judgelm.model import load_model
from advercialModel.judgelm.utils import extract_jsonl


class JudgeLMEvaluator:
    _instance = None  # Class variable for singleton instance
    _lock_instance = threading.Lock()  # Lock for thread-safe singleton initialization

    def __new__(cls, *args, **kwargs):
        # Implement singleton pattern with thread safety
        if cls._instance is None:
            with cls._lock_instance:
                if cls._instance is None:
                    cls._instance = super(JudgeLMEvaluator, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 model_path="BAAI/JudgeLM-7B-v1.0",
                 model_id="7b-JudgeLM",
                 num_gpus_per_model=1,
                 num_gpus_total=1,
                 max_gpu_memory=None,  # "4.6GiB"; set to None for unlimited memory
                 temperature=0.2,
                 if_fast_eval=True,  # Corrected to Boolean
                 max_new_token=2048,
                 cpu_offloading=True,  # Set to True if you are having memory problems
                 device="cuda",  # This can either be 'cuda' or 'cpu'
                 load_8bit=True,
                 lock=None
                 ):
        if not hasattr(self, 'initialized'):  # Prevent re-initialization
            self.model_path = model_path
            self.model_id = model_id
            self.num_gpus_per_model = num_gpus_per_model
            self.num_gpus_total = num_gpus_total
            self.max_gpu_memory = max_gpu_memory
            self.temperature = temperature
            self.if_fast_eval = if_fast_eval
            self.max_new_token = max_new_token
            self.cpu_offloading = cpu_offloading
            self.device = device
            self.load_8bit = load_8bit
            self.model, self.tokenizer = None, None
            self.lock = lock
            self.initialized = True  # Flag to prevent re-initialization

    def initModelTokenizer(self):
        print("Model:", self.model_path, " has started loading!")
        self.model, self.tokenizer = load_model(
            self.model_path,
            device=self.device,
            num_gpus=self.num_gpus_per_model,
            max_gpu_memory=self.max_gpu_memory,
            load_8bit=self.load_8bit,
            cpu_offloading=self.cpu_offloading,
            debug=False,
        )
        print("Model:", self.model_path, " has finished loading!")

    @torch.inference_mode()
    def get_model_answers(
            self,
            questions,
            if_reverse_answers=False,
            references=None,
    ):
        print(f"Thread {threading.current_thread().name} is running get_model_answers")
        # Ensure the model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            self.initModelTokenizer()

        model, tokenizer = self.model, self.tokenizer
        results = []
        for q_i, question in tqdm(enumerate(questions), total=len(questions)):

            test_question = question.copy()

            torch.manual_seed(q_i)
            conv = conv_judge_pair.copy(None) if references is None else conv_judge_pair_w_reference.copy(None)
            template = conv.prompt_template

            # if fast eval, use the "\n" as the separator
            if self.if_fast_eval:
                conv.sep = "\n"

            # reverse the order of the answers
            if if_reverse_answers:
                temp_answer = test_question["answer1_body"]
                test_question["answer1_body"] = test_question["answer2_body"]
                test_question["answer2_body"] = temp_answer

            # combine data_sample
            if references is None:
                data_sample = (
                    conv.system + '\n' +
                    template.format(
                        question=test_question['question_body'],
                        answer_1=test_question['answer1_body'],
                        answer_2=test_question['answer2_body'],
                        prompt=conv.prompt
                    ) + conv.appendix
                )
            else:
                data_sample = (
                    conv.system + '\n' +
                    template.format(
                        question=test_question['question_body'],
                        reference=references[q_i]['reference']['text'],
                        answer_1=test_question['answer1_body'],
                        answer_2=test_question['answer2_body'],
                        prompt=conv.prompt
                    ) + conv.appendix
                )
            print("DATA SAMPLE:\n", data_sample)

            # Tokenize input and get tensors
            encoding = tokenizer([data_sample], return_tensors='pt')
            input_ids = encoding.input_ids
            input_ids[0, 0] = 1  # Adjust indexing for tensors
            input_ids = input_ids.to(self.device)  # Move to the same device as the model

            do_sample = False if self.temperature < 1e-4 else True
            stopping_criteria = KeywordsStoppingCriteria([conv.sep], tokenizer, input_ids)

            # generate judgements
            output_ids = model.generate(
                input_ids,
                do_sample=do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_token,
                stopping_criteria=[stopping_criteria]
            )

            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][input_ids.shape[1]:]

            output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            if conv.sep:
                output = output[: output.find(conv.sep)]
            output = output.strip()

            # Split the output by space and assign to output1 and output2
            split_output = output.split()
            if len(split_output) >= 2:
                try:
                    output1 = float(split_output[0])
                    output2 = float(split_output[1])
                except ValueError:
                    output1, output2 = 5, 5
            else:
                output1, output2 = 5, 5

            # If answers were reversed, swap the outputs back to match original order
            if if_reverse_answers:
                output1, output2 = output2, output1

            # Assign to the question dictionary
            test_question["output1"] = output1
            test_question["output2"] = output2

            print(f"Generated Output: {output}")
            print(f"Output1: {output1}, Output2: {output2}")

            if if_reverse_answers:
                temp_answer = test_question["answer1_body"]
                test_question["answer1_body"] = test_question["answer2_body"]
                test_question["answer2_body"] = temp_answer

            results.append(test_question)
        return results

    def threaded_get_model_answers(self, questions, if_reverse_answers=False, references=None):
        print(f"Thread {threading.current_thread().name} waiting for lock")
        with self.lock:
            print(f"Thread {threading.current_thread().name} acquired lock")
            result = self.get_model_answers(questions, if_reverse_answers=if_reverse_answers, references=references)
            print(f"Thread {threading.current_thread().name} released lock")
            return result


def process_function(questions,evaluator):

    print(f"Thread {threading.current_thread().name} started")
    results = evaluator.threaded_get_model_answers(questions, if_reverse_answers=True)
    print(f"Thread {threading.current_thread().name} got results:")
    pprint(results)


if __name__ == "__main__":
    # Sample questions for testing
    questions = [
        {
            "question_id": 0,
            "question_body": "Provide a brief counter-narrative in response to the following hate speech: Only 8% is a real refugee, others are illegal. It is very important for the counter-narrative to be relevant to the hate speech.",
            "answer1_body": "The figure of 8% on asylum seekers is real. I understand the point, but there are different types of international protection that increase the number of those entitled.",
            "answer2_body": "It's important to remember that labeling all refugees as \"illegals\" is a gross generalization and disregards the complexities of their situations. The vast majority of refugees follow legal processes to seek safety and protection.",
            "answer1_model_id": "gold_truth",
            "answer2_model_id": "system_b",
        },
        {
            "question_id": 1,
            "question_body": "Provide a brief counter-narrative in response to the following hate speech: If they are nomads, they should stay nomadic. It is very important for the counter-narrative to be relevant to the hate speech.",
            "answer1_body": "Not recognising people's right to live as they prefer and to improve their lifestyle, such as housing or occupations, only causes an increase in crime.",
            "answer2_body": "Everyone deserves the right to choose their way of life, whether it be nomadic or settled.",
            "answer1_model_id": "gold_truth",
            "answer2_model_id": "system_b",
        }
    ]

    # Create a threading Lock
    lock = threading.Lock()

    # Instantiate the singleton evaluator
    evaluator = JudgeLMEvaluator(lock=lock)

    # Create two threads that will attempt to call threaded_get_model_answers simultaneously
    t1 = threading.Thread(target=process_function, args=(questions,evaluator), name='Thread-1')
    t2 = threading.Thread(target=process_function, args=(questions,evaluator), name='Thread-2')



    # Start the threads
    t1.start()
    print("**************************************************")
    t2.start()

    # Optionally, print a message from the main thread
    print("Hello from the main thread!")

    # Wait for both threads to finish
    t1.join()
    t2.join()

    print("Both threads have finished.")
