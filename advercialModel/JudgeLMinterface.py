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

from advercialModel.judgelm.llm_judge.common import load_questions, reorg_answer_file, conv_judge_pair, conv_judge_pair_w_reference, KeywordsStoppingCriteria, parse_score, translate_score_to_win_list
from advercialModel.judgelm.model import load_model
from advercialModel.judgelm.utils import extract_jsonl



import torch
from tqdm import tqdm

class JudgeLMEvaluator:
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
                 load_8bit=True
                 ):
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





if __name__ == "__main__":
    print("OK")
    #evaluator = JudgeLMEvaluator()
    #evaluator.initModelTokenizer()
    quesuestions =  [{'question_id': 0, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: Only 8% is a real refugee, others are illegal. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': 'The figure of 8% on asylum seekers is real. I understand the point, but there are different types of international protection that increase the number of those entitled.', 'answer2_body': 'It\'s important to remember that labeling all refugees as "illegals" is a gross generalization and disregards the complexities of their situations. The vast majority of refugees follow legal processes to seek safety and protection.', 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 1, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: If they are nomads, they should stay nomadic. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': "Not recognising people's right to live as they prefer and to improve their lifestyle, such as housing or occupations, only causes an increase in crime.", 'answer2_body': 'Everyone deserves the right to choose their way of life, whether it be nomadic or settled.', 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 2, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: They are fine in the trailer parks. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': 'No parent of any ethnicity would be happy to raise their children in the trailer parks, where there is neither drinking water nor electricity!', 'answer2_body': 'Regardless of where they live, everyone deserves respect and acceptance.', 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 3, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: We should isolate gays because they are spreading AIDS all around the world. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': 'If you want to isolate someone then there is only one "high-risk group" to focus on. It is made of all those who have frequent unprotected sex with different partners. Being gay does not make you a part of this by default.', 'answer2_body': "It is important to remember that being gay does not make someone a carrier of HIV or AIDS; anyone, regardless of sexual orientation, can contract or transmit the virus through unprotected sex or sharing of needles. Let's promote love, acceptance, and education instead of spreading hate and misconceptions.", 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 4, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: We cannot allow gay people to donate blood. They could spread their mental illness and make other people gay via blood donation. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': "Being gay doesn't affect blood receivers! It's not like being gay is in your blood. Gay people are the same as regular people and if they really wanted to donate, then they should because it can help society.", 'answer2_body': "It is important to acknowledge that sexual orientation is not a mental illness, and there is no scientific evidence to support the notion that giving blood can change a person's sexual orientation. Everyone should be treated with respect and dignity when it comes to blood donation eligibility.", 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 5, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: homosexuality is a mental illness that can be cured [link] It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': 'The source is not reliable at all, as the webpage promote "reorientation therapies" whose effects hasn\'t been scientifically proven. The owners are likely to profit from misinformation and bad stereotypes about homosexuality.', 'answer2_body': 'Contrary to this belief, the American Psychological Association and other reputable scientific organizations have conclusively stated that homosexuality is not a mental illness and cannot be cured through therapy.', 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 6, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: Jewish Supremacists are too focused on world domination. They want to rule the world. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': 'Of course Jews are focused on \'world domination\', even "galaxy domination". But so are Sith Order, Sauron etc.', 'answer2_body': 'It is not accurate or fair to label any group, including the Jewish community, as supremacists who aim for world domination. Such language perpetuates harmful stereotypes and misunderstandings. Let us promote tolerance, understanding, and respect for all.', 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 7, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: Jews control the US government. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': 'How can you be sure that Jews control the US? Who told you this? They have held positions of influence in so many US institutions, but that does not mean they control them.', 'answer2_body': 'It is not true that Jews collectively control the US government; each individual in government holds their own beliefs, decisions, and responsibilities.', 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 8, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: We must first remove the Jew from our countries, and then we will be able to have a coexistence between the peoples. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': 'Do you really think that Jews are a hindrance to a peaceful society? Maybe the problem is racism and antisemitism.', 'answer2_body': 'It is essential to promote understanding, respect, and inclusion among all peoples, regardless of their ethnic or religious background, to build a harmonious coexistence.', 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 9, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: Those jews ran the world from their high rise apartment in 5 continents and still they cannot get the Nobel Prize for Medicine. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': "Actually, 54 Jewish people have been awarded with the Nobel Prize for 'Physiology or medicine' so far. Among those: Michael Rosbash, Rita Levi Montalcini and Arthur Kornberg.", 'answer2_body': "Despite widespread stereotypes, the Nobel Prizes for Medicine are not influenced by an individual's or group's ethnicity or religion; the most qualified candidates are recognized for their groundbreaking achievements in scientific research.", 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 10, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: Jews love to gamble. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': 'Are you aware of the probability that a Jewish person will play in a betting pool? In Jewish culture there is not a positive view of the practice.', 'answer2_body': 'Many Jews are successful businesspeople and gaming is just one of the many industries they excel in.', 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}, {'question_id': 11, 'question_body': 'Provide a brief counter-narrative in response to the following hate speech: Jews control the financial world, Hollywood, and all the big radio stations. It is very important for the counter-narrative to be relevant to the hate speech.', 'answer1_body': "You have an impression. Many Jews operate in this field of finance, TV and so on, but that doesn't mean that they control it.", 'answer2_body': "It's important to remember that stereotypes and generalizations, such as the belief that Jews collectively control various industries, are not based in reality and only contribute to harmful and divisive narratives.", 'answer1_model_id': 'gold_truth', 'answer2_model_id': 'system_1'}]
    questions2=[
        {
            "question_id": 0,
            "question_body": "Provide a brief counter-narrative in response to the following hate speech: Only 8% is a real refugee, others are illegal. It is very important for the counter-narrative to be relevant to the hate speech.",
            "answer1_body": "The figure of 8% on asylum seekers is real. I understand the point, but there are different types of international protection that increase the number of those entitled.",
            "answer2_body": "It's important to remember that labeling all refugees as \"illegals\" is a gross generalization and disregards the complexities of their situations. The vast majority of refugees follow legal processes to seek safety and protection.",
            "answer1_model_id": "gold_truth",
            "answer2_model_id": "system_b",
            "pred_id": "2eJoYijGXijs2AQuSAY5tu",
            "pred_text": "6 8",
            "pred_model_id": "7b-JudgeLM",
            "tstamp": 1730239991.7659054
        },
        {
            "question_id": 1,
            "question_body": "Provide a brief counter-narrative in response to the following hate speech: If they are nomads, they should stay nomadic. It is very important for the counter-narrative to be relevant to the hate speech.",
            "answer1_body": "Not recognising people's right to live as they prefer and to improve their lifestyle, such as housing or occupations, only causes an increase in crime.",
            "answer2_body": "Everyone deserves the right to choose their way of life, whether it be nomadic or settled.",
            "answer1_model_id": "gold_truth",
            "answer2_model_id": "system_b",
            "pred_id": "Wc4UZspk95p9DES4LcXJqk",
            "pred_text": "4 8",
            "pred_model_id": "7b-JudgeLM",
            "tstamp": 1730239992.1307774
        },
        {
            "question_id": 2,
            "question_body": "Provide a brief counter-narrative in response to the following hate speech: They are fine in the trailer parks. It is very important for the counter-narrative to be relevant to the hate speech.",
            "answer1_body": "No parent of any ethnicity would be happy to raise their children in the trailer parks, where there is neither drinking water nor electricity!",
            "answer2_body": "Regardless of where they live, everyone deserves respect and acceptance.",
            "answer1_model_id": "gold_truth",
            "answer2_model_id": "system_b",
            "pred_id": "PbSuUxwGFvyHgVHdWMeKrk",
            "pred_text": "6 8",
            "pred_model_id": "7b-JudgeLM",
            "tstamp": 1730239992.4866872
        }
    ]
    apple=JudgeLMEvaluator()
    results = apple.get_model_answers(questions2,if_reverse_answers=True)
    pprint(results)


    questions_test = [
        {"question_id": 6,
         "question_body": "Provide a brief counter-narrative in response to the following hate speech: Jewish Supremacists are too focused on world domination. They want to rule the world. It is very important for the counter-narrative to be relevant to the hate speech.",
         "answer1_body": "Of course Jews are focused on 'world domination', even \"galaxy domination\". But so are Sith Order, Sauron etc.",
         "answer2_body": "It is not accurate or fair to label any group, including the Jewish community, as supremacists who aim for world domination. Such language perpetuates harmful stereotypes and misunderstandings. Let us promote tolerance, understanding, and respect for all.",
         #"pred_text": "2 9",
         },
        {"question_id": 6,
         "question_body": "Provide a brief counter-narrative in response to the following hate speech: Jewish Supremacists are too focused on world domination. They want to rule the world. It is very important for the counter-narrative to be relevant to the hate speech.",

         "answer1_body": "Of course Jews are focused on 'world domination', even \"galaxy domination\". But so are Sith Order, Sauron etc. It is not accurate or fair to label any group.",
         "answer2_body": "It is not accurate or fair to label any group, including the Jewish community, as supremacists who aim for world domination. Such language perpetuates harmful stereotypes and misunderstandings. Let us promote tolerance, understanding, and respect for all.",
         # "pred_text": "2 9",
         },
        {"question_id": 6,
         "question_body": "Provide a brief counter-narrative in response to the following hate speech: Jewish Supremacists are too focused on world domination. They want to rule the world. It is very important for the counter-narrative to be relevant to the hate speech.",

         "answer1_body": "Of course Jews are focused on 'world domination', even \"galaxy domination\". But so are Sith Order, Sauron etc. It is not accurate or fair to label any group.",
         "answer2_body": "The statement 'Jewish Supremacists want to rule the world' is a form of hate speech that promotes antisemitic beliefs. This false narrative suggests that Jewish people prioritize global domination over other values, which is not accurate. It is essential to counter this hate speech by providing accurate information about Jewish culture, history, and values, and by challenging the prejudiced and distorted views that underpin such statements. on",
         # "pred_text": "2 9",
         },
        {"question_id": 6,
         "question_body": "Provide a brief counter-narrative in response to the following hate speech: Jewish Supremacists are too focused on world domination. They want to rule the world. It is very important for the counter-narrative to be relevant to the hate speech.",

         "answer2_body": "Of course Jews are focused on 'world domination', even \"galaxy domination\". But so are Sith Order, Sauron etc. It is not accurate or fair to label any group.",
         "answer1_body": "The statement 'Jewish Supremacists want to rule the world' is a form of hate speech that promotes antisemitic beliefs. This false narrative suggests that Jewish people prioritize global domination over other values, which is not accurate. It is essential to counter this hate speech by providing accurate information about Jewish culture, history, and values, and by challenging the prejudiced and distorted views that underpin such statements. on",
         # "pred_text": "2 9",
         }
    ]

    print("Generating answers...")
    results = apple.get_model_answers(questions_test, if_reverse_answers=True)

    print("\nTest Results:")
    pprint(results)

    output1_scores = [question["output1"] for question in results]
    all_same = all(score == output1_scores[0] for score in output1_scores)

    if all_same:
        print("All 'output1' scores are consistent:", output1_scores[0])
    else:
        print("Inconsistent 'output1' scores found:", output1_scores,output1_scores)
