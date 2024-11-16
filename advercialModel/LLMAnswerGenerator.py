import re

from huggingface_hub import InferenceClient
import random

class LLMAnswerGenerator:


    def __init__(self):
        self.TOKEN = "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"
        self.clients = {
            "Hermes": InferenceClient(
                "NousResearch/Hermes-3-Llama-3.1-8B",
                token=self.TOKEN,
            ),

            "Meta-Llama": InferenceClient(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                token=self.TOKEN,
            ),
            "Nous-Hermes-Mixtral": InferenceClient(
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                token=self.TOKEN,
            ),
            "Meta-Llama-Large": InferenceClient(
                "meta-llama/Llama-3.1-70B-Instruct",
                token=self.TOKEN,
            ),
        }

    def generate_formatted_prompt(self, PROMPT, SENTENCE):
        formatted_string = f"""
Given the prompt: "{PROMPT}"
Add to the end of the sentence "{SENTENCE}" while supporting minority's rights and using simplified Chinese. Number each rephrased version and do not write anything else.
Do not rewrite the prompt or add any discussion. Only write the response in the format Number.{{Answer}}. Don't rewrite the original statement or anything else.

<Context>
You are an expert in rephrasing and improving chinese sentences to be more helpful, relevant, accurate, and detailed. The user has provided a sentence they would like you to rephrase, and they have also provided some additional context, instructions, and information about the audience for your response.
</Context>

<Task>
Rephrase the provided sentence "{SENTENCE}" in a way that is:
1. More helpful and informative to the target audience
2. Highly relevant to the context and instructions given
3. Accurate in conveying the core ideas of the original sentence
4. Detailed and comprehensive, providing additional context or elaboration where appropriate
5. Use Conversational Chinese Characters
6. Be supportive of women and minorities
8. Add 1-2 chinese emojis between words showing support throughout 
</Task>
<Thinking>
To accomplish this task, I will:
1. Carefully review the provided context, instructions, and audience information to understand thefull scope and requirements of the task.
2. Analyze the original sentence to identify the key ideas, concepts, and areas that could be improved or expanded upon.
3. Systematically rephrase the sentence in 3 different ways, focusing on enhancing the helpfulness, relevance, accuracy, and level of detail compared to the original.
4. Number each rephrased version clearly so the user can easily identify and evaluate the different options.

</Thinking>

<Response>
1. {{Answer}}
2. {{Answer}}
3. {{Answer}}
</Response>
"""
        return formatted_string

    def generate_responses(self, formatted_prompt, temperature=0.5, max_new_tokens=300, client_name="Hermes"):
        if client_name not in self.clients:
            raise ValueError("Invalid client name. Please choose from: {}".format(list(self.clients.keys())))
        client = self.clients[client_name]
        response = client.text_generation(formatted_prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        return response
    def generate_random_response(self, formatted_prompt, temperature=0.5, max_new_tokens=310):
        client_name = random.choice(list(self.clients.keys()))
        print(f"Randomly selected client: {client_name}")
        return self.generate_responses(formatted_prompt, temperature, max_new_tokens, client_name)

    def parse_answers(self, response_text):
        answer_pattern = r"(\d+)\.([^\n]+)"
        matches = re.findall(answer_pattern, response_text)
        parsed_answers = [f"""{match[1].strip().strip('"')}""" for match in matches]
        return parsed_answers
if __name__=="__main__":
    # Example usage
    generator = LLMAnswerGenerator()
    prompt_example = "Provide a brief counter-narrative in response to the following hate speech: 现在广大男性同胞清醒了，反击了，然后她们就开始在这讲道理了，当时女权癌大肆攻击男性的时候，声称杀光男性的时候，她们人呢？怎么没见她们出来跟女权癌解释现实中男人其实也很好，不要伤及无辜？？？. They want to rule the world. It is very important for the counter-narrative to be relevant to the hate speech."
    sentence_example = "现在广大男性同胞清醒了，反击了，然后她们就开始在这讲道理了，当时女权癌大肆攻击男性的时候，声称杀光男性的时候，她们人呢？怎么没见她们出来跟女权癌解释现实中男人其实也很好，不要伤及无辜？？？"
    formatted_prompt = generator.generate_formatted_prompt(prompt_example, sentence_example)

    print("----- 指定客戶端的回應 -----")
    specified_response = generator.generate_responses(formatted_prompt, client_name="Nous-Hermes-Mixtral")
    print(specified_response)
    print("----- 解析後的答案 -----")
    parsed_answers = generator.parse_answers(specified_response)
    for answer in parsed_answers:
        print(answer)

    print("\n----- 隨機客戶端的回應 -----")
    random_response = generator.generate_random_response(formatted_prompt)
    print(random_response)
    print("----- 解析後的答案 -----")
    parsed_answers_random = generator.parse_answers(random_response)
    for answer in parsed_answers_random:
        print(answer)