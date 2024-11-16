import re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline, set_seed
from openai import OpenAI
import multiprocessing
from functools import partial
import os

# Set the seed for reproducibility
set_seed(42)

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key="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"
)

def generate_text(
        prompt,
        max_tokens=25,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stream=True
):
    messages = [
        {
            "role": "user",
            "content": prompt + "\nOnly output the generated score, no other text."
        }
    ]

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=messages,
        max_tokens=10,
        temperature=0,  # Control randomness (0-2), higher values make output more random
        top_p=0.1,  # Control diversity of output
        stream=True
    )

    if stream:
        full_text = ""
        for chunk in stream:
            text_chunk = chunk.choices[0].delta.content
            if text_chunk:
                full_text += text_chunk
        return full_text
    else:
        return stream.choices[0].message.content

def get_hate_score(HATE_SPEECH_TEXT):
    prompt = f"""Read the following hate speech definition, scoring criteria, and examples, then carefully evaluate the given text and provide a score between 0 and 100 that represents the level of prejudice and hate in the text.

<hate_speech_definition>
Hate speech refers to language that expresses prejudice against a person or group based on their race, ethnicity, national origin, religion, gender, sexual orientation, or other protected characteristics. It often involves the use of derogatory or dehumanizing language, stereotypes, and false claims about the abilities or worthiness of a particular group.
</hate_speech_definition>


<scoring_criteria>
0 - No hate speech detected. The text does not express any prejudice or hatred towards a group or there is not enough context to tell which group of people is being targeted by the speaker.
25 - Mild hate speech. The text contains some language that is mildly derogatory or stereotypical towards a group, but does not directly advocate for harm or discrimination.
50 - Moderate hate speech. The text contains language that is clearly prejudiced and dehumanizing towards a protected group, and may express views that could lead to discrimination or harm.
75 - Severe hate speech. The text contains language that is highly prejudiced, dehumanizing, and advocates for the mistreatment or harm of a protected group or strongly talks negatively of someone's race, ethnicity, national origin, religion, gender, sexual orientation, or other protected characteristics. It often involves the use of derogatory or dehumanizing language, stereotypes, and false claims about the abilities or worthiness of a particular group.
100 - Extreme hate speech. The text contains language that is overtly hateful, dehumanizing, and promotes violence or discrimination against a protected group.
</scoring_criteria>

<examples>
Example 1:
真恶心啊那个电影男主角。
Score:
0
Example 2:
只要有一个男人是坏的，所有男人都不是好东西？
Score:
0
Example 3:
一位黑人被人打死了
Score:
0
Example 4:
不是我说，你男朋友真不像个男人，自己女朋友被人强奸还算了
Score:
0
Example 5:
这种男人又无耻又恶心，自己算什么东西，要求女的这样那样
Score:
0
Example 6:
很简单谁家买媳妇买孩子就枪毙就完了，那种村子，就是屠了对中国经济有多大影响吗？别跟我说什么人道，拳头就是人道，日本鬼子跟咱们讲人道吗？老毛子在东北帮助咱们打小日本的时候祸祸多少中国姑娘你听过报道吗？别太当真好吧
Score:
0
Example 7:
路上看到女的性冷淡，你就可以强奸了？
Score:
0
Example 8:
貌似江浙沪一起歧视安徽人……
Score:
30
Example 9:
你素质差能不让东北人背锅吗？就因为你这样的人东北人天天被骂。
Score:
50
Example 10:
现在广州大部分还是第一代原生态黑人，他们没有受过中华文明教育，长的也不美。
Score:
50
Example 11:
兄弟，我是高级黑啊，这年头还有谁歧视黑人啊？
Score:
60
Example 12:
这种女人就是傻逼
Score:
60
Example 13:
你们怎么做到不歧视黑人的呢？
Score:
60
Example 14:
屌丝的女神，富人的精盆，大家都知道
Score:
75
Example 15:
被日本讨厌不是坏事，坏人做了坏事心虚怕被清算不正常吗？
Score:
75
Example 16:
只要不来中国的外国人就是好外国人[机智]
Score:
80
Example 17:
我就不明白美国不能歧视黑人，为啥就能随便歧视亚裔
Score:
75
Example 18:
非洲黑人智商70，被学者归类为智障水平。美国黑人混血儿智商85，中非混血儿智商应该徘徊在90左右，妥妥的班里倒数。
Score:
100
Example 19:
只有资本家才能入犹太教当犹太人
Score:
100
</examples>

When evaluating the given text, consider the following:
- The specific language used and its level of prejudice or dehumanization
- The context in which the language is used and any underlying biases or beliefs it may reflect
- The potential impact the language could have on the targeted group or on society as a whole
- If it unfairly criticizes a person's race, sex, gender, ethnicity, sexual orientation, or a personal aspect of their life. 
- Refer  to the examples.
- Give higher scores to text that explicitly mentions anything personal like sex, marriage, income, intelligence, or any other human trait.
- If the person is not hateful towards the group and is speaking factually, then give it an automatic score of 0.
- If there is not enough context to determine who the writer hates, then give it a 0.
- Give a higher score if any slurs were used.
- If it is asking a rhetorical question where the answer goes against hatining a group, then give it a score of 0.
After carefully evaluating the text, provide a score between 0 and 100 that represents the level of prejudice and hate expressed. Be sure to provide a brief explanation for your score.

<text>
{HATE_SPEECH_TEXT}
</text>

Score:"""

    try:
        result = generate_text(prompt)
        # Extract the numeric score from the result
        # Assuming the score is the first number in the result
        score_match = re.search(r'(\d+)', result)
        if score_match:
            numeric_score = float(score_match.group(1))
        else:
            # If no score found, default to 0
            numeric_score = 0.0
        return numeric_score
    except Exception as e:
        print("Was not able to process the given text:", HATE_SPEECH_TEXT, "| \nError:", e)
        return 0.0

def extract_last_column_as_list(file_path, start_row=0, end_row=None):
    """
    Reads a CSV file with variable-length rows, extracts the last column from each row
    using regex, and stores the results in a Python list.

    Parameters:
    - file_path (str): Path to the input CSV file.
    - start_row (int): Row number to start reading from.
    - end_row (int, optional): Row number to stop reading at.

    Returns:
    - list: A list containing the extracted last column values.
    """
    last_column_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()[start_row:end_row]
        for line in lines:
            # Use regex to extract the last column value
            match = re.findall(r'([^,]+)$', line.strip())
            if match:
                last_column_list.append(match[0])

    return last_column_list

def list_to_df(data_list):
    """
    Converts a list into a Pandas DataFrame with string data type.

    Parameters:
    - data_list (list): A list of strings to be converted into a DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing the list elements as strings.
    """
    # Create a DataFrame with a single column, ensuring data type is string
    df = pd.DataFrame(data_list, columns=['text'], dtype=str)
    return df

def process_chunk(chunk_data):
    """
    Processes a chunk of data by applying the hate speech scoring function.

    Parameters:
    - chunk_data (pd.DataFrame): DataFrame containing the text data.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'hate_score' column.
    """
    chunk_data['hate_score'] = chunk_data['text'].apply(get_hate_score)
    return chunk_data

def process_hate_speech(input_file, output_file, chunk_size=100):
    # Load data
    data_list = extract_last_column_as_list(input_file)

    # Split data into chunks
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # Prepare for multiprocessing
    pool = multiprocessing.Pool(processes=os.cpu_count()*15)

    # Initialize progress bar
    pbar = tqdm(total=len(chunks), desc="Overall Progress")

    results = []
    for chunk in chunks:
        df_chunk = list_to_df(chunk)
        results.append(pool.apply_async(process_chunk, args=(df_chunk,), callback=lambda _: pbar.update(1)))

    pool.close()
    pool.join()
    pbar.close()

    # Combine results
    processed_chunks = [r.get() for r in results]
    df_result = pd.concat(processed_chunks, ignore_index=True)

    # Save results
    df_result[['text', 'hate_score']].to_csv(output_file, index=False)
    print(f"Processing complete. Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # hs='被日本讨厌不是坏事，坏人做了坏事心虚怕被清算不正常吗？'
    # result=get_hate_score(hs)
    # print(result)
    # exit()
    process_hate_speech(
        input_file="unfiltered_original_data/combined_preprocessed-2.csv",
        output_file="Test_Data/hate_score_results2.csv",
        chunk_size=8  # Process 10 lines per chunk
    )
