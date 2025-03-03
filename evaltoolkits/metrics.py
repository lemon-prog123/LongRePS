import re
import string

import jieba
from fuzzywuzzy import fuzz
import difflib

from typing import List
from collections import Counter
from rouge import Rouge

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def babi_score(prediction, ground_truth, **kwargs):
    
    if ground_truth in prediction:
        return 1.0
    elif prediction==ground_truth:
        return 1.0
    elif prediction==(ground_truth+')'):
        return 1.0
    elif prediction==('('+ground_truth+')'):
        return 1.0
    else:
        return 0

def babiq3_score(prediction, ground_truth, **kwargs):
    try:
        answer=prediction.split('was in')[1]
    except:
        answer=prediction
    if ground_truth in answer:
        return 1.0
    else:
        return 0

def ruler_score(prediction, ground_truth, **kwargs):
    def postprocess_pred(predict_str: str):

        predict_str = predict_str.strip()
        # Remove all non-printable characters
        np_pattern = re.compile(r'[\x00-\x1f]')
        predict_str = np_pattern.sub('\n', predict_str).strip()
        return predict_str
    positive_pattern1 = r':\s*(\d+)'
    positive_pattern2 = r'is\s*(\d+)'
    negative_pattern = r'no \s* magic number'
    positive_match1 = re.search(positive_pattern1, prediction)
    positive_match2 = re.search(positive_pattern2, prediction)
    negative_match = re.search(negative_pattern, prediction)
    if negative_match:
        return 0
    elif positive_match1 or positive_match2:
        return 1.0
    
    if ground_truth in prediction:
        return 1.0
    elif prediction==ground_truth:
        return 1.0
    elif prediction==(ground_truth+')'):
        return 1.0
    elif prediction==('('+ground_truth+')'):
        return 1.0
    elif postprocess_pred(prediction)==ground_truth:
        return 1.0
    else:
        return 0
    
def accuracy_score(prediction, ground_truth, **kwargs):
    def extract_answer(response):
        response = response.replace('*', '')
        match = re.search(r'The correct answer is \(([A-D])\)', response)
        if match:
            return match.group(1)
        else:
            match = re.search(r'The correct answer is ([A-D])', response)
            if match:
                return match.group(1)
            else:
                return None
    bool_cnt=0
    choice_list=['A','B','C','D']
    pattern1 = f'{ground_truth} '
    pattern3=f'\* {ground_truth}'
    pattern4=f'{ground_truth}$'
    for choice in choice_list:
        if ('('+choice+')') in prediction:
            bool_cnt+=1
            continue
        pattern2 = f'{choice} '
        matches = re.findall(pattern2, prediction)
        if matches:
            bool_cnt+=1
            continue
    if bool_cnt>=2: #m choices
        return 0
    if ('('+ground_truth+')') in prediction:
        return 1.0
    if ground_truth==prediction:
        return 1.0
    matches1 = re.findall(pattern1, prediction)
    matches2 = re.findall(pattern3, prediction)
    matches3 = re.findall(pattern4, prediction)
    if matches1 or matches2 or matches3:
        return 1.0
    else:
        return 0

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    #print(prediction)
    #if '\n' in prediction:
    #    prediction = prediction.lstrip('\n').split('\n')[0]
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score
    
def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def recall_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    recall = 1.0 * num_same / len(ground_truth)
    return recall

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_recall_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return recall_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)
