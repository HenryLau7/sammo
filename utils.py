from sammo.base import EvaluationScore
from sammo.data import DataTable
import re
from pebble import ProcessPool
from tqdm import tqdm


def accuracy(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    def normalize(x):
        if isinstance(x, dict):
            print(x)
        if isinstance(x, list):
            if len(x) == 0:
                return ""
            x = x[0]
        return x.lower().replace(" ", "")

    mistakes = list()
    y_in = y_true.inputs.raw_values
    # import pdb; pdb.set_trace()
    y_true, y_pred = y_true.outputs.normalized_values(), y_pred.outputs.normalized_values(on_empty="")

    for i in range(len(y_true)):
        is_mistake = normalize(y_true[i]) != normalize(y_pred[i])
        is_mistake = is_mistake and normalize(y_in[i] + y_true[i]) != normalize(y_pred[i])
        if is_mistake:
            mistakes.append(i)

    accuracy = 1 - len(mistakes) / len(y_true)
    return EvaluationScore(accuracy, mistakes)

def accuracy_GSM8K(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    def _is_number(s):
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None
    def extract_answer(completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None
        # assert isinstance(completion, str)
        try:
            if not isinstance(completion,str):
                return None
            preds = completion.split("The answer is")
            # preds = isolate_answer(completion)
            # print(preds)
            if len(preds) > 1:
                pred = preds[1] 
            else:
                pred = preds[-1]
            pred = pred.replace(",", "") 
            numbers = re.findall(r"-?\d+\.?\d*", pred)
            if len(numbers) == 0:
                return None
            else:
                pred = numbers[-1]
            if pred != "" and pred[-1] == ".":
                pred = pred[:-1]
            pred = pred.replace(",", "").replace("\n", "")
            is_number, pred = _is_number(pred)
        except Exception as e:
            print(e)
            print("Error in extracting answer from completion:\n")
            print(completion)
            return None

        if is_number:
            return pred
        else:
            return None

    def check_answer(pred, answer, tolerance=1e-6):
        '''Check if the response matches the ground truth, and return the metric (bool or number)'''

        # gt_label = extract_answer(answer)
        pred_answer = extract_answer(pred)

        if (pred_answer is None) or (answer is None):
            return False

        try:
            pred_value = float(pred_answer)
            gt_value = float(answer)
        except ValueError:
            return False

        return abs(pred_value - gt_value) <= tolerance
    
    mistakes = list()
    # y_in = y_true.inputs.raw_values
    # import pdb; pdb.set_trace()
    y_true, y_pred = y_true.outputs.normalized_values(), y_pred.outputs.normalized_values()
    for i in range(len(y_true)):
        score = check_answer(y_pred[i], y_true[i])
        if score == False:
            mistakes.append(i)

    accuracy = 1 - len(mistakes) / len(y_true)
    return EvaluationScore(accuracy, mistakes)


def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    formatted_hours = f"{hours:.0f}"
    formatted_minutes = f"{minutes:.0f}"
    formatted_secs = f"{secs:.0f}"
    
    return f"{formatted_hours} H {formatted_minutes} MIN {formatted_secs} S"



################ MATH ################
from toolkit_for_MATH.latex_answer_check import latex_answer_check as latex_equiv

# 将 check_answer 移到全局作用域
def check_answer_MATH(pred, label):
    answer_gt = extract_answer_from_gold_solution(label)
    answer_pred = extract_answer_from_model_completion(pred)

    return check_answers_equiv(answer_gt, answer_pred)

# 将 check_answers_equiv 移到全局作用域
def check_answers_equiv(answer_a: str, answer_b: str):
    if answer_a is None or answer_b is None:
        return False

    if answer_a == "" or answer_b == "":
        return False
    
    answer_a = answer_a.strip()
    answer_b = answer_b.strip()
    
    if answer_a.lower() == answer_b.lower():
        return True

    try:
        res = latex_equiv(answer_a, answer_b)
    except Exception as e:
        print(e)
        res = False

    return res

# 将 extract_answer_from_gold_solution 移到全局作用域
def extract_answer_from_gold_solution(solution: str):
    def remove_boxed(s):
        left = "\\boxed{"
        try:
            assert s[: len(left)] == left
            assert s[-1] == "}"
            return s[len(left) : -1]
        except:
            return None

    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    return remove_boxed(last_boxed_only_string(solution))

def isolate_answer(text: str):
    if text is None:
        return None
    
    assert isinstance(text, str)
    text = text.lower()
    split_ans = text.split("answer is".lower())
    if len(split_ans) > 1:
        ans = split_ans[-1].replace(":", "").strip()
        extract_ans_temp = ans.split(".\n")[0].strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip().strip("\n")
        return extract_ans
    else:
        return text

def extract_answer_from_model_completion(completion):
    answer_split = isolate_answer(completion)
    return answer_split

# 原 accuracy_MATH 函数保持不变
def accuracy_MATH(y_true: DataTable, y_pred: DataTable):
    mistakes = list()

    y_true, y_pred = y_true.outputs.normalized_values(), y_pred.outputs.normalized_values()

    with ProcessPool(max_workers=60) as pool:
        future = pool.map(check_answer_MATH, y_pred, y_true, timeout=10)
        score_list = []
        try:
            for score in tqdm(future.result(), total=len(y_pred), desc="Evaluating", leave=True):
                score_list.append(score)
        except Exception as e:
            print(f"Error occurred: {e}")

    for i in range(len(score_list)):
        score = score_list[i]
        if score == False:
            mistakes.append(i)

    accuracy = 1 - len(mistakes) / len(y_true)
    return EvaluationScore(accuracy, mistakes)
