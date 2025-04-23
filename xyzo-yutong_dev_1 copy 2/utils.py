from torch.nn import CrossEntropyLoss
import torch
import re
from word2number import w2n
def regularized_logp(logits, labels, vocab_size, label_smoothing, reduction):
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def regularized_logp_tensor(logits, labels, vocab_size, label_smoothing):
    logits = logits.float()
    # print("logits.shape=", logits.shape)
    # print("labels.shape=", labels.shape)

    loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)

    tensor_loss = torch.zeros(logits.shape[0]).to(logits.device)
    for i in range(logits.shape[0]):
        tmp_shift_logits = logits[i, :-1, :]
        tmp_shift_labels = labels[i, 1:]
        tmp_loss = loss_fct(tmp_shift_logits, tmp_shift_labels)
        tensor_loss[i] = tmp_loss
    # print("tensor_loss=", tensor_loss)
    # print("torch.meam(tensor_loss)=", torch.mean(tensor_loss))

    return tensor_loss


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]
def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text

def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def extract_answer(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def extract_answer_new(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    # if "final answer is $" in pred_str and "$. I hope" in pred_str:
    #     # minerva_math
    #     tmp = pred_str.split("final answer is $", 1)[1]
    #     pred = tmp.split("$. I hope", 1)[0].strip()
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    # elif "he answer is" in pred_str:
    #     pred = pred_str.split("he answer is")[-1].strip()
    # elif "final answer is" in pred_str:
    #     pred = pred_str.split("final answer is")[-1].strip()
    # elif "答案是" in pred_str:
    #     # Handle Chinese few-shot multiple choice problem answer extraction
    #     pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        pred = ""
        # if use_last_number:
        #     pattern = "-?\d*\.?\d+"
        #     pred = re.findall(pattern, pred_str.replace(",", ""))
        #     if len(pred) >= 1:
        #         pred = pred[-1]
        #     else:
        #         pred = ""
        # else:
        #     pred = ""

    # pred = re.sub(r"\n\s*", "", pred)
    # if pred != "" and pred[0] == ":":
    #     pred = pred[1:]
    # if pred != "" and pred[-1] == ".":
    #     pred = pred[:-1]
    # if pred != "" and pred[-1] == "/":
    #     pred = pred[:-1]
    # pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred

def extract_answer_newb(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    # if "final answer is $" in pred_str and "$. I hope" in pred_str:
    #     # minerva_math
    #     tmp = pred_str.split("final answer is $", 1)[1]
    #     pred = tmp.split("$. I hope", 1)[0].strip()
    # elif "boxed" in pred_str:
    #     ans = pred_str.split("boxed")[-1]
    #     if len(ans) == 0:
    #         return ""
    #     elif ans[0] == "{":
    #         stack = 1
    #         a = ""
    #         for c in ans[1:]:
    #             if c == "{":
    #                 stack += 1
    #                 a += c
    #             elif c == "}":
    #                 stack -= 1
    #                 if stack == 0:
    #                     break
    #                 a += c
    #             else:
    #                 a += c
    #     else:
    #         a = ans.split("$")[0].strip()
    #     pred = a
    if "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    # elif "final answer is" in pred_str:
    #     pred = pred_str.split("final answer is")[-1].strip()
    # elif "答案是" in pred_str:
    #     # Handle Chinese few-shot multiple choice problem answer extraction
    #     pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:
        pred = ""
    # else:  # use the last number
    #     pred = ""
    #     if use_last_number:
    #         pattern = "-?\d*\.?\d+"
    #         pred = re.findall(pattern, pred_str.replace(",", ""))
    #         if len(pred) >= 1:
    #             pred = pred[-1]
    #         else:
    #             pred = ""
    #     else:
    #         pred = ""

    # pred = re.sub(r"\n\s*", "", pred)
    # if pred != "" and pred[0] == ":":
    #     pred = pred[1:]
    # if pred != "" and pred[-1] == ".":
    #     pred = pred[:-1]
    # if pred != "" and pred[-1] == "/":
    #     pred = pred[:-1]
    # pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred

def extract_answer_myans(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    # elif "boxed" in pred_str:
    #     ans = pred_str.split("boxed")[-1]
    #     if len(ans) == 0:
    #         return ""
    #     elif ans[0] == "{":
    #         stack = 1
    #         a = ""
    #         for c in ans[1:]:
    #             if c == "{":
    #                 stack += 1
    #                 a += c
    #             elif c == "}":
    #                 stack -= 1
    #                 if stack == 0:
    #                     break
    #                 a += c
    #             else:
    #                 a += c
    #     else:
    #         a = ans.split("$")[0].strip()
    #     pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    # else:
    #     pred = ""
    else:  # use the last number
        pred = ""
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # pred = re.sub(r"\n\s*", "", pred)
    # if pred != "" and pred[0] == ":":
    #     pred = pred[1:]
    # if pred != "" and pred[-1] == ".":
    #     pred = pred[:-1]
    # if pred != "" and pred[-1] == "/":
    #     pred = pred[:-1]
    # pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred

import multiprocessing
from math import isclose
from typing import Union

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex


def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False

def math_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = True,
                is_close: bool = True,
                timeout: bool = False,
                ) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    try: # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=1e-4):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or \
        (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ['{', "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (prediction.startswith("[") and prediction.endswith("]")) and (reference.startswith("[") and reference.endswith("]")) or \
        (prediction.startswith("(") and prediction.endswith(")")) and (reference.startswith("(") and reference.endswith(")")):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]):
                return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s
    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a-b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):  
    result = symbolic_equal(a, b)
    output_queue.put(result)  


def call_with_timeout(func, *args, timeout=1, **kwargs):  
    output_queue = multiprocessing.Queue()  
    process_args = args + (output_queue,)  
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)  
    process.start()  
    process.join(timeout)  
  
    if process.is_alive():  
        process.terminate()  
        process.join()  
        return False  
  
    return output_queue.get()


def _test_math_equal():
    # print(math_equal("0.0833333333333333", "\\frac{1}{12}"))
    # print(math_equal("(1,4.5)", "(1,\\frac{9}{2})"))
    print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}", timeout=True))

if __name__ == "__main__":
    _test_math_equal()