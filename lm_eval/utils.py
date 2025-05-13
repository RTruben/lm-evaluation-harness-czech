import collections
import copy
import fnmatch
import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from itertools import islice
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Iterable,
    Tuple,
    Union, )

import numpy as np
import yaml
from jinja2 import BaseLoader, Environment, StrictUndefined


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
eval_logger = logging.getLogger("lm-eval")

SPACING = " " * 47

HIGHER_IS_BETTER_SYMBOLS = {
    True: "↑",
    False: "↓",
}


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def escaped_split(text, sep_char, maxsplit=-1):
    """Split text into a list on occurrences of the given separation
    character `sep_char`. The separation character may be escaped by a
    backslash to avoid splitting at that location.

    The separation character must be a string of size 1.

    If `maxsplit` is given, at most `maxsplit` splits are done (thus,
    the list will have at most `maxsplit + 1` elements). If `maxsplit`
    is not specified or less than 0, then there is no limit on the
    number of splits (all possible splits are made).
    """
    assert (
        len(sep_char) == 1
    ), "separation string must be a single character for escaped splitting"

    if maxsplit == 0:
        return text
    maxsplit = max(0, maxsplit)

    return re.split(r"(?<!\\)" + sep_char, text, maxsplit)


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def sanitize_list(sub):
    """
    Takes possible nested list and recursively converts all inner component to strings
    """
    if isinstance(sub, list):
        return [sanitize_list(item) for item in sub]
    if isinstance(sub, tuple):
        return tuple(sanitize_list(item) for item in sub)
    else:
        return str(sub)


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict


def join_iters(iters):
    for iter in iters:
        yield from iter


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    if isinstance(patterns, str):
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string


def get_file_task_name(filename: str) -> str:
    """
    Given the sample results filenames, extracts and returns the task name.
    """
    return filename[filename.find("_") + 1 : filename.rfind("_")]


def get_file_datetime(filename: str) -> str:
    """
    Given the results and sample results filenames, extracts and returns the datetime.
    """
    return filename[filename.rfind("_") + 1 :].replace(".json", "")


def sanitize_model_name(model_name: str) -> str:
    """
    Given the model name, returns a sanitized version of it.
    """
    return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_name)


def sanitize_task_name(task_name: str) -> str:
    """
    Given the task name, returns a sanitized version of it.
    """
    return re.sub(r"\W", "_", task_name)


def get_latest_filename(filenames: List[str]) -> str:
    """
    Given a list of filenames, returns the filename with the latest datetime.
    """
    return max(filenames, key=lambda f: get_file_datetime(f))


def get_results_filenames(filenames: List[str]) -> List[str]:
    """
    Extracts filenames that correspond to aggregated results.
    """
    return [f for f in filenames if "/results_" in f and ".json" in f]


def get_sample_results_filenames(filenames: List[str]) -> List[str]:
    """
    Extracts filenames that correspond to sample results.
    """
    return [f for f in filenames if "/samples_" in f and ".json" in f]


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def make_disjoint_window(pair):
    """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Provides a proper json encoding for the loggers and trackers json dumps.
    Notably manages the json encoding of dataclasses.
    """

    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


class Reorderer:
    def __init__(self, arr: List[Any], fn: Callable) -> None:
        """Reorder an array according to some function

        Args:
            arr (List[Any]): The initial array
            fn (Callable[[Any], Any]): A function to determine the priority of elements
        """
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        # arr = [([y[0] for y in x], x[0][1]) for x in arr]
        # TODO: overhaul reorderer. It currently grouped requests by content but we don't want this
        arr = [([y[0]], x[0][1]) for x in arr for y in x]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr

    def get_reordered(self):
        """Gets the reordered array

        Returns:
            List[Any]: The reordered array
        """
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        """Restores the original order of a new array based on the old array's order

        Args:
            newarr (List[Any]): The array to be restored

        Returns:
            List[Any]: The array restored to the original order
        """
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res
    
def get_chrf(tasks):
    baseline = 0
    best = 100
    result_agg = 0
    samples_agg = 0
    for task, sample_count in tasks:
        result_chrf = task.get("chrf,none", 0)
        normalized = 100 * ((result_chrf - baseline) / (best - baseline))
        result_agg += (sample_count * normalized)
        samples_agg += sample_count
    return (result_agg / samples_agg)

def get_ter(tasks):
    baseline = 150
    best = 0
    result_agg = 0
    samples_agg = 0
    for task, sample_count in tasks:
        result_ter = task.get("ter,none", 0)
        normalized = 100 * ((baseline - result_ter) / (baseline - best))
        result_agg += (sample_count * normalized)
        samples_agg += sample_count
    return (result_agg / samples_agg)

def get_exact_match(tasks):
    baseline = 0
    best = 1
    result_agg = 0
    samples_agg = 0
    for task, sample_count in tasks:
        result_em = task.get("exact_match,none", 0)
        normalized = 100 * ((result_em - baseline) / (best - baseline))
        result_agg += (sample_count * normalized)
        samples_agg += sample_count
    return (result_agg / samples_agg)

def get_acc(acc_tasks):
    result_agg = 0
    samples_agg = 0
    for task, sample_count, baseline, best in acc_tasks:
        result_acc = task.get("acc,none", 0)
        normalized = 100 * ((result_acc - baseline) / (best - baseline))
        result_agg += (sample_count * normalized)
        samples_agg += sample_count
    return (result_agg / samples_agg)
    
def calculate_single_metric(results):
    """Calculate and print single value metric"""
    agree_single = results.get("aver_agree_single_var", None)
    agree_double = results.get("aver_agree_double_var", None)
    belebele_mask = results.get("aver_belebele", None)
    aver_csgec = results.get("aver_csgec", None)
    aver_hellaswag = results.get("aver_hellaswag", None)
    aver_sqad = results.get("aver_sqad", None)
    aver_umimeto = results.get("aver_umime_to", None)
    # second value in tuple denotes amount of samples per task
    mask_tasks = [(agree_single, 713), (agree_double, 329), (belebele_mask, 750), (aver_sqad, 500)]
    # last two values in tuple denote baseline and best scores
    acc_tasks = [(aver_csgec, 1000, 0.556, 1.0), (aver_hellaswag, 1000, 0.263, 1.0), (aver_umimeto, 700, 0.5, 1.0)]
    chrf_performance = round(get_chrf(mask_tasks), 2)
    ter_performance = round(get_ter(mask_tasks), 2)
    exact_match_performance = round(get_exact_match(mask_tasks), 2)
    acc_performance = round(get_acc(acc_tasks), 2)
    combined = (chrf_performance + ter_performance + exact_match_performance + acc_performance) / 4
    return {
        "combined": round(combined, 2),
        "chrf": chrf_performance,
        "ter": ter_performance,
        "em": exact_match_performance,
        "acc": acc_performance
    }
    
def print_single_metric(results):
    print(f"Normalized CHRF performance across all tasks is: {results["chrf"]}")
    print(f"Normalized TER performance across all tasks is: {results["ter"]}")
    print(f"Normalized EM performance across all tasks is: {results["em"]}")
    print(f"Normalized ACC performance across all tasks is: {results["acc"]}")
    print(f"Combined performance across all metrics of all tasks is: {results["combined"]}")

def make_table(result_dict, column: str = "results", sort_results: bool = True):
    """Generate table of results."""
    from pytablewriter import LatexTableWriter, MarkdownTableWriter

    if column == "results":
        column_name = "Tasks"
    elif column == "groups":
        column_name = "Groups"

    all_headers = [
        column_name,
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = all_headers
    latex_writer.headers = all_headers

    values = []

    keys = result_dict[column].keys()
    if sort_results:
        # sort entries alphabetically
        keys = sorted(keys)
    for k in keys:
        dic = result_dict[column][k]
        version = result_dict["versions"].get(k, "N/A")
        n = str(result_dict["n-shot"][k])
        higher_is_better = result_dict.get("higher_is_better", {}).get(k, {})

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        if sort_results:
            metric_items = sorted(metric_items)

        for (mf), v in metric_items:
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), "")

            if m + "_stderr" + "," + f in dic:
                se = dic[m + "_stderr" + "," + f]
                if se != "N/A":
                    se = "%.4f" % se
                if type(v) == float:
                    v = "%.4f" % v
                elif type(v) == tuple:  # patch by MF, allow tuple reporting
                    v = "/".join(["%.4f" % x for x in v])
                values.append([k, version, f, n, m, hib, v, "±", se])
            else:
                values.append([k, version, f, n, m, hib, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


def positional_deprecated(fn):
    """
    A decorator to nudge users into passing only keyword args (`kwargs`) to the
    wrapped function, `fn`.
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(
                f"WARNING: using {fn.__name__} with positional arguments is "
                "deprecated and will be disallowed in a future version of "
                "lm-evaluation-harness!"
            )
        return fn(*args, **kwargs)

    return _wrapper


def ignore_constructor(loader, node):
    return node


def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)

    *module_name, function_name = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)
    module_path = os.path.normpath(os.path.join(yaml_path, "{}.py".format(module_name)))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function


def load_yaml_config(yaml_path=None, yaml_config=None, yaml_dir=None, mode="full"):
    if mode == "simple":
        constructor_fn = ignore_constructor
    elif mode == "full":
        constructor_fn = import_function

    # Add the import_function constructor to the YAML loader
    yaml.add_constructor("!function", constructor_fn)
    if yaml_config is None:
        with open(yaml_path, "rb") as file:
            yaml_config = yaml.full_load(file)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    assert yaml_dir is not None

    if "include" in yaml_config:
        include_path = yaml_config["include"]
        del yaml_config["include"]

        if isinstance(include_path, str):
            include_path = [include_path]

        # Load from the last one first
        include_path.reverse()
        final_yaml_config = {}
        for path in include_path:
            # Assumes that path is a full path.
            # If not found, assume the included yaml
            # is in the same dir as the original yaml
            if not os.path.isfile(path):
                path = os.path.join(yaml_dir, path)

            try:
                included_yaml_config = load_yaml_config(yaml_path=path, mode=mode)
                final_yaml_config.update(included_yaml_config)
            except Exception as ex:
                # If failed to load, ignore
                raise ex

        final_yaml_config.update(yaml_config)
        return final_yaml_config
    return yaml_config


def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)


env = Environment(loader=BaseLoader, undefined=StrictUndefined, keep_trailing_newline=True)
env.filters["regex_replace"] = regex_replace


def apply_template(template: str, doc: dict) -> str:
    rtemplate = env.from_string(template)
    return rtemplate.render(**doc)


def create_iterator(raw_iterator, *, rank=0, world_size=1, limit=None):
    """
    Method for creating a (potentially) sliced and limited
    iterator from a raw document iterator. Used for splitting data
    among ranks in multigpu setting or only pulling a sample of documents
    """
    return islice(raw_iterator, rank, limit, world_size)


# Multi-token stopping criteria


# from more_itertools

def apply_chat_template(obj, chat_history):
    def map_segmented_string(raw_string: str, segmented_string: SegmentedString) -> Tuple[
        List[Tuple[int, int]], List[str], str]:
        """
        Method to map segmented string to raw string. The segments in_between should be labeled as "chat_template"
        """
        segment_offsets_wgaps = []
        segment_labels_wgaps = []
        start_search = 0
        for i, (segment, label) in enumerate(zip(segmented_string.segments, segmented_string.labels)):
            segment_start = raw_string.find(segment, start_search)
            # unfortunately, chat templates influence whitespacing
            # very rarely left/right space can disappear, if this is not whitespace segment
            # check if we havent found earlier segment without space
            if segment_start != -1 and segment.strip != "":
                possible_earlier_start = raw_string.find(segment.strip(), start_search)
                if possible_earlier_start != segment_start:
                    # check if lstrip or rstrip werent only necessary
                    possible_earlier_start_l = raw_string.find(segment.lstrip(), start_search)
                    possible_earlier_start_r = raw_string.find(segment.rstrip(), start_search)
                    if possible_earlier_start_l == possible_earlier_start:
                        # only lstrip was necessary
                        segment_start = possible_earlier_start
                        segment = segment.lstrip()
                    elif possible_earlier_start_r <= possible_earlier_start:
                        # only rstrip was necessary
                        segment_start = possible_earlier_start
                        segment = segment.rstrip()
                    else:
                        # full strip was necessary
                        segment_start = possible_earlier_start
                        segment = segment.strip()

            if segment_start == -1:
                # if segment not found, try to find it without leading/trailing whitespaces
                # chat template often removes leading/trailing whitespaces
                segment_start = raw_string.find(segment.rstrip(), start_search)
                if segment_start == -1:
                    segment_start = raw_string.find(segment.lstrip(), start_search)
                    if segment_start == -1:
                        segment_start = raw_string.find(segment.strip(), start_search)
                        assert segment_start != -1, f"Raw string: {raw_string}\n\nSegment {segment}\n(type: {label}) not found in raw string"
                        segment = segment.strip()
                    else:
                        segment = segment.lstrip()
                else:
                    segment = segment.rstrip()

            # if segment.strip() == "":
            #     # whitespace segments can dissapear in chat templates
            #     # so if they won't continue immediately after the previous segment, skip them
            #     if (len(segment_offsets_wgaps)==0 and segment_start!=0) or segment_start != segment_offsets_wgaps[-1][1]:
            #         continue
            segment_end = segment_start + len(segment)
            segment_offsets_wgaps.append((segment_start, segment_end))
            segment_labels_wgaps.append(label)
            assert segment_start >= start_search, "Segment start is before the end of the previous segment"
            start_search = segment_end
        # now check for the missing gaps between offsets
        # eg. for [(1,5), (10, 15)], we need to add (0, 1) and (5, 10)
        segment_offsets = []
        segment_labels = []

        if segment_offsets_wgaps[0][0] != 0:
            segment_offsets.insert(0, (0, segment_offsets_wgaps[0][0]))
            segment_labels.insert(0, "chat_template")
        if len(segment_offsets_wgaps) == 1:
            segment_offsets.append(segment_offsets_wgaps[0])
            segment_labels.append(segment_labels_wgaps[0])

        for i in range(len(segment_offsets_wgaps) - 1):
            # add the segment
            segment_offsets.append(segment_offsets_wgaps[i])
            segment_labels.append(segment_labels_wgaps[i])
            # check for the gap between i and i+1
            if segment_offsets_wgaps[i][1] != segment_offsets_wgaps[i + 1][0]:
                segment_offsets.append((segment_offsets_wgaps[i][1], segment_offsets_wgaps[i + 1][0]))
                segment_labels.append("chat_template")

        if len(segment_offsets_wgaps) > 1:
            # add the last segment
            segment_offsets.append(segment_offsets_wgaps[-1])
            segment_labels.append(segment_labels_wgaps[-1])

        return segment_offsets, segment_labels, raw_string

    # patch for SegmentedString
    raw_string_rendered = obj.tokenizer.apply_chat_template(
        chat_history, tokenize=False, add_generation_prompt=True
    )
    orig_raw_string = copy.deepcopy(raw_string_rendered)
    # map segmentedstring back to raw_string_rendered
    result = SegmentedString(("",))
    for h in chat_history:
        segment_offsets, segment_labels, raw_string_rendered = map_segmented_string(raw_string=raw_string_rendered,
                                                                                    segmented_string=h["content"])
        result += SegmentedString([raw_string_rendered[of_s:of_e] for of_s, of_e in segment_offsets],
                                  segment_labels)
        raw_string_rendered = raw_string_rendered[segment_offsets[-1][1]:]
    if raw_string_rendered != "":
        # if there is a gap between the last segment and the end of the string, it is a chat_template again
        result += SegmentedString([raw_string_rendered], ["chat_template"])
    assert orig_raw_string == str(result), "Mismatch between original raw string and mapped segmented string"
    return result

class SegmentedString(str):
    """
    A class that acts like string, but it also provides a way to work with it in segments.

    The eq, lt, and gt methods are the same as the ones for ordinary strings.
    Therefore, there is no consideration of the segments.

    Example usage:
    >>> s = SegmentedString(("abc", "def", "ghi"), labels=("first", "second", "third"))
    >>> s
    'abcdefghi'
    >>> s.segments
    ('abc', 'def', 'ghi')
    >>> s.labels
    ('first', 'second', 'third')

    """

    def __new__(cls, value: Union[str, Iterable[str]], labels: Optional[Iterable[Any]] = None) -> "SegmentedString":
        if isinstance(value, str):
            if labels is not None:
                raise ValueError("Labels can only be provided when value is a sequence of strings")

            segments = tuple([value])
        else:
            segments = tuple(value)
            value = "".join(value)

        instance = super().__new__(cls, value)
        instance._segments = segments if len(value) > 0 else ()
        instance._labels = tuple(labels) if labels is not None else None

        return instance

    def num_of_segments(self) -> int:
        """
        Get the number of segments in the string.
        """
        return len(self._segments)

    @property
    def segments(self) -> Tuple[str, ...]:
        return self._segments

    @property
    def labels(self) -> Optional[Tuple[Any, ...]]:
        return self._labels

    def __add__(self, other):
        if isinstance(other, SegmentedString):
            # as the labels are voluntary, we need to check if they are present, there can be four cases

            labels = None
            if self._labels is not None and other._labels is not None:
                labels = self._labels + other._labels
            elif self._labels is not None:
                labels = self._labels + (None,) * other.num_of_segments()
            elif other._labels is not None:
                labels = (None,) * self.num_of_segments() + other._labels

            return SegmentedString(self._segments + other._segments, labels)
        else:
            labels = self._labels
            if labels is not None:
                labels = labels + (None,)
            return SegmentedString(self._segments + (other,), labels)

    def __mul__(self, other):
        if isinstance(other, int):
            if other < 0:
                raise ValueError("Can only multiply SegmentedString by a non-negative integer")
            if other == 0:
                return SegmentedString("")

            return SegmentedString(self._segments * other, self._labels * other if self._labels is not None else None)
        else:
            raise ValueError("Can only multiply SegmentedString by an integer")

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = range(len(self))[item]
            indices_2_segment_indices = []

            for i, s in enumerate(self._segments):
                indices_2_segment_indices.extend([i] * len(s))

            segments = [[]]

            previous_segment_index = indices_2_segment_indices[indices[0]]
            labels = [self._labels[previous_segment_index]] if self._labels is not None else None

            for i in indices:
                segment_index = indices_2_segment_indices[i]
                if segment_index != previous_segment_index:
                    segments.append([])
                    previous_segment_index = segment_index
                    if labels is not None:
                        labels.append(self._labels[segment_index])
                segments[-1].append(super().__getitem__(i))

            segments = ["".join(s) for s in segments]

            return SegmentedString(segments, labels)
        else:
            return SegmentedString(
                (super().__getitem__(item),),
                (self._labels[self.char_2_segment_index(range(len(self))[item])],) if self._labels is not None else None
            )

    def char_2_segment_index(self, char_index: int) -> int:
        """
        Get the index of the segment that contains the character at the given index.

        Args:
            char_index: The index of the character.

        Returns:
            The index of the segment.
        """
        if char_index < 0:
            raise ValueError("The character index must be non-negative")
        if char_index >= len(self):
            raise ValueError("The character index is out of range")

        for i, segment in enumerate(self._segments):
            if char_index < len(segment):
                return i
            char_index -= len(segment)

    @staticmethod
    def _add_item(item: Union["SegmentedString", str], segments: List[str], labels: Optional[List[Any]]) -> Optional[
        List[Any]]:
        """
        Helper method for join method, that helps to assemble segments and labels.

        Args:
        - item: The item to be added.
        - segments: The list of segments, is modified in place.
        - labels: The list of labels, is modified in place if not None. If None, it is created and returned.

        Returns:
            labels
        """
        if isinstance(item, SegmentedString):
            if labels is None:
                if item._labels is not None:
                    labels = [None] * len(segments)
                    labels.extend(item._labels)
            else:
                if item._labels is not None:
                    labels.extend(item._labels)
                else:
                    labels.extend([None] * item.num_of_segments())
            segments.extend(
                item._segments)  # need to be after labels, as labels are edited according to previous length of segments
        else:
            segments.append(item)
            if labels is not None:
                labels.append(None)

        return labels

    def join(self, __iterable: Iterable[Union["SegmentedString", str]]) -> "SegmentedString":
        segments = []

        __iterable = iter(__iterable)
        try:
            labels = self._add_item(next(__iterable), segments, None)
        except StopIteration:
            return SegmentedString("")

        for item in __iterable:
            labels = self._add_item(self, segments, labels)
            labels = self._add_item(item, segments, labels)

        return SegmentedString(segments, labels)
