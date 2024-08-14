import copy
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from lm_eval import evaluator, tasks
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    Collator,
    pad_and_concat,
)

eval_logger = utils.eval_logger

STUB_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class DryrunLM(HFLM):

    def __init__(self):
        super().__init__(pretrained=STUB_MODEL, max_length=2048, truncate_strategy="leave_description")

    # def loglikelihood(self, requests):
    #     res = []
    #
    #     for ctx, cont in requests:
    #         res.append((-random.random(), False))
    #         self.tokencost += len(self.tokenizer.tokenize(ctx + cont))
    #
    #     return res
    #
    def _loglikelihood_tokens(
            self,
            requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
            disable_tqdm: bool = False,
            override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        res = []
        prev_ctx = None
        for _, ctx, cont in requests:
            res.append((-random.random(), False))
            if prev_ctx != ctx:
                self.tokencost_prefix += len(ctx + cont)
                self.tokencost_suffix += 2  # choice token + eos
            else:
                self.tokencost_prefix += len(cont) + 3  # assume 3 choice delimiter tokens

            prev_ctx = ctx

        return res

    def generate_until(
            self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for idx, chunk in enumerate(chunks):
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            self.tokencost_prefix+=context_enc.shape[1]

            fewshot_target_indices = [j for j, l in enumerate(requests[idx].args[0].labels) if l =='fewshot_target']
            fewshot_targets = [requests[idx].args[0].segments[j] for j in fewshot_target_indices]
            random_fewshot_target_length = len(self.tokenizer.tokenize(random.choice(fewshot_targets))[:max_gen_toks])

            self.tokencost_suffix += random_fewshot_target_length
            res.append("lol")


        return res


def main():
    lm = DryrunLM()

    # task_list = "benczechmark_summarization"
    # task_list = "benczechmark_agree, benczechmark_belebele, benczechmark_czechnews, benczechmark_snli, benczechmark_subjectivity, benczechmark_propaganda_argumentace, benczechmark_propaganda_fabulace, benczechmark_propaganda_nazor, benczechmark_propaganda_strach, benczechmark_propaganda_zamereni, benczechmark_propaganda_demonizace, benczechmark_propaganda_lokace, benczechmark_propaganda_relativizace, benczechmark_propaganda_vina, benczechmark_propaganda_zanr, benczechmark_propaganda_emoce, benczechmark_propaganda_nalepkovani, benczechmark_propaganda_rusko, benczechmark_sentiment_mall, benczechmark_sentiment_fb, benczechmark_sentiment_csfd, benczechmark_summarization, benczechmark_grammarerrorcorrection, benczechmark_cs_naturalquestions, benczechmark_cs_sqad32, benczechmark_cs_triviaQA, benczechmark_csfever_nli, benczechmark_ctkfacts_nli, benczechmark_cs_ner, benczechmark_hellaswag, benczechmark_klokan_qa, benczechmark_cs_court_decisions_ner, benczechmark_umimeto_qa, benczechmark_cermat_mc, benczechmark_cermat_qa, benczechmark_history_ir"
    task_list = ("benczechmark_propaganda_argumentace, benczechmark_propaganda_fabulace, benczechmark_propaganda_nazor, "
                 # "benczechmark_propaganda_strach, benczechmark_propaganda_zamereni, benczechmark_propaganda_demonizace, "
                 # "benczechmark_propaganda_lokace, benczechmark_propaganda_relativizace, benczechmark_propaganda_vina, "
                 # "benczechmark_propaganda_zanr, benczechmark_propaganda_emoce, benczechmark_propaganda_nalepkovani, "
                 # "benczechmark_propaganda_rusko, benczechmark_sentiment_mall, benczechmark_sentiment_fb, "
                 # "benczechmark_sentiment_csfd, "
                 "benczechmark_summarization, benczechmark_grammarerrorcorrection, "
                 "benczechmark_cs_naturalquestions, benczechmark_cs_sqad32, benczechmark_cs_triviaQA, "
                 "benczechmark_csfever_nli, benczechmark_ctkfacts_nli, benczechmark_cs_ner, benczechmark_hellaswag, "
                 "benczechmark_klokan_qa, benczechmark_cs_court_decisions_ner, benczechmark_umimeto_qa, "
                 "benczechmark_cermat_mc, benczechmark_cermat_qa, benczechmark_history_ir")

    values = []
    for taskname in task_list.split(", "):
        lm.tokencost_prefix = 0
        lm.tokencost_suffix = 0
        evaluator.simple_evaluate(
            model=lm,
            tasks=[taskname],
            num_fewshot=3,
            log_samples=False,
        )

        print(taskname, lm.tokencost_prefix, lm.tokencost_suffix)
        values.append(
            [
                taskname,
                lm.tokencost_prefix,
                lm.tokencost_suffix
            ]
        )
    from pytablewriter import MarkdownTableWriter

    writer = MarkdownTableWriter()
    writer.headers = ["Task", "Prefix Tokens", "Suffix Tokens"]

    values.sort(key=lambda x: -x[1])
    tot_pref_cost = sum([x[1] for x in values])
    tot_suf_cost= sum([x[2] for x in values])
    values.append(
        [
            "**Total**",
            tot_pref_cost,
            tot_suf_cost
        ]
    )

    writer.value_matrix = values

    print(writer.dumps())


if __name__ == "__main__":
    main()
