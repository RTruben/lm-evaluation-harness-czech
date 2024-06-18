from transformers import AutoTokenizer

from lm_eval.utils import SegmentedString

import datasets


class ContextSampler:
    def __init__(self, docs, task, fewshot_indices=None, rnd=None) -> None:
        self.rnd = rnd
        if not self.rnd:
            raise ValueError(
                "A `random.Random` generator argument must be provided to `rnd` of FewShotSampler!"
            )

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            if not isinstance(self.docs, datasets.Dataset):
                raise ValueError(
                    "Got `fewshot_indices` but fewshot_docs are not a HF dataset. Don't use both `fewshot_indices` and a user-defined few-shot sample list simultaneously"
                )
            self.docs = self.docs.select(fewshot_indices)

    def get_context(self, doc, num_fewshot):
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )

        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        fewshot_delimiter = SegmentedString((self.fewshot_delimiter,), ("fewshot_delimiter",))
        target_delimiter = SegmentedString((self.target_delimiter,), ("target_delimiter",))
        labeled_examples = SegmentedString(("",))  # Initialize an empty SegmentedString

        for doc in selected_docs:
            doc_content = self.doc_to_text(doc)
            doc_target = self.doc_to_target(doc)

            labeled_examples += SegmentedString((
                doc_content
                if self.config.doc_to_choice is None or isinstance(doc_content, str)
                else self.doc_to_choice(doc)[doc_content]
            ,), ("fewshot_text",))
            labeled_examples += target_delimiter

            labeled_examples += SegmentedString((
                str(doc_target[0])
                if isinstance(doc_target, list)
                else doc_target
                if self.config.doc_to_choice is None or isinstance(doc_target, str)
                else str(self.doc_to_choice(doc)[doc_target])
            ,), ("fewshot_target",))
            labeled_examples += fewshot_delimiter

        return labeled_examples

    def get_chat_context(
            self,
            doc,
            num_fewshot,
            fewshot_as_multiturn: bool = False,
    ):
        chat_history = []
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )
        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        if fewshot_as_multiturn:
            for doc in selected_docs:
                doc_content = self.doc_to_text(doc)
                doc_target = self.doc_to_target(doc)
                fewshot_text = (doc_content
                                if self.config.doc_to_choice is None
                                   or isinstance(doc_content, str)
                                else self.doc_to_choice(doc)[doc_content])
                # cast to SegmentedString
                fewshot_text = SegmentedString((fewshot_text,), ("fewshot_text",))
                chat_history.append(
                    {
                        "role": "user",
                        "content": fewshot_text,
                    }
                )
                fewshot_target = (str(doc_target[0])
                                  if isinstance(doc_target, list)
                                  else doc_target
                if self.config.doc_to_choice is None
                   or isinstance(doc_target, str)
                else str(self.doc_to_choice(doc)[doc_target]))
                # cast to SegmentedString
                fewshot_target = SegmentedString((fewshot_target,), ("fewshot_target",))

                chat_history.append(
                    {
                        "role": "assistant",
                        "content": fewshot_target,
                    }
                )
        else:
            # get fewshot context as one user turn
            chat_history.append(
                {"role": "user", "content": self.get_context(doc, num_fewshot)}
            )

        return chat_history

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.rnd.sample(self.docs, n)


class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert (
                n <= len(self.docs)
        ), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, n) -> None:
        """ """
        pass



SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        )
