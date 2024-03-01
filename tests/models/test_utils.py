# -*- coding: UTF-8 -*-
"""
Created on 29.02.24

:author:     Martin Dočekal
"""
from unittest.mock import MagicMock

from transformers import AutoTokenizer

from lm_eval.models.utils import truncate_token_segments_from_left, segmented_tok_encode
from lm_eval.utils import SegmentedString


def test_truncate_token_segments_from_left_enough_space():
    assert truncate_token_segments_from_left(
        [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
        20) == [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]

    assert truncate_token_segments_from_left(
        [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
        10) == [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]


def test_truncate_token_segments_from_left_truncate_whole_segments():
    assert truncate_token_segments_from_left(
        [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
        7) == [[3, 4], [5, 6, 7, 8, 9]]

    assert truncate_token_segments_from_left(
        [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
        5) == [[5, 6, 7, 8, 9]]


def test_truncate_token_segments_from_left_truncate_part_of_segments():
    assert truncate_token_segments_from_left(
        [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
        8) == [[2], [3, 4], [5, 6, 7, 8, 9]]

    assert truncate_token_segments_from_left(
        [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
        6) == [[4], [5, 6, 7, 8, 9]]

    assert truncate_token_segments_from_left(
        [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
        4) == [[6, 7, 8, 9]]

    assert truncate_token_segments_from_left(
        [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
        1) == [[9]]


class TestSegmentedTokEncode:

    def setup_method(self):
        self.mock_tokenizer = MagicMock(AutoTokenizer)
        self.mock_tokenizer.return_value = {
            "input_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "offset_mapping": [
                (0, 3), (4, 9), (10, 17), (18, 24), (25, 32), (33, 37), (38, 41), (42, 45), (46, 51), (52, 57)
            ]
        }

        self.segmented_string = SegmentedString(
            ("the first segment ", "second segment ", "last but not least third"),
            ("description", "second", "third")
        )

    def test_enough_space(self):
        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            20,
            None) == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]], ["description", "second", "third"])

        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            10,
            None) == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]], ["description", "second", "third"])

    def test_truncate_whole_segments(self):
        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            7,
            None) == ([3, 4, 5, 6, 7, 8, 9], [[3, 4], [5, 6, 7, 8, 9]], ["second", "third"])

        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            5,
            None) == ([5, 6, 7, 8, 9], [[5, 6, 7, 8, 9]], ["third"])

    def test_truncate_part_of_segments(self):
        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            8,
            None) == ([2, 3, 4, 5, 6, 7, 8, 9], [[2], [3, 4], [5, 6, 7, 8, 9]], ["description", "second", "third"])

        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            6,
            None) == ([4, 5, 6, 7, 8, 9], [[4], [5, 6, 7, 8, 9]], ["second", "third"])

        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            4,
            None) == ([6, 7, 8, 9], [[6, 7, 8, 9]], ["third"])

        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            1,
            None) == ([9], [[9]], ["third"])

    def test_enough_space_leave_description(self):
        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            20,
            "leave_description") == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]], ["description", "second", "third"])

        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            10,
            "leave_description") == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]], ["description", "second", "third"])

    def test_truncate_whole_segments_leave_description(self):
        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            9,
            "leave_description") == ([0, 1, 2, 4, 5, 6, 7, 8, 9], [[0, 1, 2], [4], [5, 6, 7, 8, 9]], ["description", "second", "third"])

        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            7,
            "leave_description") == ([0, 1, 2, 6, 7, 8, 9], [[0, 1, 2], [6, 7, 8, 9]], ["description", "third"])

    def test_truncate_whole_segments_leave_description_description_too_long(self):
        assert segmented_tok_encode(
            self.segmented_string,
            self.mock_tokenizer,
            2,
            "leave_description") == ([8, 9], [[8, 9]], ["third"])