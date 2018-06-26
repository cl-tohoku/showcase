# -*- coding: utf-8 -*-
from showcase.utils.inputs import RawTextProcessor
from showcase.utils.subfuncs import get_word_idx_path, load_word_idx


def test_convert():
    paragraph = ['彼がデートに誘った。', '明日は雨が東京に降ります．']
    word2idx = load_word_idx(get_word_idx_path())
    processor = RawTextProcessor()
    result, idx_hash, model_inputs = processor.convert(paragraph, word2idx)
    assert isinstance(result, list)
    assert isinstance(idx_hash, list)
    assert isinstance(idx_hash[0], dict)
    assert isinstance(model_inputs, list)
    assert isinstance(model_inputs[0], dict)


if __name__ == "__main__":
    test_convert()
