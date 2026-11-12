from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace


def train_tokenizer(
    corpus, vocab_size, min_frequency, pretokenizer=Metaspace(), unk_token="[UNK]"
):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pretokenizer
    trainer = BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[unk_token]
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer
