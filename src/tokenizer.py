from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def train_tokenizer(
    corpus,
    vocab_size,
    min_frequency,
    pretokenizer=Metaspace(),
    unk_token=UNK_TOKEN,
    pad_token=PAD_TOKEN,
    lowercase=False,
):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens (special tokens included)
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization
    pad_token: filler id used to make sequences in a batch equally long (id 0)
    lowercase: if True, "A man" and "a man" become the same tokens
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    if lowercase:
        tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = pretokenizer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[pad_token, unk_token],  # pad -> id 0, unk -> id 1
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer
