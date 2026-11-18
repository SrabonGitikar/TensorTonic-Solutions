class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        vocab = set()
        for sentence in texts:
            vocab.update(sentence.split())

        for sentence in sorted(vocab):
            sentence = sentence.lower()
            tokens = sentence.split()
            for token in tokens:
                if token in self.word_to_id:
                    continue 
                idx = len(self.word_to_id)
                self.word_to_id[token] = idx
                self.id_to_word[idx] = token

        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        ids = []
        words = text.split()
        for word in words:
            word = word.lower()
            ids.append(self.word_to_id.get(word, self.word_to_id[self.unk_token]))
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        words = [self.id_to_word.get(id, self.unk_token) for id in ids]
        return " ".join(words)