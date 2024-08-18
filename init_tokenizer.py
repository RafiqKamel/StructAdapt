from transformers import T5Tokenizer


class CustomT5Tokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.vocab = {}
        self.inverse_vocab = {}
        self.current_id = len(self.base_tokenizer)  # Start after pre-existing tokens

    def tokenize(self, text):
        words = text.split()
        tokens = []
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.current_id
                self.inverse_vocab[self.current_id] = word
                self.current_id += 1
            tokens.append(word)
        return tokens

    def detokenize(self, token_ids):
        words = [self.inverse_vocab[token_id] for token_id in token_ids]
        return " ".join(words)

    def encode(self, text, **kwargs):
        # Encode the text using custom tokenization
        token_ids = self.tokenize(text)
        return token_ids

    def decode(self, token_ids, **kwargs):
        # Decode the tokens back to text
        return self.detokenize(token_ids)

    def __len__(self):
        # Length of the tokenizer (number of tokens)
        return len(self.vocab) + len(self.base_tokenizer)

    def get_vocab(self):
        # Return the vocabulary
        return {**self.base_tokenizer.get_vocab(), **self.vocab}
