from transformers import T5Tokenizer
from utils import split_with_quotes


class CustomT5Tokenizer(T5Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_dict = {}
        self.inverse_vocab_dict = {2: "", 0: "", 1: ""}
        self.current_id = 3  # Start after the IDs used by the base T5 tokenizer

    def __call__(self, *args, **kwargs):
        # Call the parent class's __call__ method but use custom tokenization logic
        # Ensure that our tokenize, encode, and decode methods are used instead of the parent's.
        returns = super().__call__(*args, **kwargs)
        input_ids = []
        for text in args[0]:
            input_ids.append(self.encode(text))
        returns["input_ids"] = input_ids
        return returns

    def tokenize(self, text, **kwargs):
        words = split_with_quotes(text)  # Split the text into words
        tokens = []
        for word in words:
            if word not in self.vocab_dict:
                self.vocab_dict[word] = self.current_id
                self.inverse_vocab_dict[self.current_id] = word
                self.current_id += 1
            tokens.append(word)  # Return the word itself as the token
        return tokens

    def encode(self, text, **kwargs):
        # Convert tokens to IDs for encoding
        tokens = self.tokenize(text)
        token_ids = [self.vocab_dict[token] for token in tokens]
        return token_ids

    def decode(self, token_ids, **kwargs):
        # Decode the IDs back to the original tokens
        return " ".join(
            [self.inverse_vocab_dict[int(token_id)] for token_id in token_ids]
        )
