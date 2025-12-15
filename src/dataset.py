import tiktoken

class Dataset:
    def __init__(self, text):
        self.text = text

    def encode(self):
        encoder = tiktoken.get_encoding("gpt2")
        return encoder.encode(text=self.text)