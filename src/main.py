
import requests
from dataset import Dataset
import config

if __name__ == "__main__":
    text = requests.get(config.URL)

    ds = Dataset

    print(text.text[:1000])
    dataset = ds.encode(text)

    print(dataset[:50])
