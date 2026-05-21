from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


class TranslitDataset(Dataset):
    def __init__(
        self,
        pairs: list[dict],
        tokenizer: AutoTokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
        input_field: str = "roman_text",
        target_field: str = "telugu_text",
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.input_field = input_field
        self.target_field = target_field

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        roman = pair.get(self.input_field) or ""
        telugu = pair.get(self.target_field) or ""

        model_inputs = self.tokenizer(
            roman,
            text_target=telugu,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = {"input_ids": model_inputs.pop("labels")}

        label_ids = labels["input_ids"].squeeze()
        # Replace padding token id with -100 so it's ignored in loss
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": label_ids,
        }


def load_jsonl(path: str) -> list[dict]:
    import json
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs
