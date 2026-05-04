import torch
import csv
from torch.utils.data import Dataset
from constant import *

class SequenceDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer, device):
        self.data_dict = []
        self.device = device
        self.label_set = set()
        self.tokenizer = tokenizer
        self.tag2id = TAG2ID

        for file in dataset_file_path:
            with open(file, newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.DictReader(csvfile)

                for row in csv_reader:
                    response_id = str(row["ResponseId"]).strip() if "ResponseId" in row and row["ResponseId"] else "NA"
                    student_answer = str(row["ResponseText.x"]).strip()
                    label = str(row["ground_truth"]).strip()
                    question_text = str(row["Question"]).strip()
                    reference_answer = str(row["Model_Answer"]).strip()
                    rubric = str(row["Rubric"]).strip()

                    identifier = str(row["TaskPrompt"]).strip() if "TaskPrompt" in row and row["TaskPrompt"] else "NA"

                    # store university and question for output file
                    university = str(row["UNIV"]).strip() if "UNIV" in row and row["UNIV"] else "NA"

                    # include rubric as an additional feature
                    #line = CLS_TOKEN + student_answer + SEP_TOKEN + reference_answer #+ SEP_TOKEN + rubric question_text + SEP_TOKEN
                    #line = CLS_TOKEN + question_text + SEP_TOKEN + student_answer + SEP_TOKEN + reference_answer
                    #line = CLS_TOKEN + question_text + SEP_TOKEN + student_answer + SEP_TOKEN + reference_answer + SEP_TOKEN + rubric
                    line = CLS_TOKEN + question_text + SEP_TOKEN + student_answer + SEP_TOKEN + reference_answer


                    self.label_set.add(label)
                    self.data_dict.append({
                        "label": label,
                        "line": line,
                        "identifier": identifier,
                        "question": question_text,
                        "university": university,
                        "response_id": response_id,   # ADD THIS
                    })

        print(self.tag2id)
        print(self.get_category_distribution())

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        item = self.data_dict[index]
        label = self.tag2id[item["label"]]
        line = item["line"]
        identifier = item["identifier"]
        question = item["question"]
        university = item["university"]
        response_id = item["response_id"]


        tokenized_data = self.tokenizer(
            line,
            padding="max_length",
            truncation=True,
            max_length=hyperparameters["max_length"]
        )

        input_ids = tokenized_data["input_ids"]
        attention_mask = tokenized_data["attention_mask"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "identifier": identifier,
            "question": question,
            "university": university,
            "task_prompt": identifier,
            "response_id": response_id,   # ADD THIS
        }

    def get_category_distribution(self):
        cat_count = {}
        for item in self.data_dict:
            cat = item["label"]
            if cat not in cat_count:
                cat_count[cat] = 0
            cat_count[cat] += 1
        return cat_count
