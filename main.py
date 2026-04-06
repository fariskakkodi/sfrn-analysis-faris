import argparse
#import wandb
import os
import csv
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import cohen_kappa_score
from torch.optim import AdamW
from constant import *
from DataModules import SequenceDataset
from SFRN_model import SFRNModel  # import the custom SFRN model

best_ckp_path = None  # global variable for saving the best checkpoint path


def train(args):
    # define the directory for saving checkpoints
    checkpoint_dir = './checkpoints'

    # set a random seed for reproducibility
    random.seed(hyperparameters['random_seed'])

    # initialize best accuracy and f1 score trackers
    best_acc, best_f1 = 0, 0
    global best_ckp_path

    # get the training device (e.g., 'cuda:0' or 'cpu')
    DEVICE = args.device
    print(DEVICE)

    # initialize tokenizer for the model
    model_name = hyperparameters['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load the training, validation, and testing datasets
    train_dataset = SequenceDataset(TRAIN_FILE_PATH, tokenizer, DEVICE)
    val_dataset = SequenceDataset(VAL_FILE_PATH, tokenizer, DEVICE)
    test_dataset = SequenceDataset(TEST_FILE_PATH, tokenizer, DEVICE)

    val_dataset.tag2id = train_dataset.tag2id
    test_dataset.tag2id = train_dataset.tag2id

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # FIX: initialize SFRNModel instead of AutoModelForSequenceClassification
    model = SFRNModel()
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    num_training_steps = len(train_loader) * hyperparameters['epochs']
    warmup_steps = int(hyperparameters['WARMUP_STEPS'] * num_training_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    # main training loop
    for epoch in range(hyperparameters['epochs']):
        model.train()
        train_loss = 0.0
        y_true, y_pred, identifiers = [], [], []  # trackers for metrics
        train_iterator = tqdm(train_loader, desc="Train Iteration")  # progress bar

        for step, batch in enumerate(train_iterator):
            # extract input and labels from the batch
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            identifiers.extend(batch["identifier"])

            # FIX: SFRNModel returns logits directly, compute loss manually
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            # track loss and predictions
            train_loss += loss.item()
            pred_idx = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred_idx.cpu().numpy())

            # apply gradient clipping and optimizer step
            nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['max_norm'])
            if (step + 1) % hyperparameters['GRADIENT_ACCUMULATION_STEPS'] == 0:
                optimizer.step()
                model.zero_grad()
                scheduler.step()

        # calculate training metrics
        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        train_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        print(f'Epoch {epoch + 1} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f}')

        # validation loop
        model.eval()
        val_loss = 0.0
        val_y_true, val_y_pred = [], []
        val_iterator = tqdm(val_loader, desc="Validation Iteration")
        with torch.no_grad():
            for step, batch in enumerate(val_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                # FIX: SFRNModel returns logits directly, compute loss manually
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                pred_idx = torch.argmax(logits, dim=1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(pred_idx.cpu().numpy())

            # calculate validation metrics
            val_acc = accuracy_score(val_y_true, val_y_pred)
            val_f1 = f1_score(val_y_true, val_y_pred, average='macro')
            val_qwk = cohen_kappa_score(val_y_true, val_y_pred, weights='quadratic')
            print(f'Validation - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Loss: {val_loss:.4f}')

            # save best model checkpoint
            if val_acc > best_acc or val_f1 > best_f1:
                best_acc, best_f1 = val_acc, val_f1
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{args.ckp_name}_epoch{epoch}.model')
                torch.save(model.state_dict(), checkpoint_path)
                best_ckp_path = checkpoint_path

    # testing the best model
    if best_ckp_path:
        model.load_state_dict(torch.load(best_ckp_path, map_location=DEVICE))
        model.eval()
        test_y_true, test_y_pred = [], []
        test_questions, test_universities = [], []
        test_response_ids = []
        test_iterator = tqdm(test_loader, desc="Test Iteration")
        with torch.no_grad():
            for batch in test_iterator:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                # FIX: SFRNModel returns logits directly
                logits = model(input_ids, attention_mask=attention_mask)
                pred_idx = torch.argmax(logits, dim=1)
                test_y_true.extend(labels.cpu().numpy())
                test_y_pred.extend(pred_idx.cpu().numpy())
                test_response_ids.extend(batch["response_id"])
                # collect question and university for output file
                test_questions.extend(batch["task_prompt"])
                test_universities.extend(batch["university"])

        # calculate and print test metrics
        test_acc = accuracy_score(test_y_true, test_y_pred)
        test_f1 = f1_score(test_y_true, test_y_pred, average='macro')
        test_qwk = cohen_kappa_score(test_y_true, test_y_pred, weights='quadratic')
        print(f'Test - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, QWK: {test_qwk:.4f}')

        # write test output CSV with question, university, true label, predicted label
        output_path = './sfrn+longformer_a+m.csv'
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['response_id', 'question', 'university', 'true_label', 'predicted_label'])
            writer.writeheader()
            for rid, q, univ, true, pred in zip(test_response_ids, test_questions, test_universities, test_y_true, test_y_pred):
                writer.writerow({
                    'response_id': rid,
                    'question': q,
                    'university': univ,
                    'true_label': true,
                    'predicted_label': pred,
                })
        print(f'Test output saved to {output_path}')


def main():
    # parse command-line arguments for model checkpoint and device
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='debug_cpt', help='Checkpoint name for saving the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device: cuda or cpu')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
