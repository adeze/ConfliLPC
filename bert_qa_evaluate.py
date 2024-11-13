print('Importing argparse...')
import argparse
print('Importing numpy...')
import numpy as np
print('Importing collections...')
import collections

print('Importing datasets...')
from datasets import load_dataset, load_from_disk
print('Importing tqdm...')
from tqdm.auto import tqdm

print('Importing re, string, sys, collections, random, and datetime......')
import re
import string
import sys
from collections import Counter
import random
import datetime

print('Importing torch...')
import torch
print('Importing tensorflow...')
import tensorflow as tf

print('Importing transformers...')
from transformers import (
    TFAutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DefaultDataCollator,
    create_optimizer,
    set_seed,
    BertForMaskedLM
    )


class QA_Evaluate:
    
    def __init__(self, **kwargs):

        # Extract kwargs and update dictionary.
        self.__dict__.update(kwargs)

        self.model_path = self.model_local_path if self.model_local_path != None else self.model_huggingface_path
        self.dataset_path = self.dataset_local_path if self.dataset_local_path != None else self.dataset_huggingface_path

        if self.model_path == None or self.dataset_path == None:
            print('A path for the model and QA dataset must be provided.')
            print(f'Provided model path: {self.model_path}')
            print(f'Provided QA dataset path: {self.dataset_path}')
            quit()

        # Check if CUDA is configured with pytorch
        gpu_id = self.check_gpu()

        # Setting global policy of tensorflow to use mixed float 16
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


    def preprocess_validation_examples(self, examples):

        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_context_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs


    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            if self.language == 'english':
                return re.sub(r"\b(a|an|the)\b", " ", text)
            elif self.language == 'spanish':
                return re.sub(r"\b(la|las|una|unas|el|los|un|unos|lo)\b", " ", text)
            else:
                return text

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def get_predicted_and_theoretical_answers(self, start_logits, end_logits, features, examples):

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)


        n_best = 20
        max_answer_length = 30
        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return predicted_answers, theoretical_answers


    def f1_score(self, prediction, ground_truth):

        # Checking for ground truth = ''
        # if len(ground_truth) == 1 and ground_truth[0] == '':
        #     self.exact_match_score(prediction, ground_truth)

        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def exact_match_score(self, prediction, ground_truth):
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)


    def metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)

        return max(scores_for_ground_truths)


    def compute_score(self, dataset, predictions):
        f1 = exact_match = total = 0
        for article in dataset:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    total += 1
                    try:
                        if qa["id"] not in predictions:
                            message = "Unanswered question " + qa["id"] + " will receive score 0."
                            print(message, file=sys.stderr)
                            continue
                        ground_truths = list(map(lambda x: x["text"], qa["answers"]))

                        # handle empty ground truths.
                        # if len(ground_truths) == 0:
                        #     ground_truths = ['']

                        prediction = predictions[qa["id"]]
                        exact_match += self.metric_max_over_ground_truths(self.exact_match_score, prediction, ground_truths)
                        f1 += self.metric_max_over_ground_truths(self.f1_score, prediction, ground_truths)
                    except Exception as e:
                        print(e)
                        print(article)
                        print(paragraph)
                        print(qa)
                        quit()

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {"exact_match": exact_match, "f1": f1}


    def check_gpu(self):

        # See if gpu is reachable by torch.
        is_gpu_available = torch.cuda.is_available()
        print(f'Is CUDA configured with Pytorch? {is_gpu_available}')

        if not is_gpu_available:
            print('GPU is not configured. Exiting...')
            quit()

        print(f'CUDA version: {torch.version.cuda}')

        gpu_id = torch.cuda.current_device()
        print(f'GPU: {torch.cuda.get_device_name(gpu_id)}')
        print(f'GPU id: {gpu_id}')

        return gpu_id
    

    def load_tokenizer_and_model(self):

        # Load tokenizer and model from directory
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = BertForMaskedLM.from_pretrained(self.model_path, from_tf=True)


    def load_qa_dataset(self):

        if self.dataset_local_path != None:
            self.dataset_raw = load_from_disk(self.dataset_path)
        elif self.dataset_config_name == None:
            self.dataset_raw = load_dataset(self.dataset_path)
        else:
            self.dataset_raw = load_dataset(self.dataset_path, self.dataset_config_name)

        try:
            self.dataset_validation_raw = self.dataset_raw['evaluation'].map(self.preprocess_validation_examples,batched=True,remove_columns=self.dataset_raw['evaluation'].column_names,)
        except Exception as e:
            print('Evaluation column not found... Using validation column instead.')
            self.dataset_validation_raw = self.dataset_raw['validation'].map(self.preprocess_validation_examples,batched=True,remove_columns=self.dataset_raw['validation'].column_names,)



    def QA_metric_compute(self, predictions, references):
        pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [{"text": answer_text} for answer_text in ref["answers"]["text"]],
                                "id": ref["id"],
                            }
                            for ref in references
                        ]
                    }
                ]
            }
        ]
        score = self.compute_score(dataset=dataset, predictions=pred_dict)
        return score


    def evaluate_model(self):

        self.load_tokenizer_and_model()
        self.load_qa_dataset()
            
        # randomly generate seed and print.
        random_seed = random.randint(1,2**31-1)
        print(f'Randomly generated seed: {random_seed}')
        set_seed(random_seed)


        # Preparing model for QA
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(self.model_path,from_pt=False)

        # Preparing datasets for use by converting to tensors:
        self.data_collator = DefaultDataCollator(return_tensors='tf')

        tf_eval_dataset = self.model.prepare_tf_dataset(
            self.dataset_validation_raw,
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=self.batch_size,
        )

        # Predict with fitted model
        predictions = self.model.predict(tf_eval_dataset)

        # Compute metrics
        try:
            predicted_answers, theoretical_answers = self.get_predicted_and_theoretical_answers(
                                                        predictions["start_logits"],
                                                        predictions["end_logits"],
                                                        self.dataset_validation_raw,
                                                        self.dataset_raw["validation"])
        except Exception as e:
            predicted_answers, theoretical_answers = self.get_predicted_and_theoretical_answers(
                                                    predictions["start_logits"],
                                                    predictions["end_logits"],
                                                    self.dataset_validation_raw,
                                                    self.dataset_raw["evaluation"])
        
        results = self.QA_metric_compute(predictions=predicted_answers,references=theoretical_answers)

        current_time_and_date = str(datetime.datetime.now())

        output_file_name = f'{self.final_model_name}_date_{current_time_and_date}' 
        output_file_path = self.final_model_output_directory + output_file_name

        print(output_file_name)
        print(output_file_path)
        print(results)

        log_dir = self.final_model_output_directory + 'log'

        with open(log_dir, 'a') as log_file:
            log_file.write(f'Date: {current_time_and_date}, ')
            log_file.write(f'Input Model Path: {self.model_path}, ')
            log_file.write(f'QA Dataset Path: {self.dataset_path}, ')
            log_file.write(f'Seed: {random_seed}, ')
            log_file.write(f'Results: {results}, ')
            log_file.write(f'Batch Size: {self.batch_size}, ')
            log_file.write(f'Language: {self.language}, ')
            log_file.write(f'Max Answer Length: {self.max_answer_length}, ')
            log_file.write(f'Max Context Length: {self.max_context_length}, ')
            log_file.write(f'Stride: {self.stride}\n')

                


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    
    # Path specific arguments
    parser.add_argument('--model_local_path', type=str, help='Path to model, if the model is stored locally.')
    parser.add_argument('--model_huggingface_path', type=str, help='Path to model, if the model is stored on huggingface.')
    
    parser.add_argument('--dataset_local_path', type=str, help='Path to QA dataset, if the QA dataset is stored locally.')
    parser.add_argument('--dataset_huggingface_path', type=str, help='Path to QA dataset, if the QA dataset is stored on huggingface.')
    parser.add_argument('--dataset_config_name', type=str, help='Some datasets require a config name, usually a version.')
    
    parser.add_argument('--final_model_name', required=True, type=str, help='Name of model after fine-tuning is completed.')
    parser.add_argument('--final_model_output_directory', required=True, type=str, help='Output directory for model after fine-tuning is completed. End with \'\\\'')

    # Model parameter arguments
    parser.add_argument('--language', required=True, type=str, help='Language that the model and dataset are in.')
    parser.add_argument('--batch_size', required=True, type=int, help='Size of batch for fine-tuning.')

    parser.add_argument('--max_answer_length', default=30, type=int, help='Maximum answer length for a model\'s answer to a question.')
    parser.add_argument('--max_context_length', default=384, type=int, help='Maximum length for a context. Bert\'s limit & suggested length is 384.')
    parser.add_argument('--stride', default=128, type=int, help='Stride length for trimming size of QA dataset contexts.')

    return vars(parser.parse_args())  


def main():
    args_dict = parse_command_line_arguments()
    model_finetune_object = QA_Evaluate(**args_dict)
    model_finetune_object.evaluate_model()


if __name__ == "__main__":
    main()



