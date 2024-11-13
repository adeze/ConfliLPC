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
    set_seed
    )

from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from models_lpc.question_answering_model import QuestionAnsweringModel_LPC
import os
from transformers import BertConfig, BertForQuestionAnswering
from safetensors.torch import load_file


class QA_Finetune:
    
    def __init__(self, **kwargs):

        # Extract kwargs and update dictionary.
        self.__dict__.update(kwargs)

        self.model_path = self.model_local_path if self.model_local_path != None else self.model_huggingface_path
        self.dataset_path = self.dataset_local_path if self.dataset_local_path != None else self.dataset_huggingface_path
        self.task_id = self.task_id

        if self.model_path == None or self.dataset_path == None:
            print('A path for the model and QA dataset must be provided.')
            print(f'Provided model path: {self.model_path}')
            print(f'Provided QA dataset path: {self.dataset_path}')
            quit()

        # Check if CUDA is configured with pytorch
        gpu_id = self.check_gpu()

        # Setting global policy of tensorflow to use mixed float 16
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    def convert_to_qa_format(self, dataset_list):
        qa_examples = []
        for dataset in dataset_list:
            contexts = dataset["context"]
            questions = dataset["question"]
            ids = dataset["key"]
            answers = dataset["answers"]
            labels = dataset["labels"]

            for context, question, id_, answer, label_list in zip(contexts, questions, ids, answers, labels):
                qas = []
                if isinstance(answer, list):
                    answer_texts = answer
                else:
                    answer_texts = [answer]

                for answer_text, label in zip(answer_texts, label_list):
                    answer_start = label["start"][0] if isinstance(label["start"], list) else label["start"]
                    qas.append({
                        "question": question,
                        "id": id_,
                        "answers": [{
                            "text": answer_text if answer_text.strip() else '',
                            # Ensure answer_text is not just whitespace
                            "answer_start": answer_start if answer_text.strip() else 0
                        }]
                    })

                qa_examples.append({
                    "context": context,
                    "qas": qas
                })

        return qa_examples

    def preprocess_training_examples(self, examples):
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
            example_ids.append(examples["key"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs


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
            example_ids.append(examples["key"][sample_idx])

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

        if isinstance(s, list):
            s = " ".join(s)

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

    def exact_match_score(self, prediction, ground_truth):
        # Normalize and check for exact match
        norm_prediction = self.normalize_answer(prediction)
        norm_ground_truth = self.normalize_answer(ground_truth)
        return norm_prediction == norm_ground_truth

    def f1_score(self, prediction, ground_truth):
        # Normalize and tokenize predictions and ground truths
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

    def metric_max_over_ground_truths(self, metric_fn, predictions, ground_truths):
        # Get the best score among all predictions for all ground truths
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            scores_for_ground_truths.append(max(metric_fn(prediction, ground_truth) for prediction in predictions))
        return max(scores_for_ground_truths)

    def compute_score(self, dataset, predictions):
        f1 = exact_match = total = 0
        for article in dataset:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    total += 1
                    if qa["id"] not in predictions:
                        message = "Unanswered question " + qa["id"] + " will receive score 0."
                        print(message, file=sys.stderr)
                        continue
                    ground_truths = [x["text"] for x in qa["answers"]]
                    prediction = predictions[qa["id"]]

                    if isinstance(prediction, list):
                        exact_match += self.metric_max_over_ground_truths(self.exact_match_score, prediction,
                                                                          ground_truths)
                        f1 += self.metric_max_over_ground_truths(self.f1_score, prediction, ground_truths)
                    else:
                        exact_match += self.metric_max_over_ground_truths(self.exact_match_score, [prediction],
                                                                          ground_truths)
                        f1 += self.metric_max_over_ground_truths(self.f1_score, [prediction], ground_truths)

        exact_match = exact_match / total
        f1 = f1 / total

        print(f"Total: {total}, Exact match: {exact_match}, F1: {f1}")
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
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_path)

    def load_qa_dataset(self):
        if self.dataset_local_path != None:
            self.dataset_raw = load_from_disk(self.dataset_path)
        elif self.dataset_config_name == None:
            self.dataset_raw = load_dataset(self.dataset_path)
        else:
            self.dataset_raw = load_dataset(self.dataset_path, self.dataset_config_name)

        # Extract raw train and validation sets
        train_list = [self.dataset_raw['train']]
        validation_list = [self.dataset_raw['validation']]

        # Convert raw datasets to DataFrames before preprocessing
        train_df = self.convert_to_qa_format(train_list)
        eval_df = self.convert_to_qa_format(validation_list)

        return train_df, eval_df

    def QA_metric_compute(self, predictions, references):
        pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [{"text": answer["text"]} for answer in ref["answers"]],
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

    def fine_tune_all_seeds(self):
        self.load_tokenizer_and_model()
        train_examples, eval_examples = self.load_qa_dataset()

        for t in range(1, self.num_seeds + 1):
            if self.seed is None:
                random_seed = random.randint(1, 2 ** 31 - 1)
                print(f'Randomly generated seed for iteration {t}: {random_seed}')
                set_seed(random_seed)
            else:
                random_seed = self.seed
                set_seed(random_seed)
                print(f'Seed set as {random_seed}')

            model_args = QuestionAnsweringArgs()
            model_args.num_train_epochs = self.num_epochs
            model_args.train_batch_size = self.batch_size
            model_args.output_dir = self.output_dir
            model_args.best_model_dir = os.path.join(self.output_dir, "best_model", "")
            model_args.evaluate_during_training = True
            model_args.save_best_model = True
            model_args.fp16 = False

            model_args.learning_rate = self.learning_rate
            model_args.adam_epsilon = self.adam_epsilon
            model_args.optimizer = self.optimizer
            model_args.lpc_anneal_w = self.lpc_anneal_w
            model_args.reg_lambda = self.reg_lambda
            model_args.lpc_anneal_fun = self.lpc_anneal_fun
            model_args.lpc_anneal_k = self.lpc_anneal_k
            model_args.lpc_anneal_t0 = self.lpc_anneal_t0
            model_args.lpc_pretrain_cof = self.lpc_pretrain_cof
            model_args.update_epoch = self.update_epoch
            model_args.logits_calibraion_degree = self.logits_calibraion_degree

            model_args.task = self.task
            model_args.task_id = self.task_id

            if model_args.task_id == 0:
                self.prev_model = QuestionAnsweringModel(
                    "bert",
                    self.model_path,
                    args=model_args,
                    use_cuda=torch.cuda.is_available()
                )

                self.model = QuestionAnsweringModel_LPC(
                    "bert",
                    self.model_path,
                    args=model_args,
                    previous_model=self.prev_model,
                    use_cuda=torch.cuda.is_available()
                )
            else:
                self.prev_model = QuestionAnsweringModel(
                    "bert",
                    self.model_path,
                    args=model_args,
                    use_cuda=torch.cuda.is_available()
                )

                # Load the configuration file and modify num_labels
                config = BertConfig.from_pretrained(self.model_path)
                config.num_labels = 2  # Explicitly set num_labels to 2 for QA task

                # Initialize the model with the modified configuration
                model = BertForQuestionAnswering(config)

                # Load the pretrained weights, excluding the classifier layer
                model_weights_path = os.path.join(self.model_path, 'model.safetensors')
                if os.path.exists(model_weights_path):
                    checkpoint = load_file(model_weights_path, device="cpu")  # Load safetensors file
                else:
                    raise FileNotFoundError(f"Model weights file not found at {model_weights_path}")

                state_dict = {k: v for k, v in checkpoint.items() if not k.startswith("classifier")}
                model.load_state_dict(state_dict, strict=False)

                # Replace the model in qa_model with the modified model
                self.prev_model.model = model

                self.model = QuestionAnsweringModel_LPC(
                    "bert",
                    self.model_path,
                    args=model_args,
                    previous_model=self.prev_model,
                    use_cuda=torch.cuda.is_available()
                )

                # Replace the model in qa_model with the modified model
                self.model.model = model

            self.model.train_model(train_examples, eval_data=eval_examples)

            result, answers = self.model.eval_model(eval_examples)

            predictions, raw_outputs = self.model.predict(eval_examples)

            # Extract the predicted answers and theoretical answers
            predicted_answers = [
                {"id": prediction['id'], "prediction_text": prediction['answer']}
                for prediction in predictions
            ]

            theoretical_answers = []
            train_examples, eval_examples = self.load_qa_dataset()
            for example in eval_examples:
                for qa in example['qas']:
                    theoretical_answers.append({
                        "id": qa['id'],
                        "answers": qa['answers']
                    })

            total_blank_guesses = 0
            for prediction in predicted_answers:
                if not prediction['prediction_text']:  # Check if the prediction text is empty
                    total_blank_guesses += 1

            print(f'total blank guesses: {total_blank_guesses}')

            results = self.QA_metric_compute(predictions=predicted_answers, references=theoretical_answers)

            current_time_and_date = str(datetime.datetime.now())

            output_file_name = f'{self.final_model_name}_date_{current_time_and_date}'
            output_file_path = self.final_model_output_directory + output_file_name

            print(output_file_name)
            print(output_file_path)
            print(results)

            self.model.save_model(output_file_path)
            self.tokenizer.save_pretrained(output_file_path)

            log_dir = self.final_model_output_directory + 'log'

            with open(log_dir, 'a') as log_file:
                log_file.write(f'Date: {current_time_and_date}, ')
                log_file.write(f'Input Model Path: {self.model_path}, ')
                log_file.write(f'QA Dataset Path: {self.dataset_path}, ')
                log_file.write(f'Output Model Path: {output_file_path}, ')
                log_file.write(f'Trial: {t}, ')
                log_file.write(f'Seed: {random_seed}, ')
                log_file.write(f'Results: {results}, ')
                log_file.write(f'Learning Rate: {self.learning_rate}, ')
                log_file.write(f'Number of Epochs: {self.num_epochs}, ')
                log_file.write(f'Batch Size: {self.batch_size}, ')
                log_file.write(f'Language: {self.language}, ')
                log_file.write(f'Max Answer Length: {self.max_answer_length}, ')
                log_file.write(f'Max Context Length: {self.max_context_length}, ')
                log_file.write(f'Stride: {self.stride}\n')

            return results


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
    parser.add_argument('--num_epochs', required=True, type=int, help='Number of epochs for each GPU.')
    parser.add_argument('--learning_rate', required=True, type=float, help='Learning rate for model. Written in scientific notation. Ex: 5e-5')
    parser.add_argument('--num_seeds', required=True, type=int, help='Number of seeds for fine-tuning. Best performing seed is saved.')
    parser.add_argument('--batch_size', required=True, type=int, help='Size of batch for fine-tuning.')
    parser.add_argument('--seed', required=False, type=int, help='Exact seed to use.')

    parser.add_argument('--max_answer_length', default=30, type=int, help='Maximum answer length for a model\'s answer to a question.')
    parser.add_argument('--max_context_length', default=384, type=int, help='Maximum length for a context. Bert\'s limit & suggested length is 384.')
    parser.add_argument('--stride', default=128, type=int, help='Stride length for trimming size of QA dataset contexts.')

    return vars(parser.parse_args())  


def main():
    args_dict = parse_command_line_arguments()
    model_finetune_object = QA_Finetune(**args_dict)
    model_finetune_object.fine_tune_all_seeds()


if __name__ == "__main__":
    main()



