from dataclasses import dataclass, asdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from modal import Stub, Image, gpu, Volume
from helpers.data import format_dataset, score_prediction
from sentence_transformers import SentenceTransformer, models, evaluation, losses
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import product
import pandas as pd
import os
import json

gpu_config = gpu.A10G()
MODELS = [
    "jinaai/jina-embeddings-v2-small-en",
    "all-MiniLM-L12-v2",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-mpnet-base-v2",
]
DATASET_SIZE = [25600, 1600, 800, 12800, 200, 3200, 102400, 100, 51200, 6400, 400]
DENSE_OUT_FEATURES = [256, 512]
LEARNING_RATE = 0.0001
SCHEDULER = ["warmuplinear"]
WARMUP_STEPS = [500]
FREEZE_EMBEDDING_MODEL = [True]
BATCH_SIZE = [32]
MAX_EPOCHS = 8

# TODO: Revert test size back to 10000
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.from_name("modal-optimisation-volume", create_if_missing=True)
TEST_SET_SIZE = 100

METRICS = {
    "accuracy": accuracy_score,  # This is the number of correct predictions by the model ( TP + TN )/ (# of samples)
    "precision": precision_score,  # This measures the number of positive class predicitons (TP) / (TP + FP)
    "recall": recall_score,  # This measure the number of negative class predictions (TP) / ( TP + FN )
    "AUC": roc_auc_score,
}


stub = Stub("modal-optimization")


def download_model():
    from sentence_transformers import SentenceTransformer

    for model in MODELS:
        SentenceTransformer(model)


image = (
    Image.debian_slim()
    .pip_install("sentence-transformers", "torch", "datasets")
    .run_function(download_model)
)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    dataset_size: int
    dense_out_features: int
    learning_rate: float
    scheduler: str
    warmup_steps: int
    freeze_embedding_model: bool
    batch_size: int
    num_epochs: int


def random_search_config(model_name, dataset_size, freeze_embedding_model):
    """
    Randomly sample from the configuration space
    """
    import random

    scheduler = random.choice(SCHEDULER)
    warmup_steps = random.choice(WARMUP_STEPS)
    batch_size = random.choice(BATCH_SIZE)
    learning_rate = LEARNING_RATE  # This could also be made configurable if desired
    dense_out_features = random.choice(DENSE_OUT_FEATURES)
    num_epochs = MAX_EPOCHS  # This could also be made configurable if desired

    return ModelConfig(
        model_name=model_name,
        dataset_size=dataset_size,
        freeze_embedding_model=freeze_embedding_model,
        dense_out_features=dense_out_features,
        learning_rate=learning_rate,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


def initialise_model(model_config: ModelConfig) -> SentenceTransformer:
    embedding_model = SentenceTransformer(model_config.model_name)
    # Model configuration
    dim_emb = embedding_model.get_sentence_embedding_dimension()

    # Freeze the embedding model
    if model_config.freeze_embedding_model:
        for param in embedding_model._first_module().auto_model.parameters():
            param.requires_grad = False

    # Define the model architecture with additional dense layer
    dense_model = models.Dense(
        in_features=dim_emb,
        out_features=model_config.dense_out_features,
        activation_function=nn.Tanh(),
    )
    pooling_model = models.Pooling(dim_emb)

    # Initialize the model
    return SentenceTransformer(
        modules=[embedding_model, pooling_model, dense_model], device="cuda"
    )


@stub.function(
    image=image,
    gpu=gpu_config,
    timeout=86400,
    volumes={DATASET_DIR: DATASET_VOLUME},
)
def objective(config: ModelConfig):
    from datasets import load_from_disk

    # Load Dataset
    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    train_dataset = dataset["train"].select(range(config.dataset_size))
    test_dataset = dataset["test"].select(range(TEST_SET_SIZE))

    train_examples, test_examples = (
        format_dataset(train_dataset),
        format_dataset(test_dataset),
    )

    # Load Model
    model = initialise_model(config)

    # Create Dataloaders
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=config.batch_size
    )
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples, batch_size=config.batch_size
    )
    train_loss = losses.OnlineContrastiveLoss(model)

    # Generate a unique identifier for the model configuration
    model_config_hash = hash(config)

    MODEL_SAVE_PATH = f"/output/{model_config_hash}"

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        warmup_steps=config.warmup_steps,
        scheduler=config.scheduler,
        optimizer_params={"lr": config.learning_rate},
        save_best_model=True,
        epochs=config.num_epochs,
        output_path=MODEL_SAVE_PATH,
    )

    predictions, test_labels = score_prediction(model, train_dataset, test_dataset)
    eval_results = {
        f"metric_{metric}": round(function(test_labels, predictions), 4)
        for metric, function in METRICS.items()
    }
    eval_results = {**eval_results, **asdict(config)}

    print(json.dumps(eval_results, indent=2))
    return eval_results


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME})
def download_dataset():
    from datasets import load_dataset

    dataset_path = f"{DATASET_DIR}/{DATASET_NAME}"

    if os.path.exists(dataset_path):
        print("Dataset Exists")
        return

    dataset = load_dataset(DATASET_NAME)

    dataset.save_to_disk(dataset_path)
    DATASET_VOLUME.commit()


def generate_configs(n_trials):
    count = 3
    for model, sample_size, freeze_embedding_model in product(
        MODELS, DATASET_SIZE, [True, False]
    ):
        if count < 0:
            break
        for _ in range(n_trials):
            count -= 1
            if count == 0:
                break
            yield random_search_config(model, sample_size, freeze_embedding_model)


@stub.local_entrypoint()
def main():
    import time

    date = time.strftime("%Y-%m-%d-%H-%M")
    results = []

    download_dataset.remote()

    for experiment_result in objective.map(
        generate_configs(n_trials=1), order_outputs=True, return_exceptions=True
    ):
        if isinstance(experiment_result, Exception):
            print(f"Encountered Exception of {experiment_result}")

        results.append(experiment_result)
        # dumb but... save the results to a file every time a new result is available
        # This is to ensure that the results are not lost if the job is interrupted
        df = pd.DataFrame(results).sort_values("metric_accuracy", ascending=False)
        df.to_csv(f"./paramsearch/{date}_plain_trial_results.csv", index=False)

        # Save the results to a markdown file, this is useful for viewing the results in a human readable format
        with open(f"./paramsearch/{date}_plain_trial_results.md", "w") as f:
            f.write(df.to_markdown())
