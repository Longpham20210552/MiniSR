import os
import gc
import argparse
from tqdm import tqdm
from scipy.special import softmax
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from datasets import load_metric

from model_src.configs.config import Config
from model_src.datasets.asr_dataset import ASRSliceDataset
from model_src.featurizers.speech_featurizers import TFSpeechFeaturizer
from model_src.featurizers.text_featurizers import SentencePieceFeaturizer
from model_src.models.conformer import ConformerCtc
from model_src.utils import env_util

# =============================
# Global setup
# =============================
DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")
CHECKPOINT_DIR = "./checkpoints"
LOSS_PLOT_DIR = "./loss_plots"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOSS_PLOT_DIR, exist_ok=True)

wer_metric = load_metric("wer")
logger = env_util.setup_environment()
tf.keras.backend.clear_session()


# =============================
# Argument parser
# =============================
def parse_arguments():
    parser = argparse.ArgumentParser(prog="Conformer Training/Testing")
    parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="Model config file")
    parser.add_argument("--mxp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--device", type=int, default=0, help="Device id")
    parser.add_argument("--cpu", action="store_true", help="Force CPU only")
    parser.add_argument("--saved", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--bs", type=int, default=None, help="Batch size")
    parser.add_argument("--input_padding", type=int, default=801)
    parser.add_argument("--label_padding", type=int, default=118)
    parser.add_argument("--beam_size", type=int, default=None, help="Beam size for decoding")
    return parser.parse_args()


# =============================
# Dataset preparation
# =============================
def build_datasets(config, speech_featurizer, text_featurizer, args):
    train_path = r"F:\\Luu_Dinh_Tu\\Project_2\\DATN\\datasets\\train_vlsp_vivos_processed.tsv"
    valid_path = r"F:\\Luu_Dinh_Tu\\Project_2\\DATN\\datasets\\valid_processed.tsv"
    test_path = r"F:\\Luu_Dinh_Tu\\Project_2\\DATN\\datasets\\test_processed.tsv"

    config.learning_config.train_dataset_config.data_paths = [train_path]
    config.learning_config.eval_dataset_config.data_paths = [valid_path]
    config.learning_config.test_dataset_config.data_paths = [test_path]

    train_dataset = ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        input_padding_length=args.input_padding,
        label_padding_length=args.label_padding,
        **vars(config.learning_config.train_dataset_config)
    )
    valid_dataset = ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        input_padding_length=args.input_padding,
        label_padding_length=args.label_padding,
        **vars(config.learning_config.eval_dataset_config)
    )
    test_dataset = ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        input_padding_length=args.input_padding,
        label_padding_length=args.label_padding,
        **vars(config.learning_config.test_dataset_config)
    )
    return train_dataset, valid_dataset, test_dataset


# =============================
# Model builder
# =============================
def build_conformer(config, vocab_size, featurizer_shape, augmentation_config=None):
    model = ConformerCtc(
        **config.model_config,
        vocabulary_size=vocab_size,
        augmentation_config=augmentation_config,
    )
    model.make(featurizer_shape)
    return model


def load_weights(conformer, config, args):
    temp_model = build_conformer(config, 128, conformer.speech_featurizer.shape)
    if args.saved:
        temp_model.load_weights(args.saved, by_name=True)
        conformer.encoder.set_weights(temp_model.encoder.get_weights())
        logger.info(f"Loaded weights from {args.saved}")
    else:
        logger.warning("Model initialized randomly.")
    del temp_model
    gc.collect()


# =============================
# Evaluation helpers
# =============================
def decode_predictions(logits, logits_len, text_featurizer, blank_id, beam_size=None):
    probs = softmax(logits)
    results = []

    if beam_size:
        beam = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits, perm=[1, 0, 2]),
            logits_len,
            beam_width=beam_size,
            top_paths=1,
        )
        beam = tf.sparse.to_dense(beam[0][0]).numpy()
    else:
        beam = None

    for i, (p, l) in enumerate(zip(probs, logits_len)):
        pred = p[:l].argmax(-1)
        decoded_prediction, prev = [], blank_id
        for token in pred:
            if (token != prev or prev == blank_id) and token != blank_id:
                decoded_prediction.append(token)
            prev = token
        if decoded_prediction:
            results.append(text_featurizer.iextract([decoded_prediction]).numpy()[0].decode("utf-8"))
        else:
            results.append("")
    return results


def evaluate(conformer, data_loader, text_featurizer, blank_id, beam_size=None):
    predictions, references = [], []
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(data_loader, total=len(data_loader)):
        labels, labels_len = batch[1]['labels'], batch[1]['labels_length']
        metrics = conformer.test_step(batch)

        total_loss += abs(metrics['loss'].numpy())
        n_batches += 1

        logits, logits_len = metrics["y_pred"]['logits'], metrics["y_pred"]['logits_length']
        decoded = decode_predictions(logits, logits_len, text_featurizer, blank_id, beam_size)

        for dec, label, ll in zip(decoded, labels, labels_len):
            label_len = tf.reduce_sum(tf.cast(label != 0, tf.int32))
            ref = text_featurizer.iextract([label[:label_len]]).numpy()[0].decode("utf-8")
            predictions.append(dec)
            references.append(ref)

    avg_loss = total_loss / max(1, n_batches)
    wer = wer_metric.compute(predictions=predictions, references=references)
    return avg_loss, wer


# =============================
# Training loop
# =============================
def train_and_evaluate(conformer, train_dataset, valid_dataset, test_dataset, text_featurizer, args):
    blank_id = text_featurizer.blank
    train_loader = train_dataset.create(args.bs)
    valid_loader = valid_dataset.create(args.bs)
    test_loader = test_dataset.create(args.bs)

    ckpt = tf.train.Checkpoint(optimizer=conformer.optimizer, model=conformer, epoch=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

    best_wer = float('inf')
    train_losses, valid_losses, test_wers, epochs = [], [], [], []

    for epoch in range(1, 301):
        logger.info(f"Epoch {epoch}/300")

        # Training
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            metrics = conformer.train_step(batch)
            total_loss += abs(metrics['loss'].numpy())
            n_batches += 1
        avg_train_loss = total_loss / max(1, n_batches)

        # Validation + Test
        val_loss, val_wer = evaluate(conformer, valid_loader, text_featurizer, blank_id, args.beam_size)
        _, test_wer = evaluate(conformer, test_loader, text_featurizer, blank_id, args.beam_size)

        logger.info(f"Train Loss={avg_train_loss:.4f} | Val Loss={val_loss:.4f} | Val WER={val_wer:.4f} | Test WER={test_wer:.4f}")

        train_losses.append(avg_train_loss)
        valid_losses.append(val_loss)
        test_wers.append(test_wer)
        epochs.append(epoch)

        # Save best checkpoint
        if val_wer < best_wer:
            best_wer = val_wer
            conformer.save_weights(os.path.join(CHECKPOINT_DIR, "best_ckpt"))
            logger.info(f"Saved new best checkpoint with WER={best_wer:.4f}")

        # Plot
        if epoch % 5 == 0:
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, train_losses, label="Train Loss")
            plt.plot(epochs, valid_losses, label="Valid Loss")
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(LOSS_PLOT_DIR, "loss_plot_latest.png"))

            plt.figure(figsize=(12, 6))
            plt.plot(epochs, test_wers, label="Test WER", marker='o')
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(LOSS_PLOT_DIR, "wer_plot_latest.png"))


# =============================
# Main
# =============================
def main():
    args = parse_arguments()
    config = Config(args.config)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})
    env_util.setup_devices([args.device], cpu=args.cpu)

    speech_featurizer = TFSpeechFeaturizer(config.speech_config)
    text_featurizer = SentencePieceFeaturizer(config.decoder_config)

    train_dataset, valid_dataset, test_dataset = build_datasets(config, speech_featurizer, text_featurizer, args)

    conformer = build_conformer(config, text_featurizer.num_classes, speech_featurizer.shape)
    conformer.add_featurizers(speech_featurizer, text_featurizer)
    load_weights(conformer, config, args)

    # Optimizer
    total_steps = 100 * len(train_dataset)
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.001,
        decay_steps=total_steps,
        end_learning_rate=1e-5,
        power=1.0
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    conformer.compile(optimizer=optimizer)

    train_and_evaluate(conformer, train_dataset, valid_dataset, test_dataset, text_featurizer, args)


if __name__ == "__main__":
    main()
