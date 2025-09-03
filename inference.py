"""
Online Speech Recognition Demo with Conformer + KenLM Language Model
-------------------------------------------------------------------
- Nhấn 'r' để bắt đầu ghi âm, thả 'r' để dừng.
- Hệ thống nhận dạng online, xử lý theo từng khung 1.75s.
- Hỗ trợ xử lý các đoạn ghi âm dài hơn 8s.
"""

import os
import re
import time
import warnings
import argparse
import threading
import numpy as np
import scipy.io.wavfile
import tensorflow as tf
from scipy.special import softmax
from pyctcdecode import build_ctcdecoder

# Suppress TF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------------------------------------------------------
# 1. Load Unigram List for KenLM
# ----------------------------------------------------------------------
with open(r"/mnt/c/Users/Phạm Quý Long/Download/Project_2/Squeezeformer-main/all_text.txt", encoding="utf-8") as f:
    text = f.read().lower()
text = re.sub(r"[^\w\s]", "", text)
unigram_list = sorted(set(text.split()))

# ----------------------------------------------------------------------
# 2. TensorFlow + Config
# ----------------------------------------------------------------------
from model_src.configs.config import Config
from model_src.datasets.asr_dataset import ASRSliceDataset
from model_src.featurizers.speech_featurizers import TFSpeechFeaturizer
from model_src.featurizers.text_featurizers import SentencePieceFeaturizer
from model_src.models.conformer import ConformerCtc
from model_src.utils import env_util

logger = env_util.setup_environment()s
DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ----------------------------------------------------------------------
# 3. Argument Parsing
# ----------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(prog="Conformer Online ASR Demo")
    parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="Path to config.yml")
    parser.add_argument("--mxp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--saved", type=str, default=None, help="Path to saved model checkpoint")
    parser.add_argument("--bs", type=int, default=None, help="Batch size for inference")
    parser.add_argument("--input_padding", type=int, default=820)
    parser.add_argument("--label_padding", type=int, default=530)
    return parser.parse_args()

args = parse_arguments()
config = Config(args.config)

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})
env_util.setup_devices([args.device], cpu=args.cpu)

# ----------------------------------------------------------------------
# 4. Build Conformer Model
# ----------------------------------------------------------------------
speech_featurizer = TFSpeechFeaturizer(config.speech_config)
text_featurizer = SentencePieceFeaturizer(config.decoder_config)

def build_conformer(cfg, vocab_size, featurizer_shape):
    model = ConformerCtc(**cfg.model_config, vocabulary_size=vocab_size)
    model.make(featurizer_shape)
    return model

conformer = build_conformer(config, 93, speech_featurizer.shape)
conformer.add_featurizers(speech_featurizer, text_featurizer)

# Optimizer + LR schedule
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    end_learning_rate=1e-5,
    power=1.0
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
conformer.compile(optimizer=optimizer)

# ----------------------------------------------------------------------
# 5. Checkpoint Loading
# ----------------------------------------------------------------------
checkpoint_dir = './checkpoints7'
ckpt = tf.train.Checkpoint(optimizer=conformer.optimizer, model=conformer, epoch=tf.Variable(0))
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f"✅ Restored from last checkpoint (epoch {int(ckpt.epoch.numpy())})")
else:
    print("🔄 Initializing from scratch.")

# ----------------------------------------------------------------------
# 6. Dataset + Decoder
# ----------------------------------------------------------------------
inference = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.train_dataset_config)
)

labels = text_featurizer.tokens
decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path=r"/mnt/c/Users/Phạm Quý Long/Downloads/Project_2_Copy/kenlm/build/model4_gramV3.binary",
    unigrams=unigram_list,
    alpha=1.2,
    beta=1.5,
)

# ----------------------------------------------------------------------
# 7. Online Recognition from File
# ----------------------------------------------------------------------
CHANNELS = 1
RATE = 16000
CHUNK = 256
PROCESS_INTERVAL = 1.75  # seconds

def common_prefix_len(s1, s2):
    """Find the length of common prefix between two strings."""
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

def recognize_from_file(filepath="recorded_audio.wav"):
    last_mtime, last_file_size = 0, 0
    committed_text, provisional_text = "", ""

    while True:
        if not os.path.exists(filepath):
            time.sleep(0.5)
            continue

        mtime, file_size = os.path.getmtime(filepath), os.path.getsize(filepath)

        if mtime != last_mtime:
            # Reset if file is overwritten (new recording)
            if file_size < last_file_size:
                print("\n--- New recording started ---")
                committed_text, provisional_text = "", ""

            last_mtime, last_file_size = mtime, file_size
            rate, audio_buffer = scipy.io.wavfile.read(filepath)

            if len(audio_buffer) < PROCESS_INTERVAL * RATE:
                time.sleep(0.1)
                continue

            audio_data = tf.expand_dims(audio_buffer, axis=-1)
            inputs_tensor, inputs_length_tensor = inference.preprocess_from_mic(audio_data)
            inputs = {
                "inputs": tf.expand_dims(inputs_tensor, axis=0),
                "inputs_length": tf.expand_dims(inputs_length_tensor, axis=0),
            }

            outputs = conformer(inputs, training=False)
            logits = outputs['logits'][0].numpy()
            decoded = decoder.decode(logits)

            if decoded in [committed_text, committed_text + provisional_text]:
                continue

            # Compare with committed text
            prefix_len = common_prefix_len(committed_text, decoded)
            text_to_commit = decoded[len(committed_text):prefix_len]
            new_provisional = decoded[prefix_len:]

            # Clear old provisional text
            if provisional_text:
                print('\b' * len(provisional_text) + ' ' * len(provisional_text) + '\b' * len(provisional_text), end='', flush=True)

            # Print committed part
            if text_to_commit:
                print(text_to_commit, end='', flush=True)
                committed_text += text_to_commit

            # Print new provisional part
            if new_provisional:
                print(new_provisional, end='', flush=True)

            provisional_text = new_provisional
        else:
            time.sleep(0.1)

# ----------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    recognize_from_file()
