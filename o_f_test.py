"""A script for testing TAPS on Onsets & Frames."""
from taps_music import Taps
from taps_music.taps import DEFAULT_EPSILON, DEFAULT_MAX_PITCH_SHIFT

import argparse
from glob import glob
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Dict, List
import warnings

from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
import numpy as np
import pandas as pd
from scipy.stats import hmean
import torch
from torch.serialization import SourceChangeWarning

sys.path.append(str(Path().resolve().parent / "onsets-and-frames"))
sys.path.append(str(Path().resolve().parent / "onsets-and-frames" / "onsets_and_frames"))
from constants import HOP_LENGTH, MIN_MIDI, SAMPLE_RATE  # type: ignore  # noqa: E402
from decoding import extract_notes, notes_to_frames  # type: ignore  # noqa: E402
from midi import parse_midi  # type: ignore  # noqa: E402
from transcribe import load_and_process_audio, transcribe  # type: ignore  # noqa: E402
from transcriber import OnsetsAndFrames  # type: ignore  # noqa: E402


LABEL_FORMATS = ["maps", "smd", "gs", "maestro", "dcs"]


def get_labels(audio_file: str, format: str = LABEL_FORMATS[0]) -> Dict[str, np.ndarray]:
    """
    Load the ground truth labels for a given audio file.

    Parameters
    ----------
    audio_file : str
        Path to the audio file to get the labels for. The labels file should always
        be in the same directory as the audio file, and should have the same name
        as the audio file, but with a different extension (depending on the format).
    format : str, optional
        The format of the labels to load, by default "maps". Options are:
        - "maps": MAPS dataset style. Labels are in .mid files in the same directory
            as the audio files.
        - "smd": SMD dataset style. Labels are in .mid files in a "midi" directory
            parallel to the audio directory.
        - "gs": GuitarSet dataset style. Labels are in the .jams files released
            with the dataset.
        - "maestro": MAESTRO dataset style. Labels are in .midi files in the same
            directory as the audio files.
        - "dcs": Daghstuhl ChoirSet styel. Labels are in .csv files with columns
            onset, offset, and pitch. They should be in a folder called
            annotations_csv_scorerepresentation parallel to the audio directory.

    Returns
    -------
    Dict[str, np.ndarray]
        The ground truth labels for the audio file. The dictionary has 2 keys:
        - "frames": A numpy array of frame labels.
        - "onsets": A numpy array of onset labels.
        Both arrays are of shape (T, 88), where T is the number of frames.
        The arrays may be cut off at the end. Any frames after the final frame
        in each array can be assumed to have all 0s.
    """
    assert format in LABEL_FORMATS, f"Invalid format: {format}"

    if format in ["maps", "smd", "maestro"]:
        if format in "maps":
            midi_file = audio_file.replace(".wav", ".mid")
        elif format == "smd":
            midi_file = str(
                Path(audio_file).parent.parent
                / "midi"
                / Path(audio_file).name.replace(".wav", ".mid")
            )
        elif format == "maestro":
            midi_file = audio_file.replace(".wav", ".midi")

        # This function handles extending note offsets to the end of any sustain pedal
        midi_notes = parse_midi(midi_file)
        label_df = pd.DataFrame(
            {
                "OnsetTime": midi_notes[:, 0],
                "OffsetTime": midi_notes[:, 1],
                "MidiPitch": midi_notes[:, 2],
            }
        )

    elif format == "dcs":
        onset_times = []
        offset_times = []
        midi_pitches = []

        for part in "SATB":
            label_file = (
                Path(audio_file).parent.parent
                / "annotations_csv_scorerepresentation"
                / Path(audio_file).name.replace(".wav", f"_{part}.csv")
            )
            label_df = pd.read_csv(label_file, index_col=None, header=None)

            onset_times.extend(label_df.loc[:, 0])
            offset_times.extend(label_df.loc[:, 1])
            midi_pitches.extend(label_df.loc[:, 2])

        label_df = pd.DataFrame(
            {
                "OnsetTime": onset_times,
                "OffsetTime": offset_times,
                "MidiPitch": midi_pitches,
            }
        )

    elif format == "gs":
        label_file = (
            Path(audio_file).parent.parent
            / "annotation"
            / Path(audio_file).name.replace("_mic.wav", ".jams")
        )
        # Convert guitarset labels to MAPS format
        with open(label_file, "r") as file:
            annotations = json.load(file)["annotations"]

        onset_times = []
        offset_times = []
        midi_pitches = []

        for string in range(6):
            label_index = 2 * string + 1  # Index of the note_midi annotations for each string
            string_annotations = annotations[label_index]["data"]

            onset_times.extend([note["time"] for note in string_annotations])
            offset_times.extend([note["time"] + note["duration"] for note in string_annotations])
            midi_pitches.extend([round(note["value"]) for note in string_annotations])

        label_df = pd.DataFrame(
            {
                "OnsetTime": onset_times,
                "OffsetTime": offset_times,
                "MidiPitch": midi_pitches,
            }
        )

    num_frames = int(round(label_df["OffsetTime"].max() * SAMPLE_RATE / HOP_LENGTH)) + 1

    frames = np.zeros((num_frames, 88), dtype=int)
    onsets = np.zeros((num_frames, 88), dtype=int)

    for _, row in label_df.iterrows():
        onset = int(round(row["OnsetTime"] * SAMPLE_RATE / HOP_LENGTH))
        offset = int(round(row["OffsetTime"] * SAMPLE_RATE / HOP_LENGTH))

        pitch = int(row["MidiPitch"]) - MIN_MIDI
        onsets[onset, pitch] = 1
        frames[onset:offset, pitch] = 1

    return {"frames": frames, "onsets": onsets}


def evaluate(
    predictions: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate the model predictions against the ground truth labels.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        The model's output for a single audio file. The dictionary has 2 keys:
        - "frames": A numpy array of frame probabilities.
        - "onsets": A numpy array of onset probabilities.
        Both arrays are of shape (T, 88), where T is the number of frames.
    labels : Dict[str, np.ndarray]
        The ground truth labels for the audio file. The dictionary has the same
        2 keys as the predictions.

    Returns
    -------
    metrics : Dict[str, float]
        A dictionary of evaluation metrics. Each string key (the name of a metric)
        maps to a float value (the value of the metric). The keys are:
    """
    metrics = {}

    p_ref, i_ref, _ = extract_notes(
        torch.Tensor(labels["onsets"]),
        torch.Tensor(labels["frames"]),
        torch.Tensor(np.ones_like(labels["frames"])),  # We don't care about the velocities
    )
    p_est, i_est, _ = extract_notes(
        torch.Tensor(predictions["onsets"]),
        torch.Tensor(predictions["frames"]),
        torch.Tensor(np.ones_like(predictions["frames"])),  # We don't care about the velocities
    )

    t_ref, f_ref = notes_to_frames(p_ref, i_ref, labels["frames"].shape)
    t_est_post, f_est_post = notes_to_frames(p_est, i_est, predictions["frames"].shape)
    t_est_pre = np.arange(predictions["frames"].shape[0])
    f_est_pre = [np.where(frame >= 0.5)[0] for frame in predictions["frames"]]

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
    t_est_post = t_est_post.astype(np.float64) * scaling
    f_est_post = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est_post]
    t_est_pre = t_est_pre.astype(np.float64) * scaling
    f_est_pre = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est_pre]

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics["note precision"] = p
    metrics["note recall"] = r
    metrics["note f1"] = f
    metrics["note overlap"] = o

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics["note-with-offsets precision"] = p
    metrics["note-with-offsets recall"] = r
    metrics["note-with-offsets f1"] = f
    metrics["note-with-offsets overlap"] = o

    to_add = len(f_est_pre) - len(f_ref)
    if to_add > 0:
        f_ref.extend([np.array([]) for _ in range(to_add)])
        t_ref = t_est_pre
    elif to_add < 0:
        f_ref = f_ref[:to_add]
        t_ref = f_est_pre[:to_add]
    frame_metrics = evaluate_frames(t_ref, f_ref, t_est_pre, f_est_pre)
    frame_metrics["F1"] = hmean([frame_metrics["Precision"], frame_metrics["Recall"]])
    for frame_metric in frame_metrics:
        metrics[f"pre-notes frame {frame_metric}"] = frame_metrics[frame_metric]

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est_post, f_est_post)
    frame_metrics["F1"] = hmean([frame_metrics["Precision"], frame_metrics["Recall"]])
    for frame_metric in frame_metrics:
        metrics[f"post-notes frame {frame_metric}"] = frame_metrics[frame_metric]

    return metrics


class OFTester:
    """
    A class for testing TAPS on Onsets & Frames.
    """

    def __init__(
        self, model_path: str = "../onsets-and-frames/model-500000.pt", device: str = "cpu"
    ):
        """
        Create a new OFTester object.

        Parameters
        ----------
        model_path : str, optional
            The model to load, by default "../onsets_and_frames/model-500000.pt".
        device : str, optional
            The device to run the model on, by default "cpu".
        """
        torch.set_grad_enabled(False)

        # Calculate the model initialization params from the loaded model
        warnings.filterwarnings("ignore", category=SourceChangeWarning)
        model = torch.load(model_path, map_location=device)
        out_features = model.onset_stack[2].out_features
        model_size = model.onset_stack[1].rnn.input_size
        in_features = model.onset_stack[0].fc[0].in_features // (model_size // 8) * 4
        model_complexity = model_size // 16

        # Re-load the model into an instantiated O&F model
        self.model = OnsetsAndFrames(in_features, out_features, model_complexity=model_complexity)
        self.model.load_state_dict(model.state_dict())
        self.model.eval()

        self.device = device

    def process_file(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Run the onsets & frames model on a given audio file.

        Parameters
        ----------
        audio_path : str
            The path to the audio file to process.

        Returns
        -------
        Dict[str, np.ndarray]
            The model's output, as a dictionary. The dictionary has 2 keys:
            - "frames": A numpy array of frame probabilities.
            - "onsets": A numpy array of onset probabilities.
            Both arrays are of shape (T, 88), where T is the number of frames.
        """
        audio = load_and_process_audio(audio_path, None, self.device)
        predictions = transcribe(self.model, audio)

        return {
            "frames": predictions["frame"].numpy(),
            "onsets": predictions["onset"].numpy(),
        }

    def evaluate(self, audio_files: List[str], outputs: List[str], format: str = LABEL_FORMATS[0]):
        """
        Evaluate the TAPS's output on a list of audio files and print the results.

        Parameters
        ----------
        audio_files : list[str]
            The paths to the audio files to test on.
        outputs : List[str]
            The path to a pickle file containing the model's output for each audio file.
            Each pickle file should contain a dictionary with 2 keys:
            - "frames": A numpy array of frame probabilities.
            - "onsets": A numpy array of onset probabilities.
            Both arrays are of shape (T, 88), where T is the number of frames.
        format : str, optional
            The format of the ground truth labels to load, by default "maps".
        """
        metrics = []

        for audio_file, output in zip(audio_files, outputs):
            with open(output, "rb") as file:
                metrics.append(evaluate(pickle.load(file), get_labels(audio_file, format=format)))

        for metric in metrics[0]:
            print(f"{metric}: {np.mean([m[metric] for m in metrics])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test TAPS on Onsets & Frames.")

    parser.add_argument(
        "audio_paths",
        type=str,
        nargs="+",
        help="Paths to directories containing wav files to test TAPS model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="../onsets-and-frames/model-500000.pt",
        help="Path to the Onsets & Frames model to use.",
    )

    parser.add_argument(
        "--max-pitch-shift",
        "-S",
        type=int,
        default=DEFAULT_MAX_PITCH_SHIFT,
        help="The maximum number of semitones to pitch shift the audio by (up and down).",
    )

    parser.add_argument(
        "--pitch-shift-list",
        type=int,
        nargs="+",
        help="A list of specific pitch shifts to use, instead of the range from -S to S.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="The epsilon value to use when finding unsure model outputs.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "The path to a directory to store output files in when running TAPS. "
            "If not given, a temporary directory will be used and no outputs will be saved."
        ),
    )

    parser.add_argument(
        "--format",
        type=str,
        default=LABEL_FORMATS[0],
        choices=LABEL_FORMATS,
        help=(
            "The format of the data and ground truth labels to load ("
            + ", ".join(LABEL_FORMATS) + "). If maps, the labels should be in '.mid' files in "
            "the same directory as the audio files. "
            "If smd, the labels should be in '.mid' files in a 'midi' directory parallel to the "
            "audio directory. "
            "If guitarset, the labels should be the default GuitarSet jams annotation files "
            "in an 'annotation' directory parallel to the audio directory. "
            "If maestro, audio_paths should point to the default as-unzipped Maestro "
            "dataset directory, and the labels should be in the default '.midi' files. "
            "If dcs, the labels should be in '.csv' files in a directory called "
            "annotations_csv_scorerepresentation parallel to the audio directory."
        ),
    )

    parser.add_argument(
        "--no-mean",
        action="store_true",
        help="Instead of using the global mean threshold, use a 0.5 threshold.",
    )

    parser.add_argument(
        "--csvs",
        type=str,
        default=None,
        help=(
            "Instead of running TAPS, generate csvs of model outputs "
            "and save them in the given directory."
        ),
    )

    parser.add_argument(
        "--all-outputs",
        action="store_true",
        help=(
            "If generating dataframes, include all outputs in the DataFrames, rather than "
            "just the unsure ones."
        ),
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics of the model outputs.",
    )

    args = parser.parse_args()

    tester = OFTester(model_path=args.model_path, device=args.device)
    taps = Taps(
        tester.process_file,
        epsilon=args.epsilon,
        max_pitch_shift=args.max_pitch_shift,
        pitch_shifts=args.pitch_shift_list,
    )

    if args.format == "maestro":
        maestro_df = pd.read_csv(Path(args.audio_paths[0]) / "maestro-v2.0.0.csv", index_col=None)
        audio_files = (
            args.audio_paths[0]
            + os.path.sep
            + maestro_df.loc[maestro_df["split"] == "test"]["audio_filename"]
        ).tolist()

    else:
        audio_files = []
        for audio_path in args.audio_paths:
            audio_files.extend(glob(f"{audio_path}/**/*.wav", recursive=True))

    if args.csvs is None and not args.stats:
        temp_dir = None
        if args.save_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            args.save_dir = temp_dir.name
        else:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        baseline, outputs = taps.get_outputs(
            audio_files, args.save_dir, no_mean=args.no_mean
        )
        print("TAPS Results")
        print("======================")
        tester.evaluate(audio_files, outputs, format=args.format)

        print("\n\n\n")
        print("Baseline Results")
        print("================")
        tester.evaluate(audio_files, baseline, format=args.format)

        if temp_dir is not None:
            temp_dir.cleanup()

    if args.csvs is not None:
        Path(args.csvs).mkdir(parents=True, exist_ok=True)
        taps.generate_data_csvs(
            audio_files,
            lambda audio_file: get_labels(audio_file, format=args.format),
            args.csvs,
            all_outputs=args.all_outputs,
        )

    if args.stats:
        taps.print_stats(
            audio_files, lambda audio_file: get_labels(audio_file, format=args.format),
        )
