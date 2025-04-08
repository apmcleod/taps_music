"""Defines the Taps class which calculates new ouptuts based on pitch shifting."""
from taps_music import taps_logger

from collections import defaultdict
import os
from pathlib import Path
import pickle
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import librosa
import numpy as np
import pyrubberband
import soundfile
from tqdm import tqdm


DEFAULT_MAX_PITCH_SHIFT = 8
DEFAULT_EPSILON = 0.2

pyrubberband_error_logged = False


def pitch_shift_with_pyrubberband_or_librosa(
    audio_in: str, pitch_shift: int, audio_out: str
) -> None:
    """
    A default pitch shift function, using pyrubberband to pitch shift the audio.
    If pyrubberband isn't installed, it falls back to using librosa with a warning.

    Parameters
    ----------
    audio_in : str
        The input audio file.
    pitch_shift : int
        The amount to pitch shift the audio by in semitones.
    audio_out : str
        The output audio file.
    """
    audio, sr = soundfile.read(audio_in)
    try:
        audio_shifted = pyrubberband.pitch_shift(audio, sr, pitch_shift)
    except RuntimeError as e:
        global pyrubberband_error_logged
        if not pyrubberband_error_logged:
            warn_msg = (
                "Error pitch shifting with rubberband. Is rubberband-cli installed?\n"
                "Try to install, for example with `sudo apt install rubberband-cli`\n\n"
                f"Original error: {e}\n\n"
                "Falling back to librosa for pitch shifting instead."
            )
            warnings.warn(warn_msg, RuntimeWarning)
            pyrubberband_error_logged = True

        audio_shifted = librosa.effects.pitch_shift(audio, sr, n_steps=pitch_shift)

    soundfile.write(audio_out, audio_shifted, sr, subtype="PCM_16")


class Taps:
    """
    A class to generate better outputs for a given AMT model by transcribing
    pitch-shifted versions of the input and aggregating the outputs.
    """

    def __init__(
        self,
        run_model_func: Callable[[str], Union[np.ndarray, Dict[str, np.ndarray]]],
        pitch_shift_func: Callable[[str, int, str], None] = (
            pitch_shift_with_pyrubberband_or_librosa
        ),
        epsilon: float = DEFAULT_EPSILON,
        max_pitch_shift: int = DEFAULT_MAX_PITCH_SHIFT,
        pitch_shifts: Optional[List[int]] = None,
    ):
        """
        Create a new Taps object.

        Parameters
        ----------
        run_model_func : Callable[[str], Union[np.ndarray, Dict[str, np.ndarray]]]
            A function that takes in a path to an audio file and returns either numpy array of the
            model's predictions, or a dictionary mapping keys (e.g., frames, onsets, offsets) to
            numpy arrays. In either case, the numpy arrays should be float piano rolls of shape
            (num_frames, num_pitches). The threshold finder doesn't care what the different keys
            are, it just looks to check if each output is a pitch-shifted version of the others.
        pitch_shift_func : Callable[[str, int, str], None], optional
            A function that takes in the path to an audio data, a pitch shift amount in semitones,
            and an output path, and pitch shifts the input audio by the given amount of semitones,
            saving the result to the output path.
            By default pitch_shift_with_pyrubberband_or_librosa is used (which uses
            the pyrubberband.pitch_shift function, but falls back to librosa with a warning message
            in case of any error).
        epsilon : float, optional
            The value above which the model's outputs are considered unsure, and TAPS
            is used (1 minus this value is used as the upper bound).
        max_pitch_shift : int, optional
            The maximum pitch shift to use, by default DEFAULT_MAX_PITCH_SHIFT.
        pitch_shifts : Optional[List[int]], optional
            If you wish to use custom pitch shift values, rather than a range from -max_pitch_shift
            to max_pitch_shift inclusively, you can specify them in as a list here. By default,
            this is None, so max_pitch_shift is used.
        """
        self.run_model_func = run_model_func
        self.pitch_shift_func = pitch_shift_func

        self.epsilon = epsilon
        taps_logger.info(f"Using epsilon = {epsilon}")

        self.pitch_shifts = (
            list(range(-max_pitch_shift, max_pitch_shift + 1))
            if pitch_shifts is None
            else pitch_shifts
        )
        if pitch_shifts is None:
            taps_logger.info(f"Using S (max pitch shift) = {max_pitch_shift}")
        else:
            taps_logger.info(f"Using custom pitch shifts: {sorted(self.pitch_shifts)}")

    def _process_audio_file(
        self, audio_file: str, pitch_shift: Optional[int] = 0
    ) -> Dict[str, np.ndarray]:
        """
        Optionally pitch shift a wav file, and then process it with the model,
        returning the model's output. If audio file needs to be pitch shifted,
        a temporary file will be created and deleted before the function returns.

        Parameters
        ----------
        audio_file : str
            The path to the audio file to process.
        pitch_shift : Optional[int], optional
            The number of semitones to pitch shift the audio by, by default 0.

        Returns
        -------
        Dict[str, np.ndarray]
            The model's output as a dictionary mapping keys (e.g., frames, onsets, offsets)
            The numpy arrays will be float piano rolls of shape (num_frames, num_pitches).
            The threshold finder doesn't care what the different keys are, it just looks
            to check if each output is a pitch-shifted version of the others. If the model
            only returns a single numpy array, it will be returned as a dictionary with
            the key "output".
        """
        if pitch_shift != 0:
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix="." + audio_file.split(".")[-1]
            )
            self.pitch_shift_func(audio_file, pitch_shift, tmp_file.name)
            audio_file = tmp_file.name
            tmp_file.close()

        model_output = self.run_model_func(audio_file)

        if pitch_shift != 0:
            os.remove(audio_file)

        return model_output if isinstance(model_output, dict) else {"output": model_output}

    def _process_nonshifted_output(
        self, audio_file: str, output_file_name: str
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Process a non-pitch-shifted version of an audio file, and save the relevant outputs to
        a file.

        Parameters
        ----------
        audio_file : str
            The path to the audio file to process.
        output_file_name : str
            The path to save the relevant output to.

        Returns
        -------
        Dict[str, int]
            A dictionary mapping keys (e.g., frames, onsets, offsets) to the number of relevant
            outputs saved.
        Dict[str, float]
            A dictionary mapping keys to the sum of the relevant outputs saved.
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            A dictionary mapping keys to a tuple of numpy arrays. The first array is the frame
            index, and the second array is the pitch index. The indices in these arrays are the
            positions where the model is unsure (i.e., where the model's output is between epsilon
            and 1 - epsilon).
        """
        relevant_output_count = {}
        relevant_output_sum = {}

        # First calculate the original output.
        output = self._process_audio_file(audio_file, pitch_shift=0)

        # Save the baseline output to a file to save memory
        with open(output_file_name, "wb") as file:
            pickle.dump(output, file)

        # Find the locations where the model is unsure
        unsure_indexes = {
            output_type: np.where(
                (output[output_type] > self.epsilon)
                & (output[output_type] < 1 - self.epsilon)
            )
            for output_type in output.keys()
        }

        # Track the number of unsure outputs total and their sum
        for output_type in unsure_indexes.keys():
            relevant_output_count[output_type] = len(unsure_indexes[output_type][0])
            relevant_output_sum[output_type] = np.sum(
                output[output_type][unsure_indexes[output_type]]
            )

        return relevant_output_count, relevant_output_sum, unsure_indexes

    def _process_shifted_output(
        self,
        audio_file: str,
        pitch_shift: int,
        unsure_indexes: Dict[str, Tuple[np.ndarray, np.ndarray]],
        output_file_name: str,
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Process a pitch-shifted version of an audio file, and save the relevant outputs to a file.

        Parameters
        ----------
        audio_file : str
            The path to the audio file to process.
        pitch_shift : int
            The number of semitones to pitch shift the audio by.
        unsure_indexes : Dict[str, Tuple[np.ndarray, np.ndarray]]
            A dictionary mapping keys (e.g., frames, onsets, offsets) to a tuple of numpy arrays.
            The first array is the frame index, and the second array is the pitch index.
            The indices in these arrays are the positions where the model is unsure (i.e.,
            the outputs we need to save).
        output_file_name : str
            The path to save the relevant output to.

        Returns
        -------
        Tuple[Dict[str, int], Dict[str, float]]
            A tuple of two dictionaries. The first dictionary maps keys (e.g., frames, onsets,
            offsets) to the number of relevant outputs saved. The second dictionary maps keys
            to the sum of the relevant outputs saved.
        """
        relevant_output_count = {}
        relevant_output_sum = {}

        output = self._process_audio_file(audio_file, pitch_shift=pitch_shift)
        output_to_save = {}  # We'll save only what's relevant to a file

        for output_type in output.keys():
            # Skip out-of-range pitches
            unsure_pitches = unsure_indexes[output_type][1] + pitch_shift
            valid_unsure_index_mask = (
                (0 <= unsure_pitches) & (unsure_pitches < len(output[output_type][1]))
            )
            valid_unsure_indexes = (
                np.array(unsure_indexes[output_type][0][valid_unsure_index_mask]),
                np.array(
                    unsure_indexes[output_type][1][valid_unsure_index_mask]
                    + pitch_shift
                ),
            )

            # Track and save the outputs at the relevant indexes
            relevant_outputs = output[output_type][valid_unsure_indexes]
            relevant_output_count[output_type] = len(valid_unsure_indexes[0])
            relevant_output_sum[output_type] = np.sum(relevant_outputs)

            output[output_type] = None
            output_to_save[f"{output_type}_mask"] = valid_unsure_index_mask
            output_to_save[output_type] = relevant_outputs

        with open(output_file_name, "wb") as file:
            pickle.dump(output_to_save, file)

        return relevant_output_count, relevant_output_sum

    def get_outputs(
        self, audio_files: List[str], save_dir: str, no_mean: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Run the TAPS on a set of audio files and return the outputs as pickle files.

        Parameters
        ----------
        audio_files : List[str]
            A List of paths to audio files whose output to return.
        save_dir : str
            The directory to save the output pickle files to.
        no_mean : bool, optional
            If True, TAPS will not use the global mean as a threshold for each
            output, but rather 0.5.

        Returns
        -------
        List[str]
            The paths to baseline output files, as dictionaries saved in pickle format.
            One file path per input audio file.
            Each dictionary maps keys (e.g., frames, onsets, offsets) to numpy arrays.
            The numpy arrays will be float piano rolls of shape (num_frames, num_pitches).
        List[str]
            The paths to TAPS output files, as dictionaries saved in pickle
            format. One file path per input audio file.
            Each dictionary maps keys (e.g., frames, onsets, offsets) to numpy arrays.
            The numpy arrays will be float piano rolls of shape (num_frames, num_pitches).
            For uncertain outputs (those between epsilon and 1 - epsilon), the value will
            be replaced by TAPS's output.
        """
        relevant_output_count = defaultdict(int)
        relevant_output_sum = defaultdict(float)

        for audio_file in tqdm(
            audio_files, desc="Transcribing Audio Files", position=0, leave=False
        ):
            # Calculate the original output and save it
            file_name = os.path.join(
                save_dir, f"{os.path.basename(audio_file)}.baseline.pkl"
            )
            relevant_output_count_nonshifted, relevant_output_sum_nonshifted, unsure_indexes = (
                self._process_nonshifted_output(audio_file, file_name)
            )

            for output_type in relevant_output_count_nonshifted.keys():
                relevant_output_count[output_type] += relevant_output_count_nonshifted[output_type]
                relevant_output_sum[output_type] += relevant_output_sum_nonshifted[output_type]

            # Calculate all pitch-shifted outputs, and save them
            for pitch_shift in tqdm(
                set(self.pitch_shifts) - set([0]), desc="Pitch Shift", position=1, leave=False
            ):
                file_name = os.path.join(
                    save_dir, f"{os.path.basename(audio_file)}.{pitch_shift}.pkl"
                )
                relevant_output_count_shifted, relevant_output_sum_shifted = (
                    self._process_shifted_output(
                        audio_file, pitch_shift, unsure_indexes, file_name
                    )
                )

                for output_type in relevant_output_count.keys():
                    relevant_output_count[output_type] += (
                        relevant_output_count_shifted[output_type]
                    )
                    relevant_output_sum[output_type] += relevant_output_sum_shifted[output_type]
            del unsure_indexes

        global_average_output = {
            output_type: relevant_output_sum[output_type] / relevant_output_count[output_type]
            for output_type in relevant_output_sum.keys()
        }
        for output_type in global_average_output.keys():
            taps_logger.info(
                f"Global average {output_type} output: {global_average_output[output_type]}"
            )
        if no_mean:
            taps_logger.info("no_mean parameter was True: using 0.5 as the threshold.")
            global_average_output = {
                output_type: 0.5 for output_type in global_average_output.keys()
            }

        baseline_file_names = []
        taps_file_names = []

        for audio_file in tqdm(audio_files, desc="Calculating new outputs"):
            # Load the original output
            file_name = os.path.join(
                save_dir, f"{os.path.basename(audio_file)}.baseline.pkl"
            )
            with open(file_name, "rb") as file:
                output = pickle.load(file)
            baseline_file_names.append(file_name)

            # Find the locations where the model is unsure
            unsure_indexes = {
                output_type: np.where(
                    (output[output_type] > self.epsilon)
                    & (output[output_type] < 1 - self.epsilon)
                )
                for output_type in output.keys()
            }

            # Track the number of outputs per unsure index
            relevant_output_counts = {
                output_type: np.ones(len(unsure_indexes[output_type][0]), dtype=int)
                for output_type in unsure_indexes.keys()
            }

            # Track the sum of outputs for each unsure indexes
            relevant_output_sums = {
                output_type: output[output_type][unsure_indexes[output_type]]
                for output_type in output.keys()
            }

            # Update counts and sums based on the pitch-shifted outputs
            for pitch_shift in set(self.pitch_shifts) - set([0]):
                file_name = os.path.join(
                    save_dir, f"{os.path.basename(audio_file)}.{pitch_shift}.pkl"
                )
                with open(file_name, "rb") as file:
                    shifted_output = pickle.load(file)
                Path(file_name).unlink()  # No need to save pitch-shifted outputs any more

                for output_type in relevant_output_counts.keys():
                    # Track and save the outputs at the relevant indexes
                    relevant_output_counts[output_type] += shifted_output[
                        f"{output_type}_mask"
                    ]
                    relevant_output_sums[output_type][shifted_output[f"{output_type}_mask"]] += (
                        shifted_output[output_type]
                    )

                    shifted_output[output_type] = None
                    shifted_output[f"{output_type}_mask"] = None
                del shifted_output

            # Calculate the average output at each unsure index
            unsure_index_averages = {
                output_type: (
                    relevant_output_sums[output_type] / relevant_output_counts[output_type]
                )
                for output_type in relevant_output_sums.keys()
            }

            # Replace unsure outputs with the TAPS's output
            for output_type in output.keys():
                positive = unsure_index_averages[output_type] >= global_average_output[output_type]
                negative = ~positive

                new_output = np.zeros(len(unsure_indexes[output_type][0]), dtype=float)
                new_output[negative] = (
                    0.5
                    * unsure_index_averages[output_type][negative]
                    / global_average_output[output_type]
                )
                new_output[positive] = (
                    0.5
                    + (
                        unsure_index_averages[output_type][positive]
                        - global_average_output[output_type]
                    )
                    / (2 * (1 - global_average_output[output_type]))
                )

                output[output_type][unsure_indexes[output_type]] = new_output

            # Write the TAPS output to file
            file_name = os.path.join(
                save_dir, f"{os.path.basename(audio_file)}.taps.pkl"
            )
            with open(file_name, "wb") as file:
                pickle.dump(output, file)
            taps_file_names.append(file_name)
            del output

        return baseline_file_names, taps_file_names

    def generate_data_csvs(
        self,
        audio_files: List[str],
        get_labels_func: Callable[[str], Dict[str, np.ndarray]],
        save_path: str,
        all_outputs: bool = False,
    ):
        """
        Generate and save csv of relevant model outputs for a given set of audio file.
        Each csv file (one per input audio file) will contain output information for
        positions (frame, pitch) where the non-pitch-shifted output is ambiguous
        (i.e., not close to 0 or 1). This information includes the pitch-shifted outputs
        as well as the ground truth label.

        Parameters
        ----------
        audio_files : List[str]
            A list of paths to audio files to use to generate the data.
        get_labels_func: Callable[[str], Dict[str, np.ndarray]]
            A function that takes as input the path to an audio file and returns a dictionary
            mapping keys (e.g., frames, onsets, offsets) to binary numpy arrays with labels.
        all_outputs : bool, optional
            If True, the csv will contain all outputs, not just the ambiguous ones.
        """
        for audio_file in tqdm(
            audio_files, desc="Processing Audio Files", position=0, leave=False
        ):
            data = defaultdict(list)
            output = self._process_audio_file(audio_file, pitch_shift=0)

            if all_outputs:
                relevant_output_indexes = {
                    output_type: np.where(
                        (output[output_type] >= 0) & (output[output_type] <= 1)
                    )
                    for output_type in output.keys()
                }
            else:
                # Get the outputs that are potentially ambiguous
                relevant_output_indexes = {
                    output_type: np.where(
                        (output[output_type] > self.epsilon)
                        & (output[output_type] < 1 - self.epsilon)
                    )
                    for output_type in output.keys()
                }

            # Add the no-shift output and the basic indexes
            for output_type in output.keys():
                data["file_name"] += [audio_file] * len(relevant_output_indexes[output_type][0])
                data["output_type"] += [output_type] * len(relevant_output_indexes[output_type][0])
                data["frame"] += relevant_output_indexes[output_type][0].tolist()
                data["pitch"] += relevant_output_indexes[output_type][1].tolist()
                data[0] += output[output_type][relevant_output_indexes[output_type]].tolist()
                output[output_type] = None  # Free up memory
            del output

            # Get the ground truth labels
            labels = get_labels_func(audio_file)

            # Pad label lengths with 0s to match the model output lengths
            for output_type in labels.keys():
                labels[output_type] = np.pad(
                    labels[output_type],
                    (
                        (
                            0,
                            max(
                                0,
                                (
                                    np.max(relevant_output_indexes[output_type][0])
                                    - len(labels[output_type]) + 1
                                ),
                            ),
                        ),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )
                data["label"] += labels[output_type][relevant_output_indexes[output_type]].tolist()
                labels[output_type] = None  # Free up memory
            del labels

            # Add the pitch-shifted outputs
            for pitch_shift in tqdm(
                set(self.pitch_shifts) - set([0]), desc="Pitch shifts", position=1, leave=False
            ):
                output = self._process_audio_file(audio_file, pitch_shift=pitch_shift)

                for output_type in output.keys():
                    for frame, pitch in zip(*relevant_output_indexes[output_type]):
                        pitch += pitch_shift

                        if 0 <= pitch < len(output[output_type][0]):
                            data[pitch_shift] += [output[output_type][frame, pitch]]
                        else:
                            # Out of bounds pitch
                            data[pitch_shift] += [""]
                    output[output_type] = None  # Free up memory
                del output

            # Write to csv (avoiding pandas dependency)
            with open(os.path.join(save_path, f"{os.path.basename(audio_file)}.csv"), "w") as file:
                keys = data.keys()
                file.write(",".join([str(key) for key in keys]) + "\n")
                for i in range(len(data["file_name"])):
                    file.write(",".join([str(data[key][i]) for key in keys]) + "\n")

    def print_stats(
        self, audio_files: List[str], get_labels_func: Callable[[str], Dict[str, np.ndarray]],
    ):
        """
        Print various statistics about the model's outputs. Those statistics are (per output type):
        - The total number of outputs.
        - The number of unsure outputs (on the non-pitch-shifted data).
        - The number of unsure outputs that are correct.
        - The number of non-unsure outputs (on the non-pitch-shifted data).
        - The number of non-unsure outputs that are correct.
        - The total number of corresponding pitch-shifted outputs.
        - The number of corresponding pitch-shifted outputs that are not themselves unsure.
        - The number of corresponding pitch-shifted outputs, that are not themselves unsure,
            that are correct.

        Parameters
        ----------
        audio_files : List[str]
            The audio files to use to generate the data.
        get_labels_func : Callable[[str], Dict[str, np.ndarray]]
            A function that takes as input the path to an audio file and returns a dictionary
            mapping keys (e.g., frames, onsets, offsets) to binary numpy arrays with labels.
        """
        total_outputs = defaultdict(int)
        unsure_outputs = defaultdict(int)
        unsure_correct = defaultdict(int)
        non_unsure_outputs = defaultdict(int)
        non_unsure_correct = defaultdict(int)
        total_shifted_outputs = defaultdict(int)
        non_unsure_shifted_outputs = defaultdict(int)
        non_unsure_shifted_correct = defaultdict(int)

        for audio_file in tqdm(
            audio_files, desc="Processing Audio Files", position=0, leave=False
        ):
            output = self._process_audio_file(audio_file, pitch_shift=0)
            labels = get_labels_func(audio_file)

            # Extend labels to equal output size
            for output_type in output.keys():
                labels[output_type] = np.pad(
                    labels[output_type],
                    (
                        (0, max(0, len(output[output_type]) - len(labels[output_type]))),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )

            # Get the outputs that are potentially ambiguous
            unsure_output_indexes = {
                output_type: np.where(
                    (output[output_type] > self.epsilon)
                    & (output[output_type] < 1 - self.epsilon)
                )
                for output_type in output.keys()
            }
            non_unsure_output_indexes = {
                output_type: np.where(
                    (output[output_type] <= self.epsilon)
                    | (output[output_type] >= 1 - self.epsilon)
                )
                for output_type in output.keys()
            }

            for output_type in output.keys():
                total_outputs[output_type] += np.prod(output[output_type].shape)
                unsure_outputs[output_type] += len(unsure_output_indexes[output_type][0])
                unsure_correct[output_type] += np.sum(
                    labels[output_type][unsure_output_indexes[output_type]]
                    == (output[output_type][unsure_output_indexes[output_type]] >= 0.5)
                )
                non_unsure_outputs[output_type] += len(non_unsure_output_indexes[output_type][0])
                non_unsure_correct[output_type] += np.sum(
                    labels[output_type][non_unsure_output_indexes[output_type]]
                    == (output[output_type][non_unsure_output_indexes[output_type]] >= 0.5)
                )

            for pitch_shift in tqdm(
                set(self.pitch_shifts) - set([0]), desc="Pitch shifts", position=1, leave=False
            ):
                shifted_output = self._process_audio_file(audio_file, pitch_shift=pitch_shift)

                for output_type in output.keys():
                    for unsure_frame, unsure_pitch in zip(
                        *unsure_output_indexes[output_type]
                    ):
                        unsure_pitch += pitch_shift

                        if 0 <= unsure_pitch < len(shifted_output[output_type][0]):
                            total_shifted_outputs[output_type] += 1
                            if (
                                (
                                    shifted_output[output_type][unsure_frame, unsure_pitch]
                                    <= self.epsilon
                                )
                                or (
                                    shifted_output[output_type][unsure_frame, unsure_pitch]
                                    >= 1 - self.epsilon
                                )
                            ):
                                non_unsure_shifted_outputs[output_type] += 1
                                non_unsure_shifted_correct[output_type] += (
                                    labels[output_type][unsure_frame, unsure_pitch]
                                    == (
                                        shifted_output[output_type][unsure_frame, unsure_pitch]
                                        >= 0.5
                                    )
                                )
                    shifted_output[output_type] = None
                del shifted_output

        for output_type in total_outputs.keys():
            print(f"Output Type: {output_type}")
            print(f"Total Outputs: {total_outputs[output_type]}")
            print(
                f"Unsure Outputs: {unsure_outputs[output_type]} = "
                f"{unsure_outputs[output_type] / total_outputs[output_type] * 100:.4f}%"
            )
            print(
                f"Unsure Correct: {unsure_correct[output_type]} = "
                f"{unsure_correct[output_type] / unsure_outputs[output_type] * 100:.4f}%"
            )
            print(
                f"Non-Unsure Outputs: {non_unsure_outputs[output_type]} = "
                f"{non_unsure_outputs[output_type] / total_outputs[output_type] * 100:.4f}%"
            )
            print(
                f"Non-Unsure Correct: {non_unsure_correct[output_type]} = "
                f"{non_unsure_correct[output_type] / non_unsure_outputs[output_type] * 100:.4f}%"
            )
            print(f"Total Shifted Outputs: {total_shifted_outputs[output_type]}")
            non_unsure_shifted_outputs_percent = (
                non_unsure_shifted_outputs[output_type] / total_shifted_outputs[output_type] * 100
            )
            print(
                f"Non-Unsure Shifted Outputs: {non_unsure_shifted_outputs[output_type]} = "
                f"{non_unsure_shifted_outputs_percent:.4f}%"
            )
            non_unsure_shifted_correct_percent = (
                non_unsure_shifted_correct[output_type]
                / non_unsure_shifted_outputs[output_type]
                * 100
            )
            print(
                f"Non-Unsure Shifted Correct: {non_unsure_shifted_correct[output_type]} = "
                f"{non_unsure_shifted_correct_percent:.4f}%"
            )
            print()
