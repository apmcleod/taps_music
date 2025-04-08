# taps_music

Code to perform zero-shot domain adaptation for AMT systems using the Transcription Adaptation via Pitch Shifting (TAPS) method.

This is the code corresponding to [my ICASSP 2025 paper](https://ieeexplore.ieee.org/abstract/document/10890396). If you use it, please cite the paper:

```
@inproceedings{McLeod:25,
  title={No Data Required: Zero-Shot Domain Adaptation for Automatic Music Transcription},
  author={McLeod, Andrew},
  booktitle={{IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)}},
  year={2025}
}
```

## Installation
To install the taps_music package, clone this repo, and run the following command in the python environment of your choice:

```
pip install -e /path/to/taps_music
```

You should also install the rubberband library for the default pitch shifting (otherwise, librosa will be used):

```
apt install rubberband-cli
```

Those 2 commands should install the `taps_music` class and all needed dependencies.

## Usage
The usage is relatively simple, and involves only a single class:

```
from taps_music import Taps

taps = Taps(run_model_func)
baseline_outputs, taps_outputs = taps.get_outputs(file_list, save_dir)
```

All functions contain pydocs with explanations for these and any other (optional) parameters.
Some notable ones:

- `run_model_func` from the initializer: A function that takes as input the path of an audio file, and returns either
a numpy array of the model's output, or a dictionary mapping string keys (e.g., frames, onset, offset)
to numpy arrays. In either case, the numpy arrays should be of size (num_frames, num_pitches).
The specific dictionary keys don't matter. TAPS just checks to see how closely
the pitch-shifted outputs correspond to the original outputs.

For example:
```
def run_model_func(audio_file) -> Dict[str, np.ndarray]:
    spectrogram = get_spectrogram_from_file(audio_file)
    output = model(spectrogram)
    return {
        "onsets": output[0],
        "frames": output[1],
    }
```

- `epsilon` from the initializer: The float threshold for uncertainty, by default 0.2.

- `max_pitch_shift` from the initializer: The maximum pitch shift (S from the paper), by default 8.

- `save_dir` from `get_outputs`: A path to an already-existing directory.
TAPS will save the outputs of the model to this directory as pickle files, and return
paths to the baseline and model output files, rather than the output dictionaries themselves.

### Generating Data
You can also generate data frames for a set of audio files, containing labels and activations for
different pitch shift amounts:

```
from taps_music.taps import Taps

# Labels is a function that takes an audio file path and returns its labels.
taps.generate_data_dfs(audio_files, get_label_func, save_dir)
```

# Onsets & Frames Testing

This repo also contains code for testing the Onsets & Frames model using this framework in the file
`o_f_test.py`.

## Installation and Setup

First, you need to gather an onsets and frames implementation and pre-trained model.
To do that, I used the 3rd party pytorch implementation [here](https://github.com/greenbech/onsets-and-frames/tree/78cff01bf9df501eb69a74a04ee4310c6b0c98f8).

1. Clone this `music_taps` repo, and install it with `pip install -e /path/to/music_taps`.
2. Clone the linked [onsets-and-frames repo](https://github.com/greenbech/onsets-and-frames/tree/78cff01bf9df501eb69a74a04ee4310c6b0c98f8) into the same directory that you cloned the music_taps repo into.
3. Download the [pre-trained model](https://drive.google.com/file/d/1Mj2Em07Lvl3mvDQCCxOYHjPiB-S0WGT1/view?usp=sharing)
into the onsets-and-frames base directory.
4. Then, in your thresholding environment, you need to install the dependencies from that repo:

These commands will first remove the `--hash` commands from the requirements, replace mir-eval with the default version, and then install the dependencies.
```
cd onsets-and-frames
sed 's/ \\//' requirements.txt | sed 's/;.*//' | sed '/^.*--hash.*$/d' | sed "s;-e git+https://github.com/craffel/mir_eval.git@ffb77ba720d4d0cea030fc6e3d034913510e21d6#egg=mir-eval;mir-eval;" | sed 's/torch==1.2.0/torch==1.13.1/' >reqs-new.txt
pip install -r reqs-new.txt
```

You also need to modify the import statements in 2 files from that repo:

- onsets_and_frames/mel.py
- onsets_and_frames/transcriber.py

There, you need to change the local imports from `from .package import function` to `from package import function`. (Remove the dot.)

## Running

```
python o_f_test.py data_dir
```

Run with `-h` to see other parameters.

# Licensing

This software is also available under a closed source license. Please contact me for more details.
