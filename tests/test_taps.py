from taps_music import Taps
from taps_music.taps import DEFAULT_EPSILON, DEFAULT_MAX_PITCH_SHIFT

import os
import pickle

import numpy as np


# Define some standard output/labels
def get_output_fn(x):
    return {
        "frame": np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0, 0, 0]]),
        "onset": np.array([[0, 0, 0], [0, 0, 0.1], [0, 0.3, 0], [0, 0, 0]]),
    }


def get_labels_fn(x):
    return {
        "frame": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        "onset": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
    }


# All outputs and labels are flipped -- this should never change the results
def get_output_fn_flipped(x):
    return {output_type: 1 - output for output_type, output in get_output_fn(x).items()}


def get_labels_fn_flipped(x):
    return {output_type: 1 - output for output_type, output in get_labels_fn(x).items()}


# Tests being here
def test_taps_init():
    # Test default values
    taps = Taps(lambda: 0)
    assert taps.epsilon == DEFAULT_EPSILON
    assert taps.pitch_shifts == list(range(-DEFAULT_MAX_PITCH_SHIFT, DEFAULT_MAX_PITCH_SHIFT + 1))

    # Test epsilon
    taps = Taps(lambda: 0, epsilon=0.1)
    assert taps.epsilon == 0.1
    assert taps.pitch_shifts == list(range(-DEFAULT_MAX_PITCH_SHIFT, DEFAULT_MAX_PITCH_SHIFT + 1))

    # Test custom pitch_shift lists
    for pitch_shifts in [[-1, 0, 1], [-2, 0, 2], [-3, 3], [20]]:
        taps = Taps(lambda: 0, pitch_shifts=pitch_shifts, max_pitch_shift=1)
        assert taps.epsilon == DEFAULT_EPSILON
        assert taps.pitch_shifts == pitch_shifts

    # Test custom max_pitch_shift
    for max_pitch_shift in range(30):
        taps = Taps(lambda: 0, max_pitch_shift=max_pitch_shift)
        assert taps.epsilon == DEFAULT_EPSILON
        assert taps.pitch_shifts == list(range(-max_pitch_shift, max_pitch_shift + 1))


def test_get_outputs(caplog):
    # Basic test
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None)
    taps.get_outputs(["test.wav", "test2.wav"], ".")

    with open("test.wav.baseline.pkl", "rb") as file:
        test_baseline = pickle.load(file)
    os.unlink("test.wav.baseline.pkl")
    with open("test2.wav.baseline.pkl", "rb") as file:
        test_baseline_2 = pickle.load(file)
    os.unlink("test2.wav.baseline.pkl")
    with open("test.wav.taps.pkl", "rb") as file:
        test_taps = pickle.load(file)
    os.unlink("test.wav.taps.pkl")
    with open("test2.wav.taps.pkl", "rb") as file:
        test_taps_2 = pickle.load(file)
    os.unlink("test2.wav.taps.pkl")

    assert all(
        np.all(test_baseline[output_type] == test_baseline_2[output_type])
        for output_type in test_baseline
    )
    assert all(
        np.all(test_taps[output_type] == test_taps_2[output_type])
        for output_type in test_taps
    )
    assert all(
        np.all(test_baseline[output_type] == get_output_fn(None)[output_type])
        for output_type in test_baseline
    )
    expected_taps = {
        "frame": np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0, 0, 0]]),
        "onset": np.array([[0, 0, 0], [0, 0, 0.1], [0, 0.5, 0], [0, 0, 0]]),
    }
    assert all(
        np.all(test_taps[output_type] == expected_taps[output_type])
        for output_type in test_taps
    )

    # Basic test with lower epsilon (no change)
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, epsilon=0.1)
    taps.get_outputs(["test.wav", "test2.wav"], ".")

    with open("test.wav.baseline.pkl", "rb") as file:
        test_baseline = pickle.load(file)
    os.unlink("test.wav.baseline.pkl")
    with open("test2.wav.baseline.pkl", "rb") as file:
        test_baseline_2 = pickle.load(file)
    os.unlink("test2.wav.baseline.pkl")
    with open("test.wav.taps.pkl", "rb") as file:
        test_taps = pickle.load(file)
    os.unlink("test.wav.taps.pkl")
    with open("test2.wav.taps.pkl", "rb") as file:
        test_taps_2 = pickle.load(file)
    os.unlink("test2.wav.taps.pkl")

    assert all(
        np.all(test_baseline[output_type] == test_baseline_2[output_type])
        for output_type in test_baseline
    )
    assert all(
        np.all(test_taps[output_type] == test_taps_2[output_type])
        for output_type in test_taps
    )
    assert all(
        np.all(test_baseline[output_type] == get_output_fn(None)[output_type])
        for output_type in test_baseline
    )
    assert all(
        np.all(test_taps[output_type] == expected_taps[output_type])
        for output_type in test_taps
    )

    # Basic test with even lower epsilon
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, epsilon=0.09)
    taps.get_outputs(["test.wav", "test2.wav"], ".")

    with open("test.wav.baseline.pkl", "rb") as file:
        test_baseline = pickle.load(file)
    os.unlink("test.wav.baseline.pkl")
    with open("test2.wav.baseline.pkl", "rb") as file:
        test_baseline_2 = pickle.load(file)
    os.unlink("test2.wav.baseline.pkl")
    with open("test.wav.taps.pkl", "rb") as file:
        test_taps = pickle.load(file)
    os.unlink("test.wav.taps.pkl")
    with open("test2.wav.taps.pkl", "rb") as file:
        test_taps_2 = pickle.load(file)
    os.unlink("test2.wav.taps.pkl")

    assert all(
        np.all(test_baseline[output_type] == test_baseline_2[output_type])
        for output_type in test_baseline
    )
    assert all(
        np.all(test_taps[output_type] == test_taps_2[output_type])
        for output_type in test_taps
    )
    assert all(
        np.all(test_baseline[output_type] == get_output_fn(None)[output_type])
        for output_type in test_baseline
    )
    expected_taps = {
        "frame": np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0, 0, 0]]),
        "onset": np.array([[0, 0, 0], [0, 0, 0.25], [0, 0.51785714, 0], [0, 0, 0]]),
    }
    assert all(
        np.allclose(test_taps[output_type], expected_taps[output_type])
        for output_type in test_taps
    )

    # Test with using 0.5 instead of th mean
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None)
    taps.get_outputs(["test.wav", "test2.wav"], ".", no_mean=True)

    with open("test.wav.baseline.pkl", "rb") as file:
        test_baseline = pickle.load(file)
    os.unlink("test.wav.baseline.pkl")
    with open("test2.wav.baseline.pkl", "rb") as file:
        test_baseline_2 = pickle.load(file)
    os.unlink("test2.wav.baseline.pkl")
    with open("test.wav.taps.pkl", "rb") as file:
        test_taps = pickle.load(file)
    os.unlink("test.wav.taps.pkl")
    with open("test2.wav.taps.pkl", "rb") as file:
        test_taps_2 = pickle.load(file)
    os.unlink("test2.wav.taps.pkl")

    assert all(
        np.all(test_baseline[output_type] == test_baseline_2[output_type])
        for output_type in test_baseline
    )
    assert all(
        np.all(test_taps[output_type] == test_taps_2[output_type])
        for output_type in test_taps
    )
    assert all(
        np.all(test_baseline[output_type] == get_output_fn(None)[output_type])
        for output_type in test_baseline
    )
    expected_taps = {
        "frame": np.array([[0, 0, 0], [0, 0, 0.166667], [0, 0.166667, 0], [0, 0, 0]]),
        "onset": np.array([[0, 0, 0], [0, 0, 0.1], [0, 0.1, 0], [0, 0, 0]]),
    }
    assert all(
        np.allclose(test_taps[output_type], expected_taps[output_type])
        for output_type in test_taps
    )

    # Test with custom pitch shifts
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, pitch_shifts=[-1, 2])
    taps.get_outputs(["test.wav", "test2.wav"], ".", no_mean=True)

    with open("test.wav.baseline.pkl", "rb") as file:
        test_baseline = pickle.load(file)
    os.unlink("test.wav.baseline.pkl")
    with open("test2.wav.baseline.pkl", "rb") as file:
        test_baseline_2 = pickle.load(file)
    os.unlink("test2.wav.baseline.pkl")
    with open("test.wav.taps.pkl", "rb") as file:
        test_taps = pickle.load(file)
    os.unlink("test.wav.taps.pkl")
    with open("test2.wav.taps.pkl", "rb") as file:
        test_taps_2 = pickle.load(file)
    os.unlink("test2.wav.taps.pkl")

    assert all(
        np.all(test_baseline[output_type] == test_baseline_2[output_type])
        for output_type in test_baseline
    )
    assert all(
        np.all(test_taps[output_type] == test_taps_2[output_type])
        for output_type in test_taps
    )
    assert all(
        np.all(test_baseline[output_type] == get_output_fn(None)[output_type])
        for output_type in test_baseline
    )
    print(test_taps)
    expected_taps = {
        "frame": np.array([[0, 0, 0], [0, 0, 0.25], [0, 0.25, 0], [0, 0, 0]]),
        "onset": np.array([[0, 0, 0], [0, 0, 0.1], [0, 0.15, 0], [0, 0, 0]]),
    }
    assert all(
        np.allclose(test_taps[output_type], expected_taps[output_type])
        for output_type in test_taps
    )

    expected_logs = [
        "Using epsilon = 0.2",
        "Using S (max pitch shift) = 8",
        "Global average frame output: 0.16666666666666666",
        "Global average onset output: 0.09999999999999999",
        "Using epsilon = 0.1",
        "Using S (max pitch shift) = 8",
        "Global average frame output: 0.16666666666666666",
        "Global average onset output: 0.09999999999999999",
        "Using epsilon = 0.09",
        "Using S (max pitch shift) = 8",
        "Global average frame output: 0.16666666666666666",
        "Global average onset output: 0.06666666666666667",
        "Using epsilon = 0.2",
        "Using S (max pitch shift) = 8",
        "Global average frame output: 0.16666666666666666",
        "Global average onset output: 0.09999999999999999",
        "no_mean parameter was True: using 0.5 as the threshold.",
        "Using epsilon = 0.2",
        "Using custom pitch shifts: [-1, 2]",
        "Global average frame output: 0.25",
        "Global average onset output: 0.15",
        "no_mean parameter was True: using 0.5 as the threshold.",
    ]
    assert expected_logs == caplog.messages


def test_generate_data_csvs():
    # Basic test
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None)
    taps.generate_data_csvs(["test.wav", "test2.wav"], get_labels_fn, ".")

    expected_out_1 = (
        """file_name,output_type,frame,pitch,0,label,1,2,3,4,5,6,7,8,-2,-8,-7,-6,-5,-4,-3,-1
test.wav,frame,1,2,0.5,1,,,,,,,,,0.0,,,,,,,0.0
test.wav,frame,2,1,0.5,0,0.0,,,,,,,,,,,,,,,0.0
test.wav,onset,2,1,0.3,0,0.0,,,,,,,,,,,,,,,0.0
"""
    )
    with open("test.wav.csv", "r") as f:
        assert f.read() == expected_out_1
    os.unlink("test.wav.csv")
    with open("test2.wav.csv", "r") as f:
        assert f.read() == expected_out_1.replace("test.wav", "test2.wav")
    os.unlink("test2.wav.csv")

    taps.generate_data_csvs(["test.wav", "test2.wav"], get_labels_fn, ".", all_outputs=True)
    expected_out_full_1 = (
        """file_name,output_type,frame,pitch,0,label,1,2,3,4,5,6,7,8,-2,-8,-7,-6,-5,-4,-3,-1
test.wav,frame,0,0,0.0,0,0.0,0.0,,,,,,,,,,,,,,
test.wav,frame,0,1,0.0,0,0.0,,,,,,,,,,,,,,,0.0
test.wav,frame,0,2,0.0,0,,,,,,,,,0.0,,,,,,,0.0
test.wav,frame,1,0,0.0,0,0.0,0.5,,,,,,,,,,,,,,
test.wav,frame,1,1,0.0,0,0.5,,,,,,,,,,,,,,,0.0
test.wav,frame,1,2,0.5,1,,,,,,,,,0.0,,,,,,,0.0
test.wav,frame,2,0,0.0,0,0.5,0.0,,,,,,,,,,,,,,
test.wav,frame,2,1,0.5,0,0.0,,,,,,,,,,,,,,,0.0
test.wav,frame,2,2,0.0,0,,,,,,,,,0.0,,,,,,,0.5
test.wav,frame,3,0,0.0,0,0.0,0.0,,,,,,,,,,,,,,
test.wav,frame,3,1,0.0,0,0.0,,,,,,,,,,,,,,,0.0
test.wav,frame,3,2,0.0,0,,,,,,,,,0.0,,,,,,,0.0
test.wav,onset,0,0,0.0,0,0.0,0.0,,,,,,,,,,,,,,
test.wav,onset,0,1,0.0,0,0.0,,,,,,,,,,,,,,,0.0
test.wav,onset,0,2,0.0,0,,,,,,,,,0.0,,,,,,,0.0
test.wav,onset,1,0,0.0,0,0.0,0.1,,,,,,,,,,,,,,
test.wav,onset,1,1,0.0,0,0.1,,,,,,,,,,,,,,,0.0
test.wav,onset,1,2,0.1,1,,,,,,,,,0.0,,,,,,,0.0
test.wav,onset,2,0,0.0,0,0.3,0.0,,,,,,,,,,,,,,
test.wav,onset,2,1,0.3,0,0.0,,,,,,,,,,,,,,,0.0
test.wav,onset,2,2,0.0,0,,,,,,,,,0.0,,,,,,,0.3
test.wav,onset,3,0,0.0,0,0.0,0.0,,,,,,,,,,,,,,
test.wav,onset,3,1,0.0,0,0.0,,,,,,,,,,,,,,,0.0
test.wav,onset,3,2,0.0,0,,,,,,,,,0.0,,,,,,,0.0
"""
    )

    with open("test.wav.csv", "r") as f:
        assert f.read() == expected_out_full_1
    os.unlink("test.wav.csv")
    with open("test2.wav.csv", "r") as f:
        assert f.read() == expected_out_full_1.replace("test.wav", "test2.wav")
    os.unlink("test2.wav.csv")

    # Test with lower epsilon, but nothing should change
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, epsilon=0.1)
    taps.generate_data_csvs(["test.wav", "test2.wav"], get_labels_fn, ".", all_outputs=True)
    with open("test.wav.csv", "r") as f:
        assert f.read() == expected_out_full_1
    os.unlink("test.wav.csv")
    with open("test2.wav.csv", "r") as f:
        assert f.read() == expected_out_full_1.replace("test.wav", "test2.wav")
    os.unlink("test2.wav.csv")

    # Test with lower epsilon, something should change
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, epsilon=0.09)
    taps.generate_data_csvs(["test.wav", "test2.wav"], get_labels_fn, ".")

    expected_out_1 = (
        """file_name,output_type,frame,pitch,0,label,1,2,3,4,5,6,7,8,-2,-8,-7,-6,-5,-4,-3,-1
test.wav,frame,1,2,0.5,1,,,,,,,,,0.0,,,,,,,0.0
test.wav,frame,2,1,0.5,0,0.0,,,,,,,,,,,,,,,0.0
test.wav,onset,1,2,0.1,1,,,,,,,,,0.0,,,,,,,0.0
test.wav,onset,2,1,0.3,0,0.0,,,,,,,,,,,,,,,0.0
"""
    )

    with open("test.wav.csv", "r") as f:
        assert f.read() == expected_out_1
    os.unlink("test.wav.csv")
    with open("test2.wav.csv", "r") as f:
        assert f.read() == expected_out_1.replace("test.wav", "test2.wav")
    os.unlink("test2.wav.csv")

    # Test with custom pitch shifts
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, pitch_shifts=[-2, 1])
    taps.generate_data_csvs(["test.wav", "test2.wav"], get_labels_fn, ".")

    expected_out_1 = (
        """file_name,output_type,frame,pitch,0,label,1,-2
test.wav,frame,1,2,0.5,1,,0.0
test.wav,frame,2,1,0.5,0,0.0,
test.wav,onset,2,1,0.3,0,0.0,
"""
    )

    with open("test.wav.csv", "r") as f:
        assert f.read() == expected_out_1
    os.unlink("test.wav.csv")
    with open("test2.wav.csv", "r") as f:
        assert f.read() == expected_out_1.replace("test.wav", "test2.wav")
    os.unlink("test2.wav.csv")


def test_print_stats(capfd):
    # Basic test
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None)
    taps.print_stats([""], get_labels_fn)

    expected_out = (
        """Output Type: frame
Total Outputs: 12
Unsure Outputs: 2 = 16.6667%
Unsure Correct: 1 = 50.0000%
Non-Unsure Outputs: 10 = 83.3333%
Non-Unsure Correct: 10 = 100.0000%
Total Shifted Outputs: 4
Non-Unsure Shifted Outputs: 4 = 100.0000%
Non-Unsure Shifted Correct: 4 = 100.0000%

Output Type: onset
Total Outputs: 12
Unsure Outputs: 1 = 8.3333%
Unsure Correct: 1 = 100.0000%
Non-Unsure Outputs: 11 = 91.6667%
Non-Unsure Correct: 10 = 90.9091%
Total Shifted Outputs: 2
Non-Unsure Shifted Outputs: 2 = 100.0000%
Non-Unsure Shifted Correct: 2 = 100.0000%

"""
    )

    out, _ = capfd.readouterr()
    assert expected_out == out

    # Test with lower epsilon, but nothing should change
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, epsilon=0.1)
    taps.print_stats([""], get_labels_fn)

    out, _ = capfd.readouterr()
    assert expected_out == out

    # Test with lower epsilon, and outputs are flipped, but nothing should change
    taps = Taps(get_output_fn_flipped, pitch_shift_func=lambda x, y, z: None, epsilon=0.1)
    taps.print_stats([""], get_labels_fn_flipped)

    out, _ = capfd.readouterr()
    assert expected_out == out

    # Test with even lower epsilon. Something changes
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, epsilon=0.09)
    taps.print_stats([""], get_labels_fn)

    expected_out = (
        """Output Type: frame
Total Outputs: 12
Unsure Outputs: 2 = 16.6667%
Unsure Correct: 1 = 50.0000%
Non-Unsure Outputs: 10 = 83.3333%
Non-Unsure Correct: 10 = 100.0000%
Total Shifted Outputs: 4
Non-Unsure Shifted Outputs: 4 = 100.0000%
Non-Unsure Shifted Correct: 4 = 100.0000%

Output Type: onset
Total Outputs: 12
Unsure Outputs: 2 = 16.6667%
Unsure Correct: 1 = 50.0000%
Non-Unsure Outputs: 10 = 83.3333%
Non-Unsure Correct: 10 = 100.0000%
Total Shifted Outputs: 4
Non-Unsure Shifted Outputs: 4 = 100.0000%
Non-Unsure Shifted Correct: 4 = 100.0000%

"""
    )

    out, _ = capfd.readouterr()
    assert expected_out == out

    # Test with lower epsilon, and outputs are flipped. Same thing changes as previous test
    taps = Taps(get_output_fn_flipped, pitch_shift_func=lambda x, y, z: None, epsilon=0.09)
    taps.print_stats([""], get_labels_fn_flipped)

    out, _ = capfd.readouterr()
    assert expected_out == out

    # Test with custom pitch shifts
    taps = Taps(get_output_fn, pitch_shift_func=lambda x, y, z: None, pitch_shifts=[-1, 2])
    taps.print_stats([""], get_labels_fn)

    expected_out = (
        """Output Type: frame
Total Outputs: 12
Unsure Outputs: 2 = 16.6667%
Unsure Correct: 1 = 50.0000%
Non-Unsure Outputs: 10 = 83.3333%
Non-Unsure Correct: 10 = 100.0000%
Total Shifted Outputs: 2
Non-Unsure Shifted Outputs: 2 = 100.0000%
Non-Unsure Shifted Correct: 2 = 100.0000%

Output Type: onset
Total Outputs: 12
Unsure Outputs: 1 = 8.3333%
Unsure Correct: 1 = 100.0000%
Non-Unsure Outputs: 11 = 91.6667%
Non-Unsure Correct: 10 = 90.9091%
Total Shifted Outputs: 1
Non-Unsure Shifted Outputs: 1 = 100.0000%
Non-Unsure Shifted Correct: 1 = 100.0000%

"""
    )

    out, _ = capfd.readouterr()
    assert expected_out == out
