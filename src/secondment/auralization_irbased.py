from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, wavfile
from scipy.interpolate import PchipInterpolator
from scipy.signal import fftconvolve, hilbert, resample_poly

SUPPORTED_SAMPLE_RATES = (
    8000,
    11025,
    12000,
    16000,
    22050,
    24000,
    32000,
    44100,
    48000,
    88200,
    96000,
    176400,
    192000,
)


@dataclass(slots=True)
class ProcessingSampleRateDebug:
    input_rate: int
    max_frequency: float
    required_rate: int
    processing_rate: int
    selection_mode: str
    resampled: bool


@dataclass(slots=True)
class AuralizationResult:
    input_signal: np.ndarray
    output_signal: np.ndarray
    sample_rate: int
    impulse_response: np.ndarray
    impulse_time: np.ndarray
    frequency_bins: np.ndarray
    interpolated_magnitude: np.ndarray
    sample_rate_debug: ProcessingSampleRateDebug
    energy_preservation_factor: float


def ensure_mono(signal_in: np.ndarray) -> np.ndarray:
    signal_array = np.asarray(signal_in, dtype=np.float64)
    if signal_array.ndim == 1:
        return signal_array
    if signal_array.ndim == 2:
        return signal_array.mean(axis=1)
    raise ValueError("Audio signal must be 1D or 2D.")


def ensure_even_length(signal_in: np.ndarray) -> np.ndarray:
    signal_array = np.asarray(signal_in, dtype=np.float64)
    if signal_array.size % 2:
        return signal_array[:-1]
    return signal_array


def resample_audio(signal_in: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
    if int(input_rate) == int(output_rate):
        return np.asarray(signal_in, dtype=np.float64)
    ratio = Fraction(int(output_rate), int(input_rate)).limit_denominator()
    return resample_poly(np.asarray(signal_in, dtype=np.float64), ratio.numerator, ratio.denominator)


def describe_processing_sample_rate(
    max_frequency: float,
    input_rate: int,
    processing_rate: int | None = None,
    supported_rates: tuple[int, ...] = SUPPORTED_SAMPLE_RATES,
) -> ProcessingSampleRateDebug:
    max_frequency = float(max_frequency)
    input_rate = int(input_rate)
    required_rate = int(np.ceil(max(2.0 * max_frequency, 1.0)))

    if processing_rate is None:
        selected_rate = input_rate
        for rate in supported_rates:
            # Select the highest supported rate that is at least twice the maximum frequency and not higher than the input rate.
            if rate >= required_rate and rate <= input_rate:
                selected_rate = int(rate)
                break
        selection_mode = "auto"
    else:
        selected_rate = int(processing_rate)
        selection_mode = "manual"

    return ProcessingSampleRateDebug(
        input_rate=input_rate,
        max_frequency=max_frequency,
        required_rate=required_rate,
        processing_rate=selected_rate,
        selection_mode=selection_mode,
        resampled=selected_rate != input_rate,
    )


def choose_processing_sample_rate(
    max_frequency: float,
    input_rate: int,
    supported_rates: tuple[int, ...] = SUPPORTED_SAMPLE_RATES,
) -> int:
    return describe_processing_sample_rate(
        max_frequency=max_frequency,
        input_rate=input_rate,
        supported_rates=supported_rates,
    ).processing_rate


def response_to_magnitude(response: np.ndarray, response_kind: str = "amplitude") -> np.ndarray:
    kind = response_kind.lower()
    if kind in {"amplitude", "magnitude"}:
        magnitude = np.abs(np.asarray(response).reshape(-1))
    elif kind == "power":
        power = np.real(np.asarray(response).reshape(-1))
        magnitude = np.sqrt(np.clip(power, 0.0, None))
    else:
        raise ValueError("response_kind must be 'amplitude', 'magnitude', or 'power'.")
    return np.maximum(magnitude.astype(np.float64), 1e-12)


def _prepare_frequency_samples(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    target_nyquist: float,
    minimum_magnitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    source_magnitude = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    if source_freqs.size != source_magnitude.size:
        raise ValueError("Frequency and response arrays must have the same length.")

    order = np.argsort(source_freqs)
    source_freqs = source_freqs[order]
    source_magnitude = np.maximum(source_magnitude[order], minimum_magnitude)

    source_freqs, unique_idx = np.unique(source_freqs, return_index=True)
    source_magnitude = source_magnitude[unique_idx]

    if source_freqs[0] > 0.0:
        source_freqs = np.insert(source_freqs, 0, 0.0)
        source_magnitude = np.insert(source_magnitude, 0, source_magnitude[0])

    if target_nyquist > source_freqs[-1]:
        source_freqs = np.append(source_freqs, target_nyquist)
        source_magnitude = np.append(source_magnitude, minimum_magnitude)

    return source_freqs, source_magnitude


def interpolate_log_magnitude(
    source_freqs: np.ndarray,
    source_magnitude: np.ndarray,
    target_freqs: np.ndarray,
    minimum_magnitude: float = 1e-12,
) -> np.ndarray:
    prepared_freqs, prepared_magnitude = _prepare_frequency_samples(
        source_freqs,
        source_magnitude,
        float(np.max(target_freqs)),
        minimum_magnitude,
    )

    if prepared_freqs.size == 1:
        return np.full_like(target_freqs, prepared_magnitude[0], dtype=np.float64)

    interpolator = PchipInterpolator(
        prepared_freqs,
        np.log10(prepared_magnitude),
        extrapolate=False,
    )
    interpolated_log = interpolator(target_freqs)
    interpolated_log = np.where(
        np.isnan(interpolated_log),
        np.log10(minimum_magnitude),
        interpolated_log,
    )
    return np.power(10.0, interpolated_log)


def _one_sided_to_two_sided(one_sided: np.ndarray, n_samples: int) -> np.ndarray:
    spectrum = np.asarray(one_sided, dtype=np.float64)
    if n_samples % 2 == 0:
        return np.concatenate([spectrum, spectrum[-2:0:-1]])
    return np.concatenate([spectrum, spectrum[-1:0:-1]])


def minimum_phase_impulse_response(
    one_sided_magnitude: np.ndarray,
    n_samples: int,
    minimum_magnitude: float = 1e-12,
) -> np.ndarray:
    two_sided_magnitude = _one_sided_to_two_sided(one_sided_magnitude, n_samples)
    log_magnitude = np.log(np.maximum(two_sided_magnitude, minimum_magnitude))
    analytic_signal = np.asarray(hilbert(log_magnitude), dtype=np.complex128)
    minimum_phase = -np.imag(analytic_signal)
    minimum_phase_spectrum = two_sided_magnitude * np.exp(1j * minimum_phase)
    return np.fft.ifft(minimum_phase_spectrum).real


def compute_energy_preservation_factor(
    reference_signal: np.ndarray,
    filtered_signal: np.ndarray,
    minimum_energy: float = 1e-18,
) -> float:
    reference_energy = np.sum(np.square(np.asarray(reference_signal, dtype=np.float64)), dtype=np.float64)
    filtered_energy = np.sum(np.square(np.asarray(filtered_signal, dtype=np.float64)), dtype=np.float64)
    if reference_energy <= minimum_energy or filtered_energy <= minimum_energy:
        return 1.0
    return float(np.sqrt(reference_energy / filtered_energy))


def auralize_with_frf(
    audio_signal: np.ndarray,
    input_rate: int,
    frf_freqs: np.ndarray,
    frf_response: np.ndarray,
    response_kind: str = "amplitude",
    processing_rate: int | None = None,
    auto_resample: bool = True,
) -> AuralizationResult:
    prepared_signal = ensure_even_length(ensure_mono(audio_signal))
    if prepared_signal.size == 0:
        raise ValueError("Audio signal is empty after preprocessing.")

    frf_freqs = np.asarray(frf_freqs, dtype=np.float64)
    if frf_freqs.size == 0:
        raise ValueError("FRF frequency vector is empty.")

    sample_rate_debug = describe_processing_sample_rate(
        max_frequency=float(np.max(frf_freqs)),
        input_rate=int(input_rate),
        processing_rate=processing_rate if not auto_resample else None,
    )

    if processing_rate is None:
        if auto_resample:
            processing_rate = sample_rate_debug.processing_rate
        else:
            processing_rate = int(input_rate)

    processing_rate = int(processing_rate)
    if processing_rate <= 0:
        raise ValueError("processing_rate must be positive.")

    if processing_rate != int(input_rate):
        prepared_signal = resample_audio(prepared_signal, int(input_rate), processing_rate)
        prepared_signal = ensure_even_length(prepared_signal)

    frequency_bins = np.fft.rfftfreq(prepared_signal.size, d=1.0 / processing_rate)
    interpolated_magnitude = interpolate_log_magnitude(
        frf_freqs,
        response_to_magnitude(frf_response, response_kind=response_kind),
        frequency_bins,
    )
    impulse_response = minimum_phase_impulse_response(interpolated_magnitude, prepared_signal.size)
    output_signal = fftconvolve(prepared_signal, impulse_response, mode="full")[: prepared_signal.size]
    energy_preservation_factor = compute_energy_preservation_factor(prepared_signal, output_signal)
    # output_signal = output_signal * energy_preservation_factor
    impulse_time = np.arange(impulse_response.size, dtype=np.float64) / processing_rate

    return AuralizationResult(
        input_signal=prepared_signal.astype(np.float64),
        output_signal=output_signal.astype(np.float64),
        sample_rate=processing_rate,
        impulse_response=impulse_response.astype(np.float64),
        impulse_time=impulse_time,
        frequency_bins=frequency_bins,
        interpolated_magnitude=interpolated_magnitude.astype(np.float64),
        sample_rate_debug=sample_rate_debug,
        energy_preservation_factor=energy_preservation_factor,
    )


def load_pressure_frf_from_mat(
    mat_path: str | Path,
    mic_index: int,
    response_key: str = "pressures",
    freq_key: str = "freq",
) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(mat_path)
    freqs = np.asarray(data[freq_key]).reshape(-1).astype(np.float64)
    responses = np.asarray(data[response_key])
    return freqs, np.asarray(responses[mic_index, :]).reshape(-1)


def normalize_for_wav(signal_in: np.ndarray) -> np.ndarray:
    signal_array = np.asarray(signal_in, dtype=np.float64)
    peak = np.max(np.abs(signal_array))
    if peak < 1e-12:
        return np.zeros(signal_array.shape, dtype=np.int16)
    return (np.clip(signal_array / peak, -1.0, 1.0) * 32767).astype(np.int16)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    filename_audio_time = project_root / "archive" / "SkyExpress_Flight_20231017_Notch207Hz_1-33s_LPF7000Hz_out.wav"
    filename_filter_frf = project_root / "archive" / "Case_3_LRM416Hz_50pm_PressureISOmics_1-7000Hz_SOL108_ModeSet_AddSDamp0p05_AddADamp0p02.mat"
    output_path = project_root / "results" / "testIR.wav"
    mic_index = 18
    sensitivity = 0.77

    print("Loading input audio file...")
    input_fs, input_sound_vector = wavfile.read(filename_audio_time)
    input_signal = ensure_mono(input_sound_vector) * sensitivity
    input_signal = ensure_even_length(input_signal)
    input_time = np.arange(input_signal.size, dtype=np.float64) / input_fs

    print("Loading frequency response function...")
    filter_freqs, filter_frf = load_pressure_frf_from_mat(filename_filter_frf, mic_index=mic_index)

    print("Auralizing with minimum-phase impulse response...")
    result = auralize_with_frf(
        input_signal,
        input_fs,
        filter_freqs,
        filter_frf,
        response_kind="amplitude",
    )
    print(
        "Internal processing sample rate: "
        f"{result.sample_rate_debug.processing_rate} Hz "
        f"({result.sample_rate_debug.selection_mode}, "
        f"input={result.sample_rate_debug.input_rate} Hz, "
        f"max_frf={result.sample_rate_debug.max_frequency:.1f} Hz, "
        f"required>={result.sample_rate_debug.required_rate} Hz)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, result.sample_rate, normalize_for_wav(result.output_signal))

    plt.figure()
    plt.plot(input_time[: result.input_signal.size], result.input_signal)
    plt.legend(["Input time signal"])
    plt.xlabel("Time [s]")
    plt.ylabel("x(t) [-]")
    plt.title("Input Signal")
    plt.grid(True)

    plt.figure()
    plt.semilogy(filter_freqs, np.abs(filter_frf), label="Imported filter FRF")
    plt.semilogy(result.frequency_bins, result.interpolated_magnitude, "--", label="Interpolated magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("FRF [-]")
    plt.title("Frequency Response Function")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(result.impulse_time, result.impulse_response)
    plt.xlabel("Time [s]")
    plt.ylabel("Impulse Response")
    plt.title("Minimum-Phase Impulse Response")
    plt.grid(True)

    plt.figure()
    output_time = np.arange(result.output_signal.size, dtype=np.float64) / result.sample_rate
    plt.plot(np.arange(result.input_signal.size, dtype=np.float64) / result.sample_rate, result.input_signal, label="Input")
    plt.plot(output_time, result.output_signal, "r", label="Output")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("y(t) [-]")
    plt.title("Input vs Output Signals")
    plt.grid(True)

    print(f"Exported audio to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()