import argparse
import librosa
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Resample a single WAV file to 24kHz")
    parser.add_argument("--input", required=True, help="Path to the input WAV file")
    parser.add_argument("--output", default=None, help="Output path (defaults to overwriting input)")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Target sample rate (default: 24000)")
    args = parser.parse_args()

    output = args.output or args.input

    audio, sr = librosa.load(args.input, sr=None, mono=True)
    print(f"Original: {sr} Hz")

    if sr == args.sample_rate:
        print(f"Already at {args.sample_rate} Hz, nothing to do.")
        return

    audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sample_rate)
    sf.write(output, audio, args.sample_rate, subtype="PCM_16")
    print(f"Resampled to {args.sample_rate} Hz -> {output}")


if __name__ == "__main__":
    main()