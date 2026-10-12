import argparse
import json
from pathlib import Path
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Multichannel awakening forecasting research workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate clearly marked synthetic signals and annotations")
    demo.add_argument("--output", required=True)
    train = sub.add_parser("train", help="Train four prespecified models and evaluate subject-held-out splits")
    train.add_argument("--manifest", required=True)
    train.add_argument("--config", help="JSON object with Config parameters")
    train.add_argument("--output", required=True, help="New output directory")
    train.add_argument("--false-alarms-per-hour", type=float, default=0.5)
    predict = sub.add_parser("predict", help="Predict with a trusted locally trained joblib model")
    predict.add_argument("--model", required=True)
    predict.add_argument("--signal", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--fs", type=float)
    predict.add_argument("--unit", choices=["V", "uV"])
    predict.add_argument("--time-column")
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import make_demo
        print(make_demo(args.output))
    elif args.command == "train":
        from .model import run_experiment
        cfg = Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
        report = run_experiment(args.manifest, cfg, args.output, args.false_alarms_per_hour)
        print(json.dumps({"synthetic": report["provenance"]["synthetic"],
                          "output": str(Path(args.output).resolve()),
                          "test_events": report["models"]["spectral"]["test"]["events"]}, indent=2))
    else:
        from .model import predict_recording
        result = predict_recording(args.model, args.signal, args.fs, args.unit, args.time_column)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
        print(f"Saved {len(result)} decisions to {path}")


if __name__ == "__main__":
    main()
