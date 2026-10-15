"""Dataset tools, individual inference, benchmarks, and saved-run reports."""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .config import load_config, resolve_config_path
from .datasets import check_images, load_dataset, summarize_dataset
from .models.base import AdapterNotImplementedError
from .models.registry import REGISTRY, get_descriptor
from .schemas.dataset import Dataset
from .schemas.outputs import InferenceResult
from .schemas.common import BoundingBox


def emit(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Fashionista Vision Lab — garment evaluation")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="validate or inspect annotations").add_subparsers(dest="action", required=True)
    for name in ("validate", "inspect"):
        action = dataset.add_parser(name)
        action.add_argument("path", type=Path)
        action.add_argument("--check-images", action="store_true")
        action.add_argument("--image-root", type=Path, default=Path("datasets/images"))
    config = commands.add_parser("config", help="inspect reproducible configuration").add_subparsers(dest="action", required=True)
    show = config.add_parser("show")
    show.add_argument("path", type=Path)
    models = commands.add_parser("models", help="list planned model capabilities").add_subparsers(dest="action", required=True)
    models.add_parser("list")
    schema = commands.add_parser("schema", help="print a JSON Schema")
    schema.add_argument("kind", choices=("dataset", "result", "config"))
    run = commands.add_parser("run", help="run one model on an image and save outputs")
    run.add_argument("--model", required=True)
    run.add_argument("--image", type=Path, required=True)
    run.add_argument("--task", choices=("detection", "attributes", "caption", "embedding"), required=True)
    run.add_argument("--config", type=Path, default=Path("config/default.json"))
    run.add_argument("--output-root", type=Path)
    run.add_argument("--region", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    for name in ("benchmark", "compare"):
        action = commands.add_parser(name, help="score compatible models against labeled data")
        if name == "benchmark":
            action.add_argument("--model", required=True)
        else:
            action.add_argument("--models", nargs="+", required=True)
        action.add_argument("--task", choices=("detection", "attributes"), required=True)
        action.add_argument("--config", type=Path, default=Path("config/default.json"))
        action.add_argument("--dataset", type=Path)
        action.add_argument("--image-root", type=Path)
        action.add_argument("--output-root", type=Path)
        action.add_argument("--detection-mode", choices=("boxes", "categories"), default="boxes")
        action.add_argument("--iou-threshold", type=float, default=0.5)
        action.add_argument("--allow-unverified", action="store_true", help="smoke tests only; label report as unverified")
        action.add_argument("--category-aliases", type=Path, help="JSON mapping of labels to canonical category names")
    report = commands.add_parser("report", help="render completed saved runs without inference")
    report.add_argument("runs", nargs="+", type=Path)
    report.add_argument("--output", type=Path, help="write Markdown to a new file (never overwrite)")
    commands.add_parser("neighbors", help="reserved for Phase 3")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "dataset":
            dataset = load_dataset(args.path)
            errors = check_images(dataset, args.image_root) if args.check_images else []
            emit({"valid": not errors, "images_checked": args.check_images,
                  "summary": summarize_dataset(dataset), "errors": errors})
            return 1 if errors else 0
        if args.command == "config":
            config = load_config(args.path)
            emit({"config": config.model_dump(mode="json"), "fingerprint": config.fingerprint(),
                  "resolved_paths": {key: str(resolve_config_path(args.path, getattr(config, key)))
                                     for key in ("dataset_path", "image_root", "results_dir")}})
        elif args.command == "models":
            emit([REGISTRY[name].as_dict() for name in sorted(REGISTRY)])
        elif args.command == "schema":
            from .config import LabConfig
            emit({"dataset": Dataset, "result": InferenceResult, "config": LabConfig}[args.kind].model_json_schema())
        elif args.command == "run":
            descriptor = get_descriptor(args.model)
            if args.task not in descriptor.planned_capabilities:
                raise ValueError(f"{args.model} has no planned {args.task} capability")
            if descriptor.factory_path is None:
                raise AdapterNotImplementedError(f"{args.model} is planned for Phase {descriptor.phase}")
            from .runner import run_image
            config = load_config(args.config)
            output_root = args.output_root or resolve_config_path(args.config, config.results_dir)
            region = BoundingBox(**dict(zip(("x_min", "y_min", "x_max", "y_max"), args.region))) if args.region else None
            path, result = run_image(args.image, args.model, args.task, config, output_root, region)
            emit({"run_dir": str(path), "result": result.model_dump(mode="json", exclude={"raw_response"})})
            return 1 if result.error else 0
        elif args.command in ("benchmark", "compare"):
            from .runner import BenchmarkProtocol, benchmark
            from .reporting import render_report
            config = load_config(args.config)
            dataset_path = args.dataset or resolve_config_path(args.config, config.dataset_path)
            # Both parser and fingerprint use the same immutable byte snapshot.
            dataset_bytes = dataset_path.read_bytes()
            from .datasets import parse_dataset
            dataset = parse_dataset(dataset_bytes.decode("utf-8"), dataset_path.suffix)
            protocol = BenchmarkProtocol(detection_mode=args.detection_mode, iou_threshold=args.iou_threshold,
                allow_unverified=args.allow_unverified,
                category_aliases=json.loads(args.category_aliases.read_text()) if args.category_aliases else {})
            paths = benchmark(dataset, dataset_bytes, args.image_root or resolve_config_path(args.config, config.image_root),
                [args.model] if args.command == "benchmark" else args.models, args.task, config,
                args.output_root or resolve_config_path(args.config, config.results_dir), protocol)
            if args.command == "compare":
                comparison_path = paths[0] / "comparison.md"
                with comparison_path.open("x", encoding="utf-8") as stream:
                    stream.write(render_report(paths))
            failed = sum(json.loads((path / "metrics.json").read_text())["failed"] for path in paths)
            emit({"run_dirs": [str(path) for path in paths], "failures": failed,
                  "comparison_report": str(paths[0] / "comparison.md") if args.command == "compare" else None})
            return 1 if failed else 0
        elif args.command == "report":
            from .reporting import render_report
            report = render_report(args.runs)
            if args.output:
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(report)
                emit({"report": str(args.output)})
            else:
                print(report)
        else:
            raise AdapterNotImplementedError(f"{args.command} is planned for Phase 3")
        return 0
    except ValidationError as exc:
        # Do not echo invalid values: configuration might accidentally contain secrets.
        errors = [{"location": list(e["loc"]), "type": e["type"]} for e in exc.errors()]
        print(json.dumps({"error": "validation failed", "details": errors}), file=sys.stderr)
        return 1
    except (OSError, ValueError, AdapterNotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, AdapterNotImplementedError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
