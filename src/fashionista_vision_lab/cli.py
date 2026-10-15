"""Small argparse CLI. Phase 1 does not run inference or emit benchmark scores."""

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


def emit(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Fashionista Vision Lab — Phase 1 foundations")
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
    run = commands.add_parser("run", help="reserved for Phase 2/3 inference")
    run.add_argument("--model", required=True)
    run.add_argument("--image", type=Path, required=True)
    run.add_argument("--task", choices=("detection", "attributes", "embedding"), required=True)
    for name in ("benchmark", "compare", "report", "neighbors"):
        commands.add_parser(name, help="reserved for a later phase (not implemented)")
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
            raise AdapterNotImplementedError(f"{args.model} inference is planned for Phase {descriptor.phase}; no model was loaded")
        else:
            raise AdapterNotImplementedError(f"{args.command} is not implemented in Phase 1; no benchmark results were generated")
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
