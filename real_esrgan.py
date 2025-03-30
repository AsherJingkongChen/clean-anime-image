#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_MODEL_NAME = "medium"
MODELS_DIR = Path(__file__).parent.resolve() / "models"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-ESRGAN: Real-World Super-Resolution via Efficient Upscaling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    basic_group = parser.add_argument_group("Basic Options")
    basic_group.add_argument(
        "-i",
        "--input",
        type=str,
        default="inputs",
        help="Directory or path to input image(s)",
    )
    basic_group.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="outputs",
        help="Directory to output image(s)",
    )
    basic_group.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name",
        metavar="MODEL",
    )
    basic_group.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.0,
        help="Upscaling factor (1.0: no upscaling)",
    )
    basic_group.add_argument(
        "-x",
        "--suffix",
        type=str,
        default="out",
        help="Suffix for output image(s) ('': no suffix)",
    )
    basic_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Hide info messages",
    )

    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--device",
        type=str,
        help="Inference device (cpu, cuda, mps) (default: auto-detect)",
    )
    advanced_group.add_argument(
        "-t",
        "--tile",
        type=int,
        default=192,
        help="Tile size (0: disable). Reduces memory usage.",
    )
    advanced_group.add_argument(
        "--tile-pad",
        type=int,
        default=16,
        help="Tile overlap size (reduces border artifacts).",
    )
    advanced_group.add_argument(
        "--pre-pad",
        type=int,
        default=4,
        help="Image border padding (reduces edge artifacts).",
    )
    advanced_group.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 model precision instead of fp16",
    )

    args = parser.parse_args()

    if args.tile == 0:
        args.tile = None
        args.tile_pad = 0
        args.pre_pad = 0
    return args


def fetch_model_path(arg_model, url):
    from basicsr.utils.download_util import load_file_from_url

    try:
        return load_file_from_url(
            url=url,
            model_dir=str(MODELS_DIR),
            progress=True,
            file_name=f"{arg_model}.pth",
        )
    except Exception as e:
        raise OSError(f"Failed to download model {arg_model} from {url}: {e}") from e


def select_device(arg_device):
    import torch

    if arg_device:
        return arg_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def setup_upscaler(args):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    MODEL_CONFIGS = {
        "large": {
            "model_cls": RRDBNet,
            "params": {
                "num_block": 23,
                "num_feat": 64,
                "num_grow_ch": 32,
                "num_in_ch": 3,
                "num_out_ch": 3,
                "scale": 4,
            },
            "scale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        },
        "medium": {
            "model_cls": RRDBNet,
            "params": {
                "num_block": 6,
                "num_feat": 64,
                "num_grow_ch": 32,
                "num_in_ch": 3,
                "num_out_ch": 3,
                "scale": 4,
            },
            "scale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        },
        "small": {
            "model_cls": SRVGGNetCompact,
            "params": {
                "act_type": "prelu",
                "num_conv": 16,
                "num_feat": 64,
                "num_in_ch": 3,
                "num_out_ch": 3,
                "upscale": 4,
            },
            "scale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        },
    }

    config = MODEL_CONFIGS.get(args.model)
    if not config:
        raise ValueError(
            f"Invalid name (--model {args.model}). Try using --model {', --model '.join(MODEL_CONFIGS.keys())}"
        )

    device = select_device(args.device)
    model_instance = config["model_cls"](**config["params"])
    model_path = fetch_model_path(args.model, config["url"])

    return RealESRGANer(
        device=device,
        dni_weight=None,
        half=not args.fp32,
        model=model_instance,
        model_path=str(model_path),
        pre_pad=args.pre_pad,
        scale=config["scale"],
        tile=args.tile,
        tile_pad=args.tile_pad,
    )


def find_input_images(arg_input):
    input_file = Path(arg_input)
    if input_file.is_file():
        return [input_file]
    if input_file.is_dir():
        images = sorted(
            p
            for p in input_file.glob("*")
            if p.suffix.lower()
            in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]
        )
        if images:
            return images
    raise ValueError(f"No valid directory or image file found (--input {arg_input})")


def process_image(
    input_file: Path,
    output_dir: Path,
    suffix,
    scale,
    upscaler,
):
    import cv2

    try:
        input_img = cv2.imread(str(input_file), cv2.IMREAD_UNCHANGED)
        if input_img is None:
            logging.warning(f"Skipped unreadable file: {input_file.name}")
            return

        output_img, _ = upscaler.enhance(input_img, outscale=scale)

        output_stem = input_file.stem
        output_suffix = "." + suffix if suffix else ""
        output_ext = (
            ".png"
            if input_img.ndim == 3 and input_img.shape[2] == 4
            else input_file.suffix.lower()
        )
        output_file = output_dir / (output_stem + output_suffix + output_ext)

        cv2.imwrite(str(output_file), output_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    except Exception as e:
        logging.error(f"Failed processing {input_file.name}: {e}")
        if "out of memory" in str(e):
            logging.error(
                "Try reducing tile size (--tile) or using CPU (--device cpu)."
            )
        raise


def main():
    exit_code = 0
    args = parse_arguments()
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    try:
        upscaler = setup_upscaler(args)
        logging.info(f"Now running Real-ESRGAN ({args.model}, x{args.scale:.2f})")
        input_files = find_input_images(args.input)
        num_outputs = len(input_files)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"Now processing {num_outputs} image{'s' if num_outputs > 1 else ''} from '{args.input}'"
        )

        for i, input_file in enumerate(input_files, 1):
            if num_outputs > 1:
                logging.info(
                    f"Processing {i} of {num_outputs} images: {input_file.name}"
                )
            process_image(
                input_file,
                args.output_dir,
                args.suffix,
                args.scale,
                upscaler,
            )

        logging.info(
            f"End processing {num_outputs} image{'s' if num_outputs > 1 else ''} to '{args.output_dir}'"
        )

    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    except Exception as e:
        logging.error(f"{e}")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
