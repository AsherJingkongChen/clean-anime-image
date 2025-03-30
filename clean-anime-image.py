#!/usr/bin/env python3

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_MODEL_NAME = "small"
MODELS_DIR = Path(__file__).parent.resolve() / "models"
MODELS_NAME = ["large", "medium", "small"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Clean-Anime-Image: Simple and Efficient Anime Image Restoration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s")

    basic_group = parser.add_argument_group("Basic Options")
    basic_group.add_argument(
        "-i",
        "--input",
        type=str,
        default=".",
        help="Directory or path to input image(s)",
    )
    basic_group.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=".",
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
        "-q",
        "--quiet",
        action="store_true",
        help="Hide info messages",
    )
    basic_group.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.0,
        help="Scaling factor (1.0: no scaling but still denoising)",
    )
    basic_group.add_argument(
        "-x",
        "--suffix",
        type=str,
        default=".out",
        help="Suffix for output image(s) with .png extension ('': no suffix)",
    )

    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--border-padding",
        type=int,
        default=4,
        help="Image border padding. Increase it to reduce border artifacts.",
    )
    advanced_group.add_argument(
        "--device",
        type=str,
        help="Inference device (cpu, cuda, mps) (default: auto-detect)",
    )
    advanced_group.add_argument(
        "--precise",
        action="store_true",
        help="Use higher precision for Real-ESRGAN (slower)",
    )
    advanced_group.add_argument(
        "--tile",
        type=int,
        default=192,
        help="Tile size (0: disable). Decrease it to reduce memory usage.",
    )
    advanced_group.add_argument(
        "--tile-padding",
        type=int,
        default=16,
        help="Tile overlap size. Increase it to reduce tile artifacts.",
    )

    args = parser.parse_args()

    if args.tile == 0:
        args.tile_padding = 0
        args.border_padding = 0

    return args


def fetch_model_path(arg_model: str, url: str):
    from basicsr.utils.download_util import load_file_from_url

    try:
        return load_file_from_url(
            url=url,
            model_dir=str(MODELS_DIR),
            progress=True,
            file_name=f"{arg_model}.pth",
        )
    except Exception as e:
        raise OSError(
            f"{e}\nFailed to download model '{arg_model}' from '{url}'"
        ) from e


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
        MODELS_NAME[0]: {
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
        MODELS_NAME[1]: {
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
        MODELS_NAME[2]: {
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
            f"Invalid name (--model '{args.model}'). Try one of '{'\', '.join(MODELS_NAME)}'"
        )

    device = select_device(args.device)
    model_instance = config["model_cls"](**config["params"])
    model_path = fetch_model_path(args.model, config["url"])
    model = RealESRGANer(
        device=device,
        dni_weight=None,
        half=not args.precise,
        model=model_instance,
        model_path=str(model_path),
        pre_pad=args.border_padding,
        scale=config["scale"],
        tile=args.tile,
        tile_pad=args.tile_padding,
    )
    return model


def find_input_image_files(arg_input: str):
    image_exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]
    input_file = Path(arg_input)
    if input_file.is_file() and input_file.suffix.lower() in image_exts:
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
    raise ValueError(f"No valid directory or image file found (--input '{arg_input}')")


def process_image(
    input_file: Path,
    output_file: Path,
    scale: float,
    upscaler,
):
    import cv2

    try:
        logging.info("Running ImageMagick")
        subprocess.check_call(
            [
                "magick",
                str(input_file),
            ]
            + "-blur 0x1 -sharpen 0x1 -despeckle -statistic median 4 -wavelet-denoise 8%".split()
            + [
                str(output_file),
            ]
        )

        input_img = cv2.imread(str(output_file), cv2.IMREAD_UNCHANGED)
        if input_img is None:
            raise RuntimeError(f"Failed reading '{output_file}'")

        logging.info("Running Real-ESRGAN")
        output_img = upscaler.enhance(input_img, outscale=scale)[0]
        cv2.imwrite(str(output_file), output_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        logging.info("Running pngquant")
        subprocess.check_call(
            "pngquant --force --skip-if-larger --strip --speed 1 --ext .png".split()
            + [str(output_file)],
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"{e}\nFailed processing '{input_file.name}' on calling '{' '.join(e.cmd)}'"
        ) from e
    except Exception as e:
        if "out of memory" in str(e):
            logging.error(
                "Try reducing tile size (--tile) or using CPU (--device cpu)."
            )
        raise RuntimeError(f"{e}\nFailed processing '{input_file.name}'") from e


def main():
    exit_code = 0
    args = parse_arguments()
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    try:
        upscaler = setup_upscaler(args)
        logging.info(f"Running Clean-Anime-Image")
        input_files = find_input_image_files(args.input)
        num_outputs = len(input_files)
        msg_plural = "s" if num_outputs > 1 else ""

        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"Now processing '{num_outputs}' image{msg_plural} from '{input_files[0].parent.resolve()}'"
        )

        for i, input_file in enumerate(input_files, 1):
            output_file = args.output_dir / (input_file.stem + args.suffix + ".png")
            logging.info(
                f"Processing {i} of {num_outputs} image{msg_plural}: "
                f"'{input_file.name}' => '{output_file.name}'"
            )
            process_image(
                input_file,
                output_file,
                args.scale,
                upscaler,
            )

        logging.info(
            f"End processing {num_outputs} image{msg_plural} to directory '{args.output_dir}'"
        )

    except KeyboardInterrupt:
        logging.warning("Interrupting")
    except Exception as e:
        logging.error(e)
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
