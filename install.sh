set -e

# magick: https://imagemagick.org/script/download.php
# cargo: https://doc.rust-lang.org/cargo/getting-started/installation.html

magick -version
pngquant -V || cargo binstall -y pngquant
uv -V || cargo binstall -y uv --git https://github.com/astral-sh/uv

uv sync

echo "Please RUN this command: source .venv/bin/activate"
