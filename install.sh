set -e

cargo -V || (echo "Install cargo: https://doc.rust-lang.org/cargo/getting-started/installation.html" && exit 1)
magick -version || (echo "Install magick: https://imagemagick.org/script/download.php" && exit 1)
pngquant -V || cargo binstall -y pngquant
uv -V || cargo binstall -y uv --git https://github.com/astral-sh/uv
uv pip install --system .
