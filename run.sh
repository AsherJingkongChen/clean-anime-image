set -e

IN=${1:-input.png}
OUT=${2:-output.png}

magick "$IN" -blur 0x1 -sharpen 0x1 -despeckle -statistic median 3 -wavelet-denoise 10% "$OUT"
./real_esrgan.py -i "$OUT" -x '' -o '' -m small
pngquant --force --skip-if-larger --strip --speed 1 --ext .png "$OUT"
