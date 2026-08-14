# Sample images

These are the images from OpenCV's `samples/data/`, converted to PNG.

The conversion is lossless: each file holds the exact pixels OpenCV decodes the
original into, so a sample reading `fruits.png` here sees what the upstream
sample reading `fruits.jpg` sees. PNG rather than the original format because
this port's PNG decoder is bit-exact against the reference library and its JPEG
decoder is not — see `../src/assets.rs`.

Upstream: https://github.com/opencv/opencv/tree/4.x/samples/data
Licence:  Apache-2.0, the same as the rest of OpenCV and of this port.
