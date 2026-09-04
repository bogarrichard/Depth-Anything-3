# DA3-Streaming

Memory-efficient inference for long videos and large-scale scenes via chunk
streaming, built on [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long).

Full setup instructions, output format, and benchmark results:
**[bogarrichard.github.io/Depth-Anything-3/streaming.html](https://bogarrichard.github.io/Depth-Anything-3/streaming.html)**

Quick start:

```bash
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
pip install -e ".[streaming]"
bash ./scripts/download_weights.sh
python da3_streaming.py --image_dir ./path_of_images
```
