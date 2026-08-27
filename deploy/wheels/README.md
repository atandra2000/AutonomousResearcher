`*.whl` files here are vendored build-time dependencies (CPU-only torch for
linux/arm64), intentionally NOT committed. Refresh with:

    curl -fL -o "deploy/wheels/torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
      "https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl"

The Dockerfile installs them before the project deps so sentence-transformers
satisfies its torch requirement without pulling CUDA packages.
