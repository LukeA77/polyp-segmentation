"""polyp-segmentation: reference pipeline template for the AI Surgical & Diagnostic
Assistant portfolio.
"""

import sys

import truststore

# Use the OS-native certificate trust store for TLS verification instead of the
# bundled certifi list: on networks with a corporate root CA (trusted by Windows,
# absent from certifi), plain HTTPS calls otherwise fail to verify. Applied once here
# (package import time) so it covers every src submodule that makes an outbound HTTPS
# call (dataset download, pretrained-weight download, W&B, ...), not just the first one.
truststore.inject_into_ssl()

# Windows' default console codepage (cp1252) can't encode Unicode status characters
# (e.g. torch.onnx's exporter prints "✅ ..."), which otherwise crashes any dependency
# that prints them. Force UTF-8 so a third-party print statement can't take down the
# pipeline over a single emoji.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
