"""Launch the local Gradio demo; --mock needs no model weights or torch."""

import argparse

from promptshift.demo import build_app
from promptshift.paths import get_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(get_paths(args.root).weights, args.device, args.mock).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()
