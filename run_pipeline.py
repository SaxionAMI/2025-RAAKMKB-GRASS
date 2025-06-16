import argparse
import json
from pipeline.controller import PipelineRunner

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.json", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = PipelineRunner(
        input_dir=config["input_dir"],
        output_dir=config["output_dir"],
        config=config
    )
    runner.run()

if __name__ == "__main__":
    main()
