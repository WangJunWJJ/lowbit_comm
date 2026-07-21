import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    from .model import load_qwen2_text_model

    model = load_qwen2_text_model(args.model_path)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model_type={model.config.model_type} parameters={parameters}")


if __name__ == "__main__":
    main()
