import yaml


def read_yaml(file_path):
    with open(file_path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    file_path = "data/configs/learn_yaml.yaml"
    data = read_yaml(file_path)
    print(data)
    print("Data loaded successfully.")