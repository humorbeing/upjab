# Upjab


../../example_data


# To Do

- [ ] copy to use
  - [ ] args_setup. make a simple version


# pyproject.toml

```toml
[build-system]
requires = ["setuptools", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
authors = [
    {name = "Example Author", email = "author@example.com"}
]
description = "package description"
version = "0.0.1"
readme = "README.md"
requires-python = ">=3.7"
dependencies = [
    "requests > 2.26.0",
    "pandas"
]
```