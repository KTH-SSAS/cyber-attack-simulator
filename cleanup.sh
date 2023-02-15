autoflake --in-place . -r --remove-all-unused-imports # Remove unused imports
docformatter . -r --in-place # Docstring formatting
black .
isort .
# flakehell lint .
# pydocstyle .
#mypy src/attack_simulator
#pytest --cov .
