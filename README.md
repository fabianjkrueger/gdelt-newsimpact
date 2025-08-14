# mlopsproject2
Second project for DataTalksClub MLOps Zoomcamp

## Setup

### Get the Code

1. Clone the repository:

```bash
# clone the repository
git clone git@github.com:fabianjkrueger/mlopsproject2.git
# navigate to the project directory
cd mlopsproject2
```

### Get the Environment

This project uses uv for Python dependency management.

#### Install uv (macOS/Linux)

```bash
# install uv on macOS or Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) or use WSL.

#### Create and activate the environment

```bash
# install the dependencies
uv sync

# activate the environment
source .venv/bin/activate
```

The environment will automatically use Python 3.12 and install all required dependencies as specified in `pyproject.toml`.



