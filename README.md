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

## Setup Instructions

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your unique project ID:
   ```env
   GOOGLE_CLOUD_PROJECT=mlops-zoomcamp-yourname
   ```

3. Run the setup script:
   ```bash
   chmod +x scripts/setup_gcloud.sh
   ./scripts/setup_gcloud.sh
   ```

**Note**: Choose a globally unique project ID. If your chosen name is taken, try adding your initials or random numbers.




# MLflow Docker Setup

## Quick Start

### Requirements

- Docker
- Git (optional, but makes it easier to get code via cloning)

### Instructions

1. Clone this repository
2. Create a `.env` file with your database credentials:
   ```bash
   POSTGRES_USER=your_username_here
   POSTGRES_PASSWORD=your_secure_password_here
   POSTGRES_DB=mlflow
   ```
3. Start the services:
   ```bash
   docker-compose up -d
   ```
4. Access MLflow UI at: `http://localhost:5001`

## Environment Variables

Create a `.env` file with these variables:
- `POSTGRES_USER` - Database username
- `POSTGRES_PASSWORD` - Database password (choose a secure one)
- `POSTGRES_DB` - `mlflow`


