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

## Data

### Get the Data


I will just check if I can get some usable data from the GDELT 2.0 Event
Database via Google BigQuery.

So far, I didn't decide for which problem to solve and what kind of model to
train, because I want to find a suitable dataset first.

Here's what I'm looking for data:
- with time stamps that's updated regularly, so I can train an initial model
and then schedule it to run periodically and monitor it
- that's sufficiently large to train a model on
- that has some interesting features and a suitable target variable

During a brief search, I found the GDELT 2.0 Event Database, which is a public
and free database that contains event data from all over the world.
It seems to fulfill these requirements and is available via BigQuery.

Here, I will check if I can get some data from it and if it's suitable for my
needs.

Data is downloaded using Google BigQuery.

You need to install the Google Cloud CLI.
FIXME: Include instructions or at least a link.

Then adapt the `.env` file to your project ID. Use `.env.example` as a template.
Then run the setup script I wrote: `./scripts/setup_gcloud.sh`

Finally, download and save the actual data using the script I wrote:
`.scripts/download_data_with_BigQuery.py`.


Get 10k random rows from GDELT events table from year 2024.

I decided to go for a sample size 10k rows, because that should be an acceptable
balance between speed of model training and showing it enough data.
If I go for an 80:20 train:test split, I will end up with 8k rows for training
and 2k rows for testing.
There are just 24 features and one target variable.
So basically the ration rows to features is 10000:24, which is 416.67.
I intend to use tree based algorithms such as XGBoost, CatBoost and LightGBM.
They are rather data efficient, and at this ratio, maybe it's even already
enough for acceptable performance.

Honestly, I could go for **much** more than that though, but then models would
train much longer, too.
This is some sort of a subset for speed of development.
At the same time, I could have also gone for much less than that, but then it
would definitely become a true subset, and whatever I train would likely be
underperforming.
So I decide to go with this as a compromise and check how well it performs.
If it does good enough, I won't need to go for a larger subset.
If it doesn't perform well, I can at least select hyperparameters and then go
for a larger subset.
Then again, this is not a machine learning engineering course, but a machine
learning *operations*, so I don't need to get the best possible model in the
first place.
A good model is sufficient.


Split the data first to prevent data leakage.
Make a truly unseen hold out test set, which will not be used for training or
validation at all.
It will only be used to evaluate one single final model in the very end.

I will use a 80:20 split for training and testing.
This will leave me with 8k rows for training and 2k rows for testing.
For development, I will use 5-fold cross validation.



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
   # first time setup (builds images)
   docker-compose up --build -d
   
   # subsequent starts (uses existing images)
   docker-compose up -d
   ```

4. Access MLflow UI at: `http://localhost:5001`

5. Stop the services:

   ```bash
   docker-compose down
   ```

## Environment Variables

Create a `.env` file with these variables:
- `POSTGRES_USER` - Database username
- `POSTGRES_PASSWORD` - Database password (choose a secure one)
- `POSTGRES_DB` - `mlflow`




