# gdelt-newsimpact

MLOps project predicting news media coverage from GDELT event data.

## Problem Description

This project predicts **NumArticles** - the count of unique news articles that report on events tracked by GDELT. 

### What is NumArticles?

NumArticles represents the scope and intensity of media coverage for global events. It's a real-world signal of an event's newsworthiness, capturing how many unique news articles report on a particular event. Each article is counted only once, regardless of how many times the event is mentioned within it or which outlet publishes it.

NumArticles is distinct from other media metrics:
- **NumMentions**: Could be inflated by repeated reporting or commentary
- **NumSources**: Describes coverage diversity rather than volume

High NumArticles typically indicates events considered globally or regionally significant, urgent, or especially fit for public consumption.

### Problem Value

By predicting NumArticles from event attributes, we answer questions like:
- "Given who is involved, what the event is, and where it takes place, how widespread will the news coverage be?"
- "Which kinds of events will turn into major talking points the moment they happen?"
- "Does the origin (actors or geography) or timing of an event affect its likelihood of getting more reporting?"

**Applications:**
- **Media Impact Forecasting**: Organizations can anticipate stories requiring rapid response or public relations resources
- **Event Prioritization**: Governments, NGOs, and journalists can predict which incidents will dominate news cycles
- **Trend Analysis**: Researchers can investigate patterns in what makes news "go viral" or why some events get under-reported
- **Early Warning Systems**: Proactive management of communication, resources, or interventions for expected high-profile events
- **Understanding Bias**: Explore why certain regions, actor groups, or event types get more attention

### Features

The model uses intrinsic event characteristics (not downstream media outcomes):

- **Event Type & Impact**: EventCode, QuadClass, GoldsteinScale - defines what happened and its significance
- **Actor Details**: Actor1/2 codes, names, countries, types - identifies prominent participants
- **Location Info**: Country codes, lat/long, administrative regions - regional context affects coverage
- **Timing Features**: Date, day of week, seasonality - captures news cycles and timing effects

## Setup

### Requirements
- Docker
- Python 3.12+ with uv
- Google Cloud CLI

### Installation

1. Clone and setup environment:
```bash
git clone git@github.com:fabianjkrueger/gdelt-newsimpact.git
cd gdelt-newsimpact

# install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync
source .venv/bin/activate
```

2. Configure Google Cloud:
```bash
cp .env.example .env
# edit .env and set GOOGLE_CLOUD_PROJECT=your-unique-project-id
chmod +x scripts/setup_gcloud.sh
./scripts/setup_gcloud.sh
```

3. Start services:
```bash
docker-compose up --build -d
```

## Usage

### Data Pipeline

The entire pipeline is orchestrated using **Prefect** and deployed in Docker containers:

```bash
# download data from BigQuery
uv run python scripts/download_data_with_BigQuery.py

# prepare data for modeling
uv run python scripts/prepare_data_for_modeling.py --train --query --subset

# train models
uv run python scripts/develop_models.py
```

Prefect handles workflow scheduling, dependency management, and automated pipeline execution. The flows are containerized and can be triggered on schedule or demand.

### Model Serving

Access MLflow UI at `http://localhost:5001`

### API Examples

**Single prediction:**
```bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "QuadClass": 3,
    "GoldsteinScale": -2.5,
    "ActionGeo_Lat": 45.5,
    "ActionGeo_Long": -75.2,
    "EventCode": 120.0,
    "EventBaseCode": 12.0,
    "EventRootCode": 1.0,
    "Actor1Code": 100.0,
    "Actor1Name": 50.0,
    "Actor1CountryCode": 10.0,
    "ActionGeo_CountryCode": 30.0,
    "year": 2024,
    "month": 8,
    "day_of_year": 230,
    "day_of_week": 3,
    "is_weekend": 0
  }'
```

**Batch prediction:**
```bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'  # array of prediction objects
```

## Monitoring

Model performance and data drift monitoring is implemented using **Evidently**. This includes tracking of prediction quality, feature drift, and target drift over time.

**Note**: Monitoring implementation is currently under construction. A basic version is available that logs metrics to the monitoring database, but dashboards and alerting are still being developed.

### Services
- MLflow: `http://localhost:5001` 
- Model API: `http://localhost:5002`
- Grafana: Check docker-compose.yaml for port
- Prefect UI: Check docker-compose.yaml for port

### Stop Services
```bash
docker-compose down
```