FROM python:3.12.11-slim

WORKDIR /app

# copy model requirements and install dependencies
COPY mlartifacts/2/models/m-e17ab8ef2ccf46c3bff6e65e7c9509d3/artifacts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# install further dependencies for serving
RUN pip install --no-cache-dir flask==3.0.0 gunicorn==23.0.0

# copy the serving script
COPY scripts/serve_model.py .

# expose the port the app runs on
EXPOSE 5000

# run the serving script
CMD ["gunicorn", "serve_model:app", "--bind", "0.0.0.0:5000", "--workers", "1"]

