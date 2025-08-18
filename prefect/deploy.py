"""
Prefect deployment configuration for GDELT inference pipeline
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule  # Correct import for Prefect 3.x
from datetime import datetime, timezone
import sys
from pathlib import Path

# add the prefect directory to Python path
sys.path.append(str(Path(__file__).parent))

from flows.inference_pipeline import daily_inference_pipeline


# DEVELOPMENT SCHEDULES
DEV_SCHEDULES = {
    "every_minute": CronSchedule(cron="* * * * *", timezone="UTC"),           # Every minute (for immediate testing)
    "every_15_min": CronSchedule(cron="*/15 * * * *", timezone="UTC"),       # Every 15 minutes (GDELT update frequency)
    "every_hour": CronSchedule(cron="0 * * * *", timezone="UTC"),            # Every hour
    "daily": CronSchedule(cron="0 2 * * *", timezone="UTC"),                 # Daily at 2 AM (production)
}

# CHOOSE YOUR DEVELOPMENT SCHEDULE HERE:
ACTIVE_SCHEDULE = "every_minute"  # Change this to switch schedules


def create_daily_deployment():
    """Create deployment for daily inference pipeline"""
    
    # use the active development schedule
    schedule = DEV_SCHEDULES[ACTIVE_SCHEDULE]
    
    deployment = daily_inference_pipeline.to_deployment(
        name="gdelt-daily-inference-production",
        description=f"GDELT inference pipeline - {ACTIVE_SCHEDULE}",
        tags=["gdelt", "inference", "production", ACTIVE_SCHEDULE],
        schedule=schedule,
        parameters={
            "target_date": None,  # defaults to yesterday
            "limit": 100 if ACTIVE_SCHEDULE in ["every_minute", "every_15_min"] else 10000  # smaller limits for frequent runs
        },
        work_pool_name="default",
    )
    
    return deployment


def create_test_deployment():
    """Create deployment for manual testing (no schedule)"""
    
    deployment = daily_inference_pipeline.to_deployment(
        name="gdelt-daily-inference-test", 
        description="Manual test deployment for GDELT inference pipeline",
        tags=["gdelt", "inference", "test"],
        parameters={
            "target_date": "2024-01-01",  # fixed test date
            "limit": 50  # small test limit
        },
        work_pool_name="default",
        # NO schedule - manual only
    )
    
    return deployment


if __name__ == "__main__":
    print(f"🚀 Starting Prefect worker with {ACTIVE_SCHEDULE} schedule...")
    print(f"📅 Schedule: {DEV_SCHEDULES[ACTIVE_SCHEDULE].cron}")
    
    # serve both deployments
    serve(
        create_daily_deployment(),
        create_test_deployment(),
        limit=5  # max concurrent flow runs
    )
