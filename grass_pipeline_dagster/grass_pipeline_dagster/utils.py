import json
import os
import boto3
from botocore.client import Config
from influxdb_client import InfluxDBClient

def load_settings() -> dict:
    """
    Load settings from the JSON config file located at:
    C:/Users/Gebruiker/Desktop/GRASS_Pipeline/config/settings.json
    """
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(current_file, "..", "..", ".."))
    config_path = os.path.join(project_root, "config", "settings.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Settings file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)

def get_minio_client():
    """
    Create and return a boto3 client configured for MinIO.
    Uses Docker-internal hostname 'minio' and default credentials.
    """
    return boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

def get_influx_client():
    """
    Create and return an InfluxDB v2 client.
    Assumes Docker-internal hostname 'influxdb' and default setup token.
    """
    return InfluxDBClient(
        url="http://influxdb:8086",
        token="mytoken",              # Use the same token as in docker-compose.yml
        org="grass"
    )
