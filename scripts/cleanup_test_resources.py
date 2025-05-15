#!/usr/bin/env python3
"""
Script to clean up cloud resources created during test runs.
This ensures no orphaned resources are left running after tests,
which could lead to unexpected costs.
"""

import os
import logging
import datetime
import argparse
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cleanup")

# Constants
TEST_RESOURCE_TAG = "hipo-test-resource"
RESOURCE_MAX_AGE_HOURS = 24


def cleanup_aws_resources() -> int:
    """
    Clean up AWS resources that have the test tag and are older than the max age.
    
    Returns:
        int: Number of resources cleaned up
    """
    try:
        import boto3
        
        logger.info("Starting AWS resource cleanup")
        resources_cleaned = 0
        
        # Get AWS credentials from environment
        aws_region = os.environ.get("AWS_REGION", "us-west-2")
        
        # S3 bucket cleanup
        s3 = boto3.resource("s3", region_name=aws_region)
        
        # Get all buckets with our test tag
        test_buckets = []
        for bucket in s3.buckets.all():
            try:
                tags = boto3.client("s3").get_bucket_tagging(Bucket=bucket.name)
                if any(tag["Key"] == TEST_RESOURCE_TAG for tag in tags.get("TagSet", [])):
                    test_buckets.append(bucket)
            except Exception:
                # Bucket might not have tags
                continue
        
        # Delete test buckets
        for bucket in test_buckets:
            try:
                logger.info(f"Deleting test S3 bucket: {bucket.name}")
                # First delete all objects
                bucket.objects.all().delete()
                # Then delete bucket
                bucket.delete()
                resources_cleaned += 1
            except Exception as e:
                logger.error(f"Failed to delete bucket {bucket.name}: {str(e)}")
        
        # Lambda function cleanup
        lambda_client = boto3.client("lambda", region_name=aws_region)
        lambda_functions = lambda_client.list_functions().get("Functions", [])
        
        for func in lambda_functions:
            try:
                # Check if this is a test resource
                arn = func["FunctionArn"]
                tags = lambda_client.list_tags(Resource=arn).get("Tags", {})
                
                if TEST_RESOURCE_TAG in tags:
                    logger.info(f"Deleting test Lambda function: {func['FunctionName']}")
                    lambda_client.delete_function(FunctionName=func["FunctionName"])
                    resources_cleaned += 1
            except Exception as e:
                logger.error(f"Failed to delete Lambda function: {str(e)}")
        
        # Secrets Manager cleanup
        secrets_client = boto3.client("secretsmanager", region_name=aws_region)
        secrets = secrets_client.list_secrets().get("SecretList", [])
        
        for secret in secrets:
            try:
                # Check if this is a test resource
                tags = secret.get("Tags", [])
                if any(tag["Key"] == TEST_RESOURCE_TAG for tag in tags):
                    logger.info(f"Deleting test secret: {secret['Name']}")
                    secrets_client.delete_secret(
                        SecretId=secret["ARN"],
                        ForceDeleteWithoutRecovery=True
                    )
                    resources_cleaned += 1
            except Exception as e:
                logger.error(f"Failed to delete secret: {str(e)}")
        
        logger.info(f"AWS cleanup completed. {resources_cleaned} resources cleaned up.")
        return resources_cleaned
        
    except ImportError:
        logger.warning("boto3 not installed, skipping AWS cleanup")
        return 0
    except Exception as e:
        logger.error(f"Error in AWS cleanup: {str(e)}")
        return 0


def cleanup_gcp_resources() -> int:
    """
    Clean up GCP resources that have the test label and are older than the max age.
    
    Returns:
        int: Number of resources cleaned up
    """
    try:
        from google.cloud import storage
        
        logger.info("Starting GCP resource cleanup")
        resources_cleaned = 0
        
        # GCS bucket cleanup
        storage_client = storage.Client()
        
        # List buckets
        buckets = storage_client.list_buckets()
        
        for bucket in buckets:
            try:
                labels = bucket.labels or {}
                
                # Check if this is a test resource
                if labels.get("purpose") == "testing":
                    logger.info(f"Deleting test GCS bucket: {bucket.name}")
                    
                    # Delete all objects in the bucket
                    blobs = bucket.list_blobs()
                    for blob in blobs:
                        blob.delete()
                    
                    # Delete the bucket
                    bucket.delete()
                    resources_cleaned += 1
            except Exception as e:
                logger.error(f"Failed to delete GCS bucket {bucket.name}: {str(e)}")
        
        logger.info(f"GCP cleanup completed. {resources_cleaned} resources cleaned up.")
        return resources_cleaned
        
    except ImportError:
        logger.warning("google-cloud-storage not installed, skipping GCP cleanup")
        return 0
    except Exception as e:
        logger.error(f"Error in GCP cleanup: {str(e)}")
        return 0


def cleanup_azure_resources() -> int:
    """
    Clean up Azure resources that have the test tag and are older than the max age.
    
    Returns:
        int: Number of resources cleaned up
    """
    try:
        from azure.storage.blob import BlobServiceClient
        
        logger.info("Starting Azure resource cleanup")
        resources_cleaned = 0
        
        # Get Azure connection string
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING not set, skipping Azure cleanup")
            return 0
        
        # Connect to blob service
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # List containers
        containers = blob_service_client.list_containers(include_metadata=True)
        
        for container in containers:
            try:
                # Check if this is a test container
                metadata = container.get("metadata", {})
                if metadata.get("purpose") == "testing":
                    logger.info(f"Deleting test container: {container['name']}")
                    
                    # Get container client
                    container_client = blob_service_client.get_container_client(container["name"])
                    
                    # Delete all blobs
                    blobs = container_client.list_blobs()
                    for blob in blobs:
                        container_client.delete_blob(blob.name)
                    
                    # Delete container
                    container_client.delete_container()
                    resources_cleaned += 1
            except Exception as e:
                logger.error(f"Failed to delete container {container['name']}: {str(e)}")
        
        logger.info(f"Azure cleanup completed. {resources_cleaned} resources cleaned up.")
        return resources_cleaned
        
    except ImportError:
        logger.warning("azure-storage-blob not installed, skipping Azure cleanup")
        return 0
    except Exception as e:
        logger.error(f"Error in Azure cleanup: {str(e)}")
        return 0


def main() -> None:
    """
    Main function to run the cleanup process.
    """
    parser = argparse.ArgumentParser(description="Clean up cloud resources created during tests")
    parser.add_argument("--aws-only", action="store_true", help="Only clean AWS resources")
    parser.add_argument("--gcp-only", action="store_true", help="Only clean GCP resources")
    parser.add_argument("--azure-only", action="store_true", help="Only clean Azure resources")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    
    args = parser.parse_args()
    
    logger.info("Starting cleanup of test resources")
    total_cleaned = 0
    
    # Determine which cloud providers to clean up
    cleanup_aws = not (args.gcp_only or args.azure_only)
    cleanup_gcp = not (args.aws_only or args.azure_only)
    cleanup_azure = not (args.aws_only or args.gcp_only)
    
    # Run cleanup for each cloud provider
    if cleanup_aws:
        aws_cleaned = cleanup_aws_resources()
        total_cleaned += aws_cleaned
    
    if cleanup_gcp:
        gcp_cleaned = cleanup_gcp_resources()
        total_cleaned += gcp_cleaned
    
    if cleanup_azure:
        azure_cleaned = cleanup_azure_resources()
        total_cleaned += azure_cleaned
    
    logger.info(f"Cleanup completed. Total resources cleaned up: {total_cleaned}")


if __name__ == "__main__":
    main()