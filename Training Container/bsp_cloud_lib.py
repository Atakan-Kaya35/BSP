import boto3
import os
from config import Config

class Cloud_Storage:
    def upload_to_s3(s3_key, local_path, bucket_name = Config.S3_BUCKET):
        """
        Uploads a file from the local system to a specified S3 bucket and key.

        Args:
            bucket_name (str): Name of the S3 bucket.
            local_path (str): Local file path to upload.
            s3_key (str): S3 object key (including folders if needed).

        Returns:
            None
        """
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")

    def download_from_s3(s3_key, local_path, bucket_name = Config.S3_BUCKET):
        """
        Downloads a file from S3 to a specified local path.

        Args:
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): S3 object key to download.
            local_path (str): Destination path on local filesystem.

        Returns:
            None
        """
        #TODO: local testing erase
        # Ensure the parent directory exists
        #os.makedirs(os.path.dirname(local_path), exist_ok=True)


        s3 = boto3.client('s3')
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Downloaded s3://{bucket_name}/{s3_key} to {local_path}")

    def list_files_in_s3_folder(bucket_name, folder_prefix):
        """
        Lists all files in a specific S3 bucket folder.

        Args:
            bucket_name (str): Name of the S3 bucket.
            folder_prefix (str): S3 folder prefix (e.g., 'user123/models/')

        Returns:
            list: List of S3 object keys within the folder.
        """
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
        return []

    def delete_from_s3(bucket_name, s3_key):
        """
        Deletes a specific object from an S3 bucket.

        Args:
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): S3 object key to delete.

        Returns:
            None
        """
        s3 = boto3.client('s3')
        s3.delete_object(Bucket=bucket_name, Key=s3_key)
        print(f"Deleted s3://{bucket_name}/{s3_key}")

    # Optional: a utility to ensure a local folder exists for downloading
    #os.makedirs("/tmp/downloads", exist_ok=True)

    # Example usage (commented out):
    # upload_to_s3("my-bucket", "model.h5", "user123/model.h5")
    # download_from_s3("my-bucket", "user123/model.h5", "./model.h5")
