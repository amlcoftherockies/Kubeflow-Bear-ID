import boto3
import requests
import zipfile
import io
import os

def seed_demo_bucket(
    dataset_url="https://zenodo.org/records/10892011/files/bear_dataset.zip",
    s3_bucket="bearid-incoming",
    s3_endpoint="http://localhost:8333" # Assuming port-forwarded SeaweedFS
):
    print("Downloading Zenodo dataset...")
    r = requests.get(dataset_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    s3 = boto3.client('s3', endpoint_url=s3_endpoint, aws_access_key_id='any', aws_secret_access_key='any')
    
    print("Extracting and uploading to SeaweedFS...")
    for filename in z.namelist():
        if filename.endswith('.jpg'):
            file_bytes = z.read(filename)
            s3.put_object(Bucket=s3_bucket, Key=filename, Body=file_bytes)
            
    print("Upload complete. Argo Events should now trigger the Kubeflow pipeline.")

if __name__ == "__main__":
    seed_demo_bucket()