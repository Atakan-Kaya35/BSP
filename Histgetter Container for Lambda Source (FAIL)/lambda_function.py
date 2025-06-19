import os
import json
import boto3
from playwright.sync_api import sync_playwright

def run_playwright(username, password, playwright) -> None:
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--single-process",
            "--disable-gpu",
            "--user-data-dir=/tmp/user-data-dir"
        ],
        env={
            "HOME": "/tmp",
            "TMPDIR": "/tmp"
        }
    )
  
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://clarity.dexcom.eu/")
    page.get_by_role("button", name="Dexcom Clarity for Home Users").click()
    page.get_by_label("Email or username").fill(username)
    page.get_by_label("Email or username").press("Tab")
    page.locator("#password").fill(password)
    page.locator("#password").press("Enter")
    page.get_by_role("button", name="Dışa Aktar").click()
    page.locator("#ember50").get_by_label("Bir tarih aralığı seçin").click()
    page.get_by_role("button", name="90").click()

    print("pulse")
    with page.expect_download() as download_info:
        page.locator("#ember50").get_by_role("button", name="Dışa Aktar").click()

    try:
        download = download_info.value
        suggested_filename = download.suggested_filename
        download_path = './' + username + "." + suggested_filename.split(".")[-1]
        download.save_as(download_path)
        print("Download succeeded:", download_path)
    except Exception as e:
        print("Download failed:", e)
        raise e

    print("pulse")
    context.close()
    browser.close()

def upload_to_s3(username, source_file_name=None, destination_blob_name=None, bucket_name="bspuserartifacts"):
    if source_file_name is None:
        source_file_name = f"{username}.csv"
    print("pulse")

    if destination_blob_name is None:
        destination_blob_name = f"{username}/{source_file_name}"
        
    print("here")
    s3_client = boto3.client('s3')
    print("here")
    s3_client.upload_file(source_file_name, bucket_name, destination_blob_name)

    print(f"File {source_file_name} uploaded to s3://{bucket_name}/{destination_blob_name}.")

def lambda_handler(event, context):
    # Parse username/password from event
    body = json.loads(event['body'])
    username = body.get('username')
    password = body.get('password')

    if not username or not password:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing username or password"})
        }

    # Do the work
    with sync_playwright() as playwright:
        run_playwright(username, password, playwright)

    upload_to_s3(username)

    # Return success response
    return {
        "statusCode": 200,
        "body": json.dumps({"success": "Playwright script executed successfully!"})
    }
