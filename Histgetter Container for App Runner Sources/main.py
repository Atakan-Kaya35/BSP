import os
import boto3
from playwright.sync_api import sync_playwright
from flask import Flask, jsonify

app = Flask(__name__)

def run_playwright(username, password, playwright) -> None:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://clarity.dexcom.eu/")
    page.get_by_role("button", name="Dexcom Clarity for Home Users").click()
    #page.get_by_label("E-mail or username").click()
    page.get_by_label("Email or username").fill(username)
    page.get_by_label("Email or username").press("Tab")
    page.locator("#password").fill(password)
    page.locator("#password").press("Enter")
    #page.locator("input[name=\"op\"]").click()
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
        raise e  # Let Flask return a 500 error
    
    print("pulse")
    context.close()
    browser.close()

def upload_to_s3(username, source_file_name=None, destination_blob_name=None, bucket_name="bspuserartifacts"):
    """
    Uploads created file to determined S3 bucket.

    Args:
        source_file_name: name of the file which is PREFFEREBLY the USERNAME of the owner
        bucket_name: the S3 bucket name
    """
    if source_file_name is None:
        source_file_name = f"{username}.csv"
    print("pulse")

    if destination_blob_name is None:
        destination_blob_name = f"{username}/{source_file_name}"
        
    print("here")
    # Initialize S3 client
    s3_client = boto3.client('s3')
    print("here")
    # Upload file
    s3_client.upload_file(source_file_name, bucket_name, destination_blob_name)

    print(f"File {source_file_name} uploaded to s3://{bucket_name}/{destination_blob_name}.")

@app.route('/run/<string:username>/<string:password>', methods=['GET', 'POST'])
def run(username, password):
    with sync_playwright() as playwright:
        run_playwright(username, password, playwright)

    upload_to_s3(username)
    os.remove(username + ".csv")

    return jsonify({"success": "Playwright script executed successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8080, host="0.0.0.0")
