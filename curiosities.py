import requests

url = "https://***REDACTED***.execute-api.eu-central-1.amazonaws.com/dev/new-entry"
data = {
    "UserID": "abc123",
    "Timestamp": 4,
    "tags": ["test3", "entry"],
    "text": "Hello world"
}

response = requests.post(url, json=data)  # Make sure it's POST
print(response.status_code)
print(response.text)

url = "https://***REDACTED***.execute-api.eu-central-1.amazonaws.com/dev/get-entry"
params = {
    "UserID": "abc123",
    "Timestamp": 4}

response = requests.get(url, params=params)  # <- GET with params in URL
print(response.status_code)
print(response.text)
