import os
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Please set the SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")


# =======================================================================================================
import requests
import json 
from loguru import logger
from time import sleep
import subprocess
import random

ERROR_CODES = {"Invalid access token": 401}


def get_access_token():
    curl_command = [
        "curl", "-s", "-X", "POST", "https://accounts.spotify.com/api/token",
        "-H", "Content-Type: application/x-www-form-urlencoded",
        "-d", "grant_type=client_credentials",
        "-d", f"client_id={CLIENT_ID}",
        "-d", f"client_secret={CLIENT_SECRET}",
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)

    try:
        response_dict = json.loads(result.stdout)
        access_token = response_dict.get('access_token')
        return access_token
    except json.JSONDecodeError:
        logger.error("Error decoding JSON response.")
        return None


def send_request(url, params=None, headers=None):
    sleep(random.uniform(0.1, 0.5)) # to avoid rate limiting
    if params:
        response = requests.get(url, params=params)
    elif headers:
        response = requests.get(url, headers=headers)
    else:
        response = requests.get(url)
    if response.status_code == 429:
        logger.error(f"Request failed with status code {response.status_code}")
        logger.error(response.text)
        retry_after = int(response.headers.get('Retry-After', -1))  # Defaults to -1 second if not provided
        logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
        exit()
        return None
    try:
        response_json = response.json()
        
    except json.JSONDecodeError:
        logger.error("Error decoding JSON response.")
        return None
    if response_json.get('error'):
        logger.error(response_json['error'])
        # if the error is due to invalid access token, renew the token and retry the request
        status = response_json['error']['status']
        if ERROR_CODES["Invalid access token"] == int(status):
            logger.warning("Renewing access token...")
            access_token = get_access_token()
            if params:
                params['access_token'] = access_token
                response = requests.get(url, params=params)
            elif headers:
                headers['Authorization'] = f"Bearer {access_token}"
                response = requests.get(url, headers=headers)
            response_json = response.json()
            # if the error persists, log the error message and return None
            if response_json.get('error'):
                status, message = response_json['error']['status'], response_json['error'].get(['message'])
                logger.error(message + " " + status + ", " + "Failed to renew token")
                return None
        else:
            return None
    return response_json