"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 21st 2018

Download binary files shared on google drive
"""

import io
import sys
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from httplib2 import Http
    from oauth2client import file, client, tools
except ImportError as ex:
    print("To download shared google drive file via python,"
          "google-api-python-client, oauth2client is required."
          "Please use pip install google-api-python-client oauth2client.")
    raise ex

SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
CREDENTIALS = './Data/credentials.json'


def require_authorize(store, credentials, scope):
    sys.argv.clear()
    flow = client.flow_from_clientsecrets(credentials, scope)
    return tools.run_flow(flow, store)


def drive_download(name, fileid, path):
    store_path = Path(path) / name
    if store_path.exists():
        print("{} exists, skip download.".format(name))
        return store_path
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    store = file.Storage('/tmp/token.json')
    creds = store.get()
    if not creds or creds.invalid:
        creds = require_authorize(store, CREDENTIALS, SCOPES)
    service = build('drive', 'v3', http=creds.authorize(Http()))

    request = service.files().get_media(fileId=fileid)

    fh = io.FileIO(store_path.resolve(), 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print("\rDownload {}%%.".format(int(status.progress() * 100)))
    print('\n', flush=True)
    if done:
        return store_path
