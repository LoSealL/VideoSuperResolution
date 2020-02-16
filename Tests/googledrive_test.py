#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 16

import os
import unittest

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')

try:
  from googleapiclient.discovery import build
  from httplib2 import Http
  from oauth2client import file, client, tools
except ImportError as ex:
  print(f"[!] {ex}")
  exit(-1)

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive.readonly'


class FetchGoogleDriveTest(unittest.TestCase):
  def test_downloads(self):
    """Shows basic usage of the Drive v3 API.
       Prints the names and ids of the first 10 files the user has access to.
    """
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    store = file.Storage('/tmp/token.json')
    creds = store.get()
    if not creds or creds.invalid:
      flow = client.flow_from_clientsecrets('../Data/credentials.json', SCOPES)
      creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))
    file_id = '1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2'
    request = service.files().get_media(fileId=file_id)
    request.execute()


if __name__ == '__main__':
  unittest.main()
