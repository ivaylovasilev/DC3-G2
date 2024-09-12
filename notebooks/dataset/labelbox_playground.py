import labelbox as lb
import json
import os

os.getcwd()

d = None

with open("challenges/coralreefs2/src/key.json", "r") as f:
    d = json.load(f)

LABELBOX_API_KEY = d["labelbox"]["api_key"]


project_id = "clld3lxl50etx07yvb3pydzh5"  # Taken from the ndjson
client = lb.Client(LABELBOX_API_KEY)

project = client.get_project(project_id)
# Looking at the ndjson, it seems that we only have the segmentation masks available
