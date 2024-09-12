#!/usr/bin/env python3

import json
import os
from datetime import datetime

import cv2
import ndjson
import numpy as np
import requests
import yaml
from tqdm import tqdm

"""
https://medium.com/@stefan.herdy/how-to-export-labelbox-annotations-eedb8cb4f217
"""


def logits2rgb(img, colour_mapping):
    shape = np.shape(img)
    h = int(shape[0])
    w = int(shape[1])
    col = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, (background_color, contour_color) in colour_mapping.items():
        mask = np.where(img == idx)

        # Set background color
        col[mask] = background_color

        # Find contours
        contours, _ = cv2.findContours(
            (img == idx).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contours with contour color
        cv2.drawContours(col, contours, -1, contour_color, thickness=1)

    return col


def get_mask(
    project_id,
    api_key,
    save_dir,
    class_indices,
    colour_mapping,
    overwrite=True,
    start_date=None,
    end_date=None,
):
    # Open export json. Change name if required
    with open(f"{save_dir}/export-result.ndjson") as f:
        data = ndjson.load(f)
        # Iterate over all images
        for i, d in enumerate(tqdm(data)):
            # Check if the stitched mask file already exists
            stitched_mask_path = f"{save_dir}/masks_stitched/{os.path.splitext(os.path.basename(data[i]['data_row']['external_id']))[0]}_mask.png"
            if not overwrite and os.path.exists(stitched_mask_path):
                # print(f"Stitched mask already exists for {data[i]['data_row']['external_id']}. Skipping...")
                continue  # Skip to the next image

            # Extract creation date
            creation_date = datetime.strptime(
                data[i]["data_row"]["details"]["created_at"],
                "%Y-%m-%dT%H:%M:%S.%f+00:00",
            )

            # Check if the creation date falls within the specified range
            if start_date and creation_date.date() < start_date.date():
                # print(f"Creation date {creation_date} is before start date {start_date}. Skipping...")
                continue
            if end_date and creation_date.date() > end_date.date():
                # print(f"Creation date {creation_date} is after end date {end_date}. Skipping...")
                continue

            # Download image
            img_url = data[i]["data_row"]["row_data"]
            with requests.get(img_url, stream=True) as r:
                r.raw.decode_content = True
                mask = r.raw
                image = np.asarray(bytearray(mask.read()), dtype="uint8")
                image = cv2.imdecode(image, -1)
            # Save Image
            cv2.imwrite(
                f"{save_dir}/images/{data[i]['data_row']['external_id']}", image
            )

            mask_full = np.zeros((
                data[i]["media_attributes"]["height"],
                data[i]["media_attributes"]["width"],
            ))
            # Iterate over all masks
            for idx, obj in enumerate(
                data[i]["projects"][project_id]["labels"][0]["annotations"]["objects"]
            ):
                # Extract mask name and mask url
                name = data[i]["projects"][project_id]["labels"][0]["annotations"][
                    "objects"
                ][idx]["name"]
                url = data[i]["projects"][project_id]["labels"][0]["annotations"][
                    "objects"
                ][idx]["mask"]["url"]

                cl = class_indices[name]

                # Download mask
                headers = {"Authorization": api_key}
                with requests.get(url, headers=headers, stream=True) as r:
                    r.raw.decode_content = True
                    mask = r.raw
                    image = np.asarray(bytearray(mask.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                # Assign mask index to image-mask
                mask = np.where(image == 255)
                mask_full[mask] = cl

                mask_single = np.zeros((
                    data[i]["media_attributes"]["height"],
                    data[i]["media_attributes"]["width"],
                ))
                mask_single[mask] = cl
                mask_single_colour = logits2rgb(mask_single, colour_mapping)
                mask_single_colour = cv2.cvtColor(
                    mask_single_colour.astype("float32"), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(
                    f"{save_dir}/masks/{os.path.splitext(os.path.basename(data[i]['data_row']['external_id']))[0]}_mask_{idx}.png",
                    mask_single_colour,
                )

            mask_full_colour = logits2rgb(mask_full, colour_mapping)
            mask_full_colour = cv2.cvtColor(
                mask_full_colour.astype("float32"), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(
                f"{save_dir}/masks_stitched/{os.path.splitext(os.path.basename(data[i]['data_row']['external_id']))[0]}_mask.png",
                mask_full_colour,
            )


if __name__ == "__main__":
    with open("./key_json", "r") as file:
        config = yaml.safe_load(file)
    LB_API_KEY = config["labelbox"]["api_key"]

    class_indices = {"Hard Coral": 1, "Soft Coral": 2}

    colour_mapping = {
        0: [[0, 0, 0], [0, 0, 0]],
        1: [[255, 0, 0], [255, 255, 0]],
        2: [[0, 0, 255], [255, 165, 0]],
    }

    BUCKET_NAME = "rs_storage_open"
    BUCKET_SAVE_DIR = "benthic_datasets/mask_labels/reef_support"

    with open("../labelbox_projects.json") as f:
        labelbox_projects = json.load(f)

    for project_name, project_id in labelbox_projects.items():
        # start_date = datetime(2023, 7, 25)
        start_date = None
        # end_date = datetime(2023, 8, 1)
        end_date = None

        SAVE_DIR = f"./reef_support/{project_name}"
        print(f"Exporting {project_name} to {SAVE_DIR}")
        if not os.path.exists(SAVE_DIR):
            print("Directory does not exist")
            continue
        if not os.path.exists(f"{SAVE_DIR}/export-result.ndjson"):
            print("Export file does not exist")
            continue
        if not os.path.exists(f"{SAVE_DIR}/images"):
            os.makedirs(f"{SAVE_DIR}/images")
        if not os.path.exists(f"{SAVE_DIR}/masks"):
            os.makedirs(f"{SAVE_DIR}/masks")
        if not os.path.exists(f"{SAVE_DIR}/masks_stitched"):
            os.makedirs(f"{SAVE_DIR}/masks_stitched")
        get_mask(
            project_id,
            LB_API_KEY,
            SAVE_DIR,
            class_indices,
            colour_mapping,
            overwrite=False,
            start_date=start_date,
            end_date=end_date,
        )

        print("Done.")
        print("______________________________________________________")
