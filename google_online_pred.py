#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:34:29 2024

@author: gaffliu
"""

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_tabular_classification_sample]
from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


															
															
def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

predict_tabular_classification_sample(
    project="528297461974",
    endpoint_id="5647489743466790912",
    location="us-central1",
    instance_dict = [
                    {
                        "id": "0",
                        "cli": "0.7102743",
                        "ili": "0.7240316",
                        "hh_cmnty_cli": "14.5616417",
                        "nohh_cmnty_cli": "11.1175833",
                        "wearing_mask": "52.9058154",
                        "travel_outside_state": "17.842008",
                        "work_outside_home": "31.7333285",
                        "shop": "66.9642023",
                        "restaurant": "31.8637288",
                        "spent_time": "41.5639747",
                        "large_event": "18.7579293",
                        "public_transit": "4.1010579",
                        "anxious": "11.7455548",
                        "depressed": "9.2002811",
                        "worried_finances": "31.2324546"
                    }
                ]
)
# [END aiplatform_predict_tabular_classification_sample]