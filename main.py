import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from ontology_dc8f06af066e4a7880a5938933236037.train_data import chatbot_response
import logging

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
logger = logging.getLogger(__name__)

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        if text.lower()=="end":
            break
        if text.lower()=="" or text.lower()=="*":
            response = "Please re-phrase your query!"
        else:
            response = chatbot_response(text.lower())
        output.append(response)

    return SimpleText(dict(text=output))
