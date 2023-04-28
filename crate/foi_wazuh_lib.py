"""
This file contains the abstract methods for querying Wazuh and sending actions
to Wazuh.
This would ideally be implemented by FOI. 
"""
from abc import abstractmethod
from typing import Dict
import asyncio


@abstractmethod
async def query_wazuh(wazuh_hostname: str) -> Dict:
    """Query Wazuh for a list of alerts
    returns:
                Dict: A dictionary containing the current alerts
    """

    print("Querying Wazuh...")
    response = await asyncio.sleep(1, "success")
    print("Response from Wazuh: ", response)
    return {"alerts": [], "info": {"message": response}}


@abstractmethod
async def send_action_to_wazuh(payload: str) -> Dict:
    """Send an action to Wazuh
    Args:
            payload (str): A JSON string containing the action to send to Wazuh
    Returns:
            Dict: A dictionary containing the response from Wazuh.
    """
    print("Sending action to Wazuh...")
    response  = await asyncio.sleep(1, "success")
    print("Response from Wazuh: ", response)
    return {"response": response}  
