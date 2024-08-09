from typing import Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from openai import AzureOpenAI, OpenAI
import logging
from env_config import read_env_config, set_env
from os import environ
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI


logger = logging.getLogger("client_utils")

load_dotenv()  # take environment variables from .env.

def build_openai_client(**kwargs: Any) -> OpenAI:
    """
    Build OpenAI client based on the environment variables.
    """
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")  
    env = read_env_config("COMPLETION")
    with set_env(**env):
        if is_azure():
            client = AzureOpenAI(api_version="2024-02-15-preview", azure_endpoint="https://learn-shared-eastus.openai.azure.com/", azure_ad_token_provider=token_provider)
        else:
            client = OpenAI(**kwargs)
        return client

def build_langchain_embeddings(**kwargs: Any) -> OpenAIEmbeddings:
    """
    Build OpenAI embeddings client based on the environment variables.
    """

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    env = read_env_config("EMBEDDING")

    with set_env(**env):
        if is_azure():
            client = AzureOpenAIEmbeddings(azure_endpoint="https://learn-shared-eastus.openai.azure.com/", azure_ad_token_provider=token_provider)
        else:
            client = OpenAIEmbeddings(**kwargs)
        return client

def is_azure():
    # azure = "AZURE_OPENAI_ENDPOINT" in environ or "AZURE_OPENAI_KEY" in environ or "AZURE_OPENAI_AD_TOKEN" in environ
    # if azure:
    #     logger.info("Using Azure OpenAI environment variables")
    # else:
    #     logger.info("Using OpenAI environment variables")
    # return azure
    return True
