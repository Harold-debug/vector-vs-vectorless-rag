"""
AWS Bedrock client factories
"""

from __future__ import annotations

import os

import boto3
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrock

load_dotenv()

DEFAULT_REGION = os.getenv("AWS_REGION", "us-west-2")
DEFAULT_CHAT_MODEL = os.getenv("BEDROCK_CHAT_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
DEFAULT_EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")


def get_embeddings(
    model_id: str = DEFAULT_EMBEDDING_MODEL,
    region: str = DEFAULT_REGION,
) -> BedrockEmbeddings:
    client = boto3.client("bedrock-runtime", region_name=region)
    return BedrockEmbeddings(client=client, model_id=model_id, region_name=region)


def get_chat_model(
    model_id: str = DEFAULT_CHAT_MODEL,
    region: str = DEFAULT_REGION,
) -> ChatBedrock:
    return ChatBedrock(model_id=model_id, region_name=region)
