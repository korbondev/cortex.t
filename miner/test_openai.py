import asyncio
import os
import traceback
from openai import OpenAI
from openai import AsyncOpenAI
from msgspec import json

os.environ["OPENAI_API_KEY"] = "YOURAPIKEY"
# OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
# if not OpenAI.api_key:
#     raise ValueError("Please set the OPENAI_API_KEY environment variable.")

from alt_key_handler import get_endpoint_overrides

ENDPOINT_OVERRIDE_MAP = get_endpoint_overrides()

api_key = ENDPOINT_OVERRIDE_MAP["ENVIRONMENT_KEY"][ENDPOINT_OVERRIDE_MAP["ServiceEndpoint"].get("OpenRouter", {}).get("ENVIRONMENT_KEY", "")]
base_url = ENDPOINT_OVERRIDE_MAP["ServiceEndpoint"].get("OpenRouter", {}).get("api", "")
client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60)


api_key = ENDPOINT_OVERRIDE_MAP["ENVIRONMENT_KEY"][ENDPOINT_OVERRIDE_MAP["ServiceEndpoint"].get("AIMLAPI", {}).get("ENVIRONMENT_KEY", "")]
base_url = ENDPOINT_OVERRIDE_MAP["ServiceEndpoint"].get("AIMLAPI", {}).get("api", "")
image_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60)


async def send_openai_request(prompt, engine="gpt-4-1106-preview"):
    try:
        messages = [{"role": "user", "content": "Explore the impact of Thomas Edison's inventions on modern society, such as the light bulb and phonograph"}]
        model = "openai/gpt-4o"
        extra_body = {
            "transforms": [],
            "provider": {"allow_fallbacks": False},
            "allow_fallbacks": False,
        }
        stream = await client.chat.completions.create(
            extra_body=extra_body,
            messages=messages,
            stream=True,
            model=model,
            seed=1234,
            temperature=0.0001,
            max_tokens=4096,
            # top_p=0.01,
            # top_k=1,
        )
        collected_messages = []

        async for part in stream:
            print(part.choices[0].delta.content or "")
            collected_messages.append(part.choices[0].delta.content or "")

        all_messages = "".join(collected_messages)
        return all_messages

    except Exception as e:
        print(f"Got exception when calling openai {e}")
        traceback.print_exc()
        return "Error calling model"


async def send_openai_image_request(
    prompt="a jeep ride through aruba on a majestical dirt sunny road in the forest",
    model="unknown",
):
    try:
        response = await image_client.images.generate(
            model=model,
            prompt=prompt,
            size="1792x1024",
            quality="standard",  # or hd (double the cost)
            style="vivid",  # or naturual
            n=1,
        )

        image_url = response.data[0].url
        image_revised_prompt = response.data[0].revised_prompt
        image_created = response.created
        return f"created at {image_created}\n\nrevised_prompt = {image_revised_prompt}\n\nurl = {image_url}"

    except Exception as e:
        print(f"Got exception when calling openai {e}")
        traceback.print_exc()
        return "Error calling model"


async def llm_test():
    prompts = ["count to 10", "tell me a joke"]
    tasks = [send_openai_request(prompt) for prompt in prompts]
    # tasks = [send_openai_request("")]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)


async def dalle_test():
    # prompts = ["count to 10", "tell me a joke"]
    # tasks = [send_openai_request(prompt) for prompt in prompts]
    model = "dall-e-3"
    tasks = [send_openai_image_request(model=ENDPOINT_OVERRIDE_MAP["LlmModelMap"].get(model, {}).get("ModelName", ""))]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)


# asyncio.run(llm_test())


asyncio.run(dalle_test())
