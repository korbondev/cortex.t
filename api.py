import bittensor as bt
import asyncio
import json
import traceback

try:
    from template.protocol import StreamPrompting, TextPrompting, ImageResponse
except Exception as e:
    print(str(e))
    from cortext.protocol import StreamPrompting, TextPrompting, ImageResponse

# Assuming initial setup remains the same
wallet = bt.wallet(name="validator", hotkey="default")
axon = bt.axon(wallet=wallet)
dendrite = bt.dendrite(wallet=wallet)
subtensor = bt.subtensor(network="test")
metagraph = subtensor.metagraph(netuid=24)

# StreamPrompting variables
question = [{"role": "user", "content": "quick question"}]
vali_uid = 1
target_uid = 3
provider = "OpenAI"
model = "gpt-3.5-turbo"
seed = 1234
temperature = 0.5
max_tokens = 2048
top_p = 0.8
top_k = 1000
timeout = 3
streaming = True

synapse = StreamPrompting(
    messages=question,
    uid=target_uid,
    provider=provider,
    model=model,
    seed=seed,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    top_k=top_k,
    timeout=timeout,
    streaming=streaming,
)

# ImageResponse variables
messages = "a thick white cloud over a river"

synapse = ImageResponse(messages=messages)

print("messages", messages)
bt.trace()
response = dendrite.query(metagraph.axons[vali_uid], synapse, deserialize=False, timeout=synapse.timeout)
print("completion:", response.completion)

# async def query_miner(synapse):
#     try:
#         axon = metagraph.axons[vali_uid]
#         responses = dendrite.query(
#             axons=[axon],
#             synapse=synapse,
#             deserialize=False,
#             timeout=timeout,
#             streaming=streaming,
#         )
#         return await handle_response(responses)
#     except Exception as e:
#         print(f"Exception during query: {traceback.format_exc()}")
#         return None

# async def handle_response(responses):
#     full_response = ""
#     try:
#         for resp in responses:
#             async for chunk in resp:
#                 if isinstance(chunk, str):
#                     full_response += chunk
#                     bt.logging.info(chunk)
#     except Exception as e:
#         print(f"Error processing response for uid {e}")
#     return full_response

# async def main():
#     response = await query_miner(synapse)
#     bt.logging.info(f"full_response = {response}")

# if __name__ == "__main__":
#     asyncio.run(main())
