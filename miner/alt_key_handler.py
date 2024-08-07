from os import environ, path
from random import choice as random_choice
from typing import Callable, List
from asyncio import Lock as async_Lock
from msgspec import yaml
from sys import argv

from openai import AsyncOpenAI


def check_endpoint_overrides() -> bool:
    """Check if the endpoint_overrides.yaml file exists in the current directory or script directory.

    Returns:
        bool: True if the file exists in either location, False otherwise."""

    file_path = path.join(path.dirname(__file__), "endpoint_overrides.yaml")
    script_dir_file_path = path.join(path.dirname(path.abspath(argv[0])), "endpoint_overrides.yaml")
    return path.exists(file_path) or path.exists(script_dir_file_path)


def get_endpoint_overrides() -> dict:
    """
    Retrieves the endpoint overrides from the `endpoint_overrides.yaml` file.

    Returns:
        dict: A dictionary containing the endpoint overrides.

    Raises:
        FileNotFoundError: If the `endpoint_overrides.yaml` file is not found.
        Exception: If an error occurs while reading the file.
    """

    if not check_endpoint_overrides():
        return {}
    try:
        with open(path.join(path.dirname(__file__), "endpoint_overrides.yaml"), "r") as f:
            return yaml.decode(f.read())
    except FileNotFoundError:
        with open(path.join(path.dirname(path.abspath(argv[0])), "endpoint_overrides.yaml"), "r") as f:
            return yaml.decode(f.read())
    except Exception:
        raise


def override_endpoint_keys() -> None:
    """
    Updates the environment variables with the endpoint overrides.

    This function retrieves the endpoint overrides from the `get_endpoint_overrides()` function
    and updates the environment variables using the `update()` method of the `environ` dictionary.
    The endpoint overrides are obtained from the "OVERRIDE_ENVIRONMENT_KEYS" key in the returned
    dictionary.

    Parameters:
        None

    Returns:
        None
    """

    environ.update(get_endpoint_overrides().get("ENVIRONMENT_KEY", {}))


def provider_client_lfu_closure(provider_client_keys: List[str], base_url: str = "https://api.openai.com/v1", timeout: float = 60.0) -> Callable[[], AsyncOpenAI]:
    """
    Returns a closure that creates an LFU (Least Frequently Used) client object with random tie breaking.

    The function creates a list of lists, containing a unique AsyncOpenAI client object and its usage frequency.
    The usage frequency is initially set to 0 for all clients.

    The closure function `provider_client_lfu_with_random_tie_breaking` calculates the least frequently used client
    by finding the minimum usage frequency among all the clients. It selects the clients with the minimum frequency
    and randomly selects one of them. It increments the usage frequency of the selected client by 1.

    Returns:
        Callable[[], AsyncOpenAI]: A closure function that returns the least frequently used AsyncOpenAI client object.
    """
    lock = async_Lock()
    provider_multi_clients = [
        [
            AsyncOpenAI(
                api_key=ModelKey,
                base_url=base_url,
                timeout=timeout,
            ),
            0,
        ]
        for ModelKey in provider_client_keys
    ]

    async def provider_client_lfu_with_random_tie_breaking() -> AsyncOpenAI:
        async with lock:
            nonlocal provider_multi_clients

            min_frequency = min(item[1] for item in provider_multi_clients)

            least_used_clients = [x for x in provider_multi_clients if x[1] == min_frequency]

            image_client = random_choice(least_used_clients)

            image_client[1] += 1

            return image_client[0]

    return provider_client_lfu_with_random_tie_breaking


def results_padding(result_length: int, padding_pct: float) -> str:
    pad_by = round((padding_pct / 100) * result_length, ndigits=None)
    return " .".join(["" for _ in range(pad_by + 1)]) + " "
