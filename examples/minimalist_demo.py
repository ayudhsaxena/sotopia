# This demo servers as a minimal example of how to use the sotopia library.

# 1. Import the sotopia library
# 1.1. Import the `run_async_server` function: In sotopia, we use Python Async
#     API to optimize the throughput.
import asyncio
import logging

# 1.2. Import the `UniformSampler` class: In sotopia, we use samplers to sample
#     the social tasks.
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from rich.logging import RichHandler

import os
os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"
# 2. Run the server
# 2.1. Configure the logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

# 2.2. Run the simulation
asyncio.run(
    run_async_server(
        model_dict={
            "env": "custom/env_model@http://localhost:8020/v1",
            "agent1": "custom/qwen_base_model@http://localhost:8000/v1",
            "agent2": "custom/opp_model@http://localhost:8010/v1",
        },
        sampler=UniformSampler(),
    )
)