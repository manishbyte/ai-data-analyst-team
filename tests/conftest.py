
import asyncio
import sys

# Configure the event-loop policy once, rather than overriding
# pytest-asyncio's deprecated event_loop_policy fixture.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_service():
    return SimpleNamespace(
        analyze=AsyncMock(),
        engine=object(),
    )