from collections.abc import Iterator
from pathlib import Path

import pytest

from cmbench.data.seed import generate_database


@pytest.fixture
def seeded_db(tmp_path: Path) -> Iterator[Path]:
    db_path = tmp_path / "crm.db"
    generate_database(db_path, seed=42)
    yield db_path
