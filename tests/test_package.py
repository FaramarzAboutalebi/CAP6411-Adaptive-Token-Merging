"""Lightweight tests for the initial package scaffold."""

import unittest

import cap6411_atm
import cap6411_atm.benchmark
import cap6411_atm.data
import cap6411_atm.evaluation
import cap6411_atm.merging
import cap6411_atm.models


class PackageTests(unittest.TestCase):
    def test_project_root_contains_readme(self) -> None:
        self.assertTrue((cap6411_atm.PROJECT_ROOT / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
