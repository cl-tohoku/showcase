# -*- coding: utf-8 -*-
from click.testing import CliRunner

from showcase.test import cli


def test_parser():
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0
