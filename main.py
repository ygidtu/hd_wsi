#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""
包一层外包脚本
"""
import os
import sys

from glob import glob

import click

from run_patch_inference import patch
from run_wsi_inference import wsi
from summarize_tme_features import summarize


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    u"""
    Welcome
    :return:
    """
    pass


cli.add_command(wsi)
cli.add_command(summarize)
cli.add_command(patch)


if __name__ == "__main__":
    cli()

