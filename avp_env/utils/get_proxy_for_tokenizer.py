#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


def get_proxy():
    config_path = '../config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        proxies = config.get('proxies', None)
    except Exception as e:
        print(f"Reading proxies error: {e}")
        proxies = None
    print(f"Proxies setting: {proxies}")
    return proxies
