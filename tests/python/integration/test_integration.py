# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario

PROJECT_KEY = "TESTONTOLOGYTAGGER"

"""Each scenario tests the three output formats"""


def add_test_scenario(user_dss_clients, scenario_id):
    """run any scenario"""
    dss_scenario.run(user_dss_clients, project_key=PROJECT_KEY, scenario_id=scenario_id)


def test_monolingual_category(user_dss_clients):
    """Plugin parameters: -language: en
    - category_column specified"""
    add_test_scenario(user_dss_clients, "MONOLINGUALCATEGORY")


def test_monolingual_no_category(user_dss_clients):
    """Plugin parameters:
    -language: en
    - category_column not specified"""
    add_test_scenario(user_dss_clients, "MONOLINGUALNOCATEGORY")


def test_multilingual_category(user_dss_clients):
    """Plugin parameters:
    -language: language column with the 58 supported languages in it
    -category_column specified"""
    add_test_scenario(user_dss_clients, "MULTILINGUALCATEGORY")


def test_multilingual_no_category(user_dss_clients):
    """Plugin parameters:
    -language: language column with the 58 supported languages in it
    -category_column not specified"""
    add_test_scenario(user_dss_clients, "MULTILINGUALNOCATEGORY")


def test_case_sensitivity(user_dss_clients):
    """Plugin parameters : -case-insensitivity activated"""
    add_test_scenario(user_dss_clients, "CASESENSITIVITY")
