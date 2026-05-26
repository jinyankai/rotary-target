from rotary_target.algorithms import get_algorithm, list_algorithms


def test_algorithm_registry_contains_coursework_models():
    keys = {spec.key for spec in list_algorithms()}
    assert {"roi_trans", "gliding_vertex", "redet", "csl", "gwd", "kld"} <= keys


def test_get_algorithm_reports_upstream_config():
    spec = get_algorithm("roi_trans")
    assert spec.display_name == "RoI Transformer"
    assert spec.upstream_config.endswith(".py")
