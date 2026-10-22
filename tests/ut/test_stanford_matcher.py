from types import SimpleNamespace

import pytest

from kernel_eval.benches.stanford_matcher import StanfordMatcher


class FakeStanfordLoader:
    def __init__(self, tasks):
        self.tasks = tasks

    def get_operator_by_name(self, name):
        return next((task for task in self.tasks if task.name.lower() == name.lower()), None)


@pytest.mark.parametrize(
    ("task_name", "snake_name"),
    [("LayerNorm", "layer_norm"), ("Softmax", "softmax")],
)
def test_find_operator_info_by_snake_converts_names(task_name, snake_name):
    task = SimpleNamespace(name=task_name)
    matcher = StanfordMatcher(operator_loader=FakeStanfordLoader([task]))

    assert matcher.find_operator_info_by_snake(snake_name) is task
