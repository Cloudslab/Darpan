from __future__ import annotations

import pytest

from darpan import ComponentSpec, RetryPolicy
from darpan.core.codec import application_spec_from_dict


def test_retry_policy_decodes_from_application_yaml_mapping() -> None:
    app = application_spec_from_dict(
        {
            "id": "retry-app",
            "components": {
                "task": {
                    "retry": {
                        "max_retries": 2,
                        "on": ["execution_failed", "node_offline"],
                    }
                }
            },
        }
    )
    retry = app.component("task").retry
    assert retry.max_retries == 2
    assert retry.on == frozenset({"execution_failed", "node_offline"})
    assert retry.allows("node_offline")
    assert not retry.allows("execution_error")


def test_retry_policy_accepts_integer_shorthand() -> None:
    app = application_spec_from_dict(
        {"id": "retry-app", "components": {"task": {"retry": 3}}}
    )
    assert app.component("task").retry == RetryPolicy(max_retries=3)


def test_long_running_component_rejects_automatic_retry_policy() -> None:
    with pytest.raises(ValueError, match="finite components"):
        ComponentSpec("service", kind="service", retry=RetryPolicy(max_retries=1))
