"""Test that router_eval_gold.py supports --llm-suffix to use alternative Layer 4 LLM."""
import subprocess


def test_router_eval_gold_supports_llm_suffix_arg():
    result = subprocess.run(
        ["python", "-m", "src.eval.router_eval_gold", "--help"],
        capture_output=True, text=True,
    )
    assert "--llm-suffix" in result.stdout, (
        "Expected --llm-suffix in CLI help:\n" + result.stdout
    )
