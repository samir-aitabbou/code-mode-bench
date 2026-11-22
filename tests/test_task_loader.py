from cmbench.tasks.loader import load_tasks


def test_loads_all_6_tasks() -> None:
    tasks = load_tasks()
    assert set(tasks.keys()) == {"T1", "T2", "T3", "T4", "T5", "T6"}


def test_prompt_includes_statement_and_format() -> None:
    tasks = load_tasks()
    prompt = tasks["T1"].build_prompt()
    assert "C0042" in prompt
    assert "FINAL_ANSWER" in prompt
