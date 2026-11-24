from cmbench.tasks.loader import load_tasks


def test_loads_all_6_tasks() -> None:
    tasks = load_tasks()
    # T4p: same task as T4, served by a paginated list_invoices tool
    # (ANALYSE-CLAUDE.md point A complementary probe).
    assert set(tasks.keys()) == {"T1", "T2", "T3", "T4", "T4p", "T5", "T6"}


def test_prompt_includes_statement_and_format() -> None:
    tasks = load_tasks()
    prompt = tasks["T1"].build_prompt()
    assert "C0042" in prompt
    assert "FINAL_ANSWER" in prompt
