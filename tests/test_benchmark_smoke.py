from benchmarks.benchmark_aps import main


def test_benchmark_script_smoke(tmp_path):
    output = tmp_path / "benchmark.json"
    assert main(["--repeats", "1", "--n-estimators", "10", "--output", str(output)]) == 0
    assert output.exists()
