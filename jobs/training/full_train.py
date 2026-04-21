import subprocess
import sys


MODULES = [
    "jobs.training.build_dataset",
    "jobs.training.train_model",
    "jobs.training.backtest",
    "jobs.training.analyze",
]


def run_step(module_name: str) -> None:
    print(f"\n=== Running {module_name} ===")
    result = subprocess.run([sys.executable, "-m", module_name])

    if result.returncode != 0:
        raise SystemExit(f"Erro ao executar {module_name} (code={result.returncode})")


def main() -> None:
    for module_name in MODULES:
        run_step(module_name)

    print("\nPipeline finalizada com sucesso.")


if __name__ == "__main__":
    main()