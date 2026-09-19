from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Planeja a execução final do pipeline.")
    parser.add_argument("--config", type=Path, default=Path("microsservices_final/manifests/final_run_config.json"))
    parser.add_argument("--execute", action="store_true", help="Reservado para a execução final após revisão do plano.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_logs = Path(config["input"]["raw_logs"])
    if not raw_logs.exists():
        raise FileNotFoundError(f"Diretório de logs não encontrado: {raw_logs}")
    if not args.execute:
        print(json.dumps({"status": "dry_run", "pipeline": config["pipeline"],
                          "raw_logs": str(raw_logs), "message": "Nenhum dado foi processado."},
                         indent=2, ensure_ascii=False))
        return
    raise SystemExit("Execução final ainda requer revisão/aprovação do manifesto e dos comandos.")


if __name__ == "__main__":
    main()
