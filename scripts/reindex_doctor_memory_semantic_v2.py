from __future__ import annotations

import argparse
import json

from backend.app.services.doctor_memory_service import DoctorMemoryService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reindex Doctor Memory vectors to semantic v2 embedding text."
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    summary = DoctorMemoryService().reindex_semantic_v2(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        max_batches=args.max_batches,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
