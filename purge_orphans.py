import argparse
from src.io_utils import delete_orphaned_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    delete_orphaned_runs(dry_run=args.dry_run)