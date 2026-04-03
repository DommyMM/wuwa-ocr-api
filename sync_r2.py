"""
sync_r2.py — download all images from R2 bucket into local r2-backup/.

Only fetches files not already present locally (skips existing).
Run this before benchmark to ensure r2-backup is up to date.

Usage:
  py sync_r2.py          # dry run — shows what would be downloaded
  py sync_r2.py --run    # actually downloads
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
R2_BACKUP   = BACKEND_DIR.parent / "r2-backup"
ENV_FILE    = BACKEND_DIR.parent / "wuwabuilds" / ".env"
DRY_RUN     = "--run" not in sys.argv


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main():
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    R2_BACKUP.mkdir(exist_ok=True)
    env        = load_env(ENV_FILE)
    account_id = env["CLOUDFLARE_ACCOUNT_ID"]
    bucket     = env.get("R2_BUCKET_NAME", "wuwabuilds")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )

    # List all objects in bucket (paginated)
    print(f"Listing objects in R2 bucket '{bucket}' ...")
    paginator = s3.get_paginator("list_objects_v2")
    all_keys  = []
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            all_keys.append(obj["Key"])

    print(f"Found {len(all_keys)} objects in R2")

    local_files = {p.name for p in R2_BACKUP.iterdir() if p.is_file()}
    to_download = [k for k in all_keys if k not in local_files]

    print(f"Local r2-backup: {len(local_files)} files")
    print(f"To download:     {len(to_download)} new files")

    if not to_download:
        print("Already in sync.")
        return

    if DRY_RUN:
        print(f"\n[DRY RUN] Would download {len(to_download)} files. Re-run with --run to apply.")
        for k in to_download[:20]:
            print(f"  {k}")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download) - 20} more")
        return

    WORKERS = 32
    print(f"\nDownloading with {WORKERS} parallel workers ...\n")

    downloaded = 0
    failed     = 0
    total      = len(to_download)

    def fetch(idx_key):
        idx, key = idx_key
        try:
            s3.download_file(bucket, key, str(R2_BACKUP / key))
            return idx, key, None
        except Exception as e:
            return idx, key, str(e)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch, (i, k)): k for i, k in enumerate(to_download, 1)}
        for fut in as_completed(futures):
            idx, key, err = fut.result()
            if err:
                print(f"  [{idx}/{total}] FAILED {key}: {err}")
                failed += 1
            else:
                print(f"  [{idx}/{total}] {key}")
                downloaded += 1

    print(f"\n{'─'*50}")
    print(f"  Downloaded: {downloaded}  Failed: {failed}")
    print(f"  r2-backup now has {len(list(R2_BACKUP.iterdir()))} files")


if __name__ == "__main__":
    main()
