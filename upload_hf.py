# -*- coding: utf-8 -*-
"""Upload this English mirror to Hugging Face Hub.

Usage:
  1. Get a token at https://huggingface.co/settings/tokens (write scope).
  2. set HF_TOKEN=hf_xxx
  3. python upload_hf.py <your-hf-username> [repo-id-suffix]

Creates <username>/heart-protocol-en as a public model repo and uploads
the whole folder. Re-runs are incremental.
"""
import sys
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    user = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else "heart-protocol-en"
    repo_id = f"{user}/{suffix}"
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN first:  set HF_TOKEN=hf_xxx")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="model",
        commit_message="English reference edition of the 16-Sephirot Heart Protocol",
    )
    print(f"Done -> https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
