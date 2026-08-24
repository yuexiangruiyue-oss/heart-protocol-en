# -*- coding: utf-8 -*-
"""Upload this English mirror to ModelScope (魔搭社区).

Usage:
  1. Get a token at https://modelscope.cn/my/myaccesstoken
  2. set MODELSCOPE_API_TOKEN=xxxx
  3. python upload_ms.py <your-ms-username> [repo-name]

Creates <username>/<repo-name> as a public model repo and uploads the folder.
"""
import sys
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    user = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "heart-protocol-en"
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        sys.exit("Set MODELSCOPE_API_TOKEN first:  set MODELSCOPE_API_TOKEN=xxx")

    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(token)
    repo_id = f"{user}/{name}"
    try:
        api.create_model(repo_id=repo_id, visibility=1)  # 1 = public
    except Exception as e:
        print(f"create_model: {e} (may already exist — continuing)")

    # ModelScope SDK uploads via a local git clone; use its push API
    from modelscope.hub.snapshot_download import snapshot_model_repo  # noqa
    api.push_model(repo_id=repo_id,
                   model_dir=str(ROOT),
                   commit_message="English reference edition of the 16-Sephirot Heart Protocol")
    print(f"Done -> https://modelscope.cn/models/{repo_id}")

if __name__ == "__main__":
    main()
