#!/usr/bin/env python3
"""
remove_playlist_from_faiss.py

Deletes all chunks from FAISSDatabase whose doc_url contains a given YouTube playlist id.

Usage:
  python remove_playlist_from_faiss.py --index-dir ./rag_index --target-url "<youtube url>" [--dry-run] [--backup]
  OR
  python remove_playlist_from_faiss.py --index-dir ./rag_index --playlist-id PLxxxxxxx

Behavior:
- Dry-run prints what would be removed.
- Without --dry-run the script removes matching documents' chunks and saves the index back to index-dir.
- --backup will copy the index files to index-dir.bak_TIMESTAMP before modification.
"""

import argparse
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# adjust import path if needed so we can import FAISSDatabase
# this assumes database.py is in the same directory or package
from database import FAISSDatabase


from dotenv import load_dotenv
import os

load_dotenv()

def extract_playlist_id_from_url(url: str) -> str | None:
    try:
        qs = parse_qs(urlparse(url).query)
        return qs.get("list", [None])[0]
    except Exception:
        return None


def backup_index(index_dir: Path) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%S")
    target = index_dir.parent / f"{index_dir.name}.bak_{ts}"
    shutil.copytree(index_dir, target)
    return target


def main():
    p = argparse.ArgumentParser(description="Remove FAISS chunks for a YouTube playlist")
    p.add_argument("--index-dir", required=True, help="Directory containing faiss.index and metadata.pkl")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-url", help="Any YouTube URL containing the playlist id (will extract `list` parameter)")
    group.add_argument("--playlist-id", help="Playlist id directly, e.g. PLvb7SEPUrI6uZbq-...")
    p.add_argument("--dry-run", action="store_true", help="List what would be removed but do not change the index")
    p.add_argument("--backup", action="store_true", help="Backup index directory before modifying (recommended)")
    p.add_argument("--save", action="store_true", help="Save the index after removal (default if not dry-run)")
    args = p.parse_args()

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise SystemExit(f"Index directory not found: {index_dir!s}")

    # determine playlist id
    if args.playlist_id:
        playlist_id = args.playlist_id
    else:
        playlist_id = extract_playlist_id_from_url(args.target_url)
    if not playlist_id:
        raise SystemExit("Could not extract playlist id from target URL. Provide --playlist-id or a valid --target-url containing 'list=' param.")

    print(f"[INFO] Target playlist id: {playlist_id}")

    # backup if requested
    if args.backup and not args.dry_run:
        bak = backup_index(index_dir)
        print(f"[INFO] Backed up index directory to: {bak}")

    # instantiate DB (it will load saved index if present)
    # NOTE: provide a dummy API key because FAISSDatabase requires one in __init__
    # but we won't call embedding functions here. Use environment or pass a placeholder if allowed.
    # If your FAISSDatabase insists on a real key, set GEMINI_API_KEY in env or pass gemini_api_key param.
    import os
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    db = FAISSDatabase(index_path=str(index_dir), gemini_api_key=gemini_key)

    pre_stats = db.stats()
    print(f"[INFO] Index stats before: total_chunks={pre_stats['total_chunks']}, total_documents={pre_stats['total_documents']}")

    # collect doc_index values whose doc_url contains the playlist id
    to_remove_doc_indices = set()
    to_remove_int_ids = set()
    for int_id, chunk in db._meta.items():
        url = getattr(chunk, "doc_url", None)
        if not url:
            continue
        pid = extract_playlist_id_from_url(url)
        if pid and pid == playlist_id:
            to_remove_doc_indices.add(chunk.doc_index)
            to_remove_int_ids.add(int(int_id))

    if not to_remove_doc_indices:
        print("[INFO] No documents found matching that playlist id. Nothing to do.")
        return

    print(f"[INFO] Found {len(to_remove_doc_indices)} document(s) referencing playlist {playlist_id}")
    # show sample titles (dry-run useful)
    sample = []
    for iid in sorted(to_remove_int_ids)[:10]:
        c = db._meta.get(iid)
        if c:
            sample.append((iid, getattr(c, "doc_title", "<no title>"), getattr(c, "doc_url", "<no url>")))
    print("[INFO] Sample items (int_id, title, url):")
    for s in sample:
        print("   ", s)

    if args.dry_run:
        print(f"[DRY-RUN] Would remove {len(to_remove_int_ids)} chunk(s) from {len(to_remove_doc_indices)} doc(s).")
        return

    # Remove by doc_index using provided helper (deletes all chunks of that document)
    total_removed = 0
    for doc_idx in sorted(to_remove_doc_indices):
        removed = db.delete_by_doc_index(doc_idx)
        print(f"[ACTION] Removed {removed} chunk(s) for doc_index={doc_idx}")
        total_removed += removed

    print(f"[INFO] Total chunks removed: {total_removed}")

    # Save index back to disk if requested (default save True if not dry-run)
    if args.save or not args.dry_run:
        db.save(str(index_dir))
        post_stats = db.stats()
        print(f"[INFO] Index saved. After: total_chunks={post_stats['total_chunks']}, total_documents={post_stats['total_documents']}")

    print("[DONE]")


if __name__ == "__main__":
    main()