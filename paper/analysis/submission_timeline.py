"""Submission timeline for Figure (participation): one row per published leaderboard entry with its
submission time and public-split Dice. Reads the Firestore backup inside ~/Desktop/rexbackup_2026-10-05.zip
and writes paper/figures/data/submission_timeline.csv (no e-mail addresses). Run from the project root."""
import zipfile, json, csv, os
Z = os.path.expanduser("~/Desktop/rexbackup_2026-10-05.zip")
NAME = "rexbackup_2026-10-05/01_leaderboard_admin/auto/backups/firestore_pre_postchallenge_20260928T115008Z.json"
d = json.loads(zipfile.ZipFile(Z).read(NAME))
lb, sub = d["leaderboard"], d["submissions"]
rows = []
for sid, e in lb.items():
    if e.get("status") != "published": continue
    s = sub.get(sid, {})
    rows.append({"submission_id": sid, "created_at": s.get("createdAt") or e.get("updatedAt"), "team": e.get("registeredTeam") or e.get("team"),
                 "public_dice": e.get("dice")})
rows.sort(key=lambda r: r["created_at"])
with open("paper/figures/data/submission_timeline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "published submissions;", sum(1 for r in rows if r["submission_id"] not in sub), "without a submission record;",
      rows[0]["created_at"][:10], "to", rows[-1]["created_at"][:10])
