import json,sqlite3,shutil,tempfile,hashlib
from pathlib import Path
root=Path("/work/dlclarge1/lushtake-hiwi/judge-tuning-heldout-3000-20261015")
cache=Path("/work/dlclarge1/lushtake-hiwi/judge-tuning-local-20261012/cache/judgements/meta-eval-lmarena-140k-en")
report=root/"analysis/cache-integrity.json"
rows=json.loads(report.read_text())["databases"] if report.exists() else []
completed={x["path"] for x in rows}
for p in sorted(cache.rglob("*.db")):
    if str(p) in completed: continue
    print("Checking",str(p),"bytes",p.stat().st_size,flush=True)
    with tempfile.TemporaryDirectory(prefix="ja-check-",dir="/tmp") as tmp:
        local=Path(tmp)/"judgements.db"
        shutil.copyfile(p,local)
        digest=hashlib.file_digest(local.open("rb"),"sha256").hexdigest()
        print("Copied to node-local disk",p.name,flush=True)
        with sqlite3.connect(local.as_uri()+"?mode=ro",uri=True) as conn:
            result=[r[0] for r in conn.execute("PRAGMA integrity_check")]
            mode=conn.execute("PRAGMA journal_mode").fetchone()[0]
    rows.append({"path":str(p),"integrity":result,"journal_mode":mode,"checked_copy_sha256":digest})
    report.write_text(json.dumps({"databases":rows,"complete":False,"all_ok":False},indent=2))
    print("Finished",len(rows),result,mode,flush=True)
out={"complete":True,"databases":rows,"all_ok":bool(rows) and all(r["integrity"]==["ok"] and r["journal_mode"]!="wal" for r in rows)}
(root/"analysis/cache-integrity.json").write_text(json.dumps(out,indent=2))
print("databases",len(rows),"all_ok",out["all_ok"],flush=True)
assert out["all_ok"]
