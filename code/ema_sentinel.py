"""Sidecar EMA sentinel: evals the EMA weights from runs/s1/last.pt whenever
training saves a new checkpoint. Read-only w.r.t. training; runs in its own
tmux session and GPU process. Deploy via scp (NOT ssh heredoc: nested quoting
corrupts single quotes in the script body)."""
import time, os, sys, torch, logging
sys.path.insert(0, "/root/autodl-tmp/depth300m_v2")
logging.getLogger().setLevel(logging.WARNING)
import student as S
from train import fast_eval_nyu, fast_eval_ppr, nyu_eval_list
from dataset import build_entries, DualTeacherBatchDataset

CK = "/root/autodl-tmp/depth300m_v2/runs/s1/last.pt"
LOG = open("/root/autodl-tmp/depth300m_v2/ema_sentinel.log", "a")
last_mt = 0
nyu_list = nyu_eval_list(100)
entries = build_entries()
ds = DualTeacherBatchDataset(entries)
while True:
    try:
        mt = os.path.getmtime(CK)
        if mt != last_mt:
            ck = torch.load(CK, map_location="cpu")
            m = S.load_student()
            m.load_state_dict(ck["ema"], strict=False)
            ar, d1 = fast_eval_nyu(m, nyu_list)
            corr, _ = fast_eval_ppr(m, ds, entries)
            line = "[ema] iter=%d nyu=%.4f d1=%.4f ppr_corr=%.4f" % (ck["iter"], ar, d1, corr)
            print(line, flush=True)
            LOG.write(line + "\n"); LOG.flush()
            last_mt = mt  # mark consumed only after a successful eval
            del m
            torch.cuda.empty_cache()
    except Exception as e:
        print("[ema] err:", repr(e), flush=True)
    time.sleep(120)
