"""
src/test.py - Inference & Inpainting Execution Script
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.image")

import os
import sys
import json
import csv
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.utils import save_image

from folder_manager import load_config, create_run_folders
from utils.logger import ExperimentLogger
from data.dataset import TestInpaintDataset, get_transform
from model.unet import UNetCond
from model.conditioning import ConditionEmbed
from model.diffusion import Diffusion
from utils.mask_utils import build_mask_batch

try:
    from tools import evaluator as _ev
    _HAS_EVAL = True
except Exception:
    _ev = None
    _HAS_EVAL = False


def _n_proc(r, s):
    def _c(v, k):
        if v == 'n' or v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return (float(v) - s[k]['mean']) / (s[k]['std'] + 1e-8)
    return _c(r.get('FAR'), 'FAR'), _c(r.get('n_buildings'), 'n_buildings'), _c(r.get('UTCI'), 'UTCI')


def _fetch_stats(r_path, cfg=None):
    p0 = os.path.join(r_path, "data_stats.json")
    if os.path.exists(p0):
        with open(p0, 'r', encoding='utf-8') as f:
            return json.load(f)

    cands = []
    if cfg and 'data' in cfg and 'train_dir' in cfg['data']:
        cands.append(os.path.join(cfg['data']['train_dir'], "train_meta.csv"))
    cands.extend([
        os.path.join(r_path, "..", "..", "data", "train", "train_meta.csv"),
        os.path.join(r_path, "..", "data", "train", "train_meta.csv")
    ])

    for c in cands:
        if os.path.exists(c):
            df = pd.read_csv(c, encoding='utf-8')
            res = {}
            for col in ['FAR', 'n_buildings', 'UTCI']:
                arr = df[col].astype(float).values
                res[col] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr)) if np.std(arr) > 0 else 1.0}
            return res
    raise FileNotFoundError("Missing data_stats.json or fallback CSVs.")


def main(cfg_p, ckpt_p):
    _cfg = load_config(cfg_p)
    _paths = create_run_folders(_cfg)
    _log = ExperimentLogger(_paths, use_tb=False)
    _log.log(f"Init run: {ckpt_p}")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(ckpt_p):
        raise FileNotFoundError(f"Checkpoint unavailable: {ckpt_p}")

    ckpt_data = torch.load(ckpt_p, map_location=dev)
    net = UNetCond(in_ch=3, base_ch=_cfg['model']['base_channels'], cond_dim=_cfg['model']['cond_dim']).to(dev)
    c_emb = ConditionEmbed(in_dim=3, cond_dim=_cfg['model']['cond_dim']).to(dev)

    net.load_state_dict(ckpt_data['model_state'])
    c_emb.load_state_dict(ckpt_data['cond_state'])
    net.eval()

    diff_engine = Diffusion(net, c_emb, timesteps=_cfg['train']['timesteps'], device=dev)
    stats_map = _fetch_stats(os.path.dirname(os.path.dirname(ckpt_p)), _cfg)

    t_csv = _cfg['meta']['test_meta_csv']
    if not os.path.exists(t_csv):
        raise FileNotFoundError(f"Meta file missing: {t_csv}")

    meta_df = pd.read_csv(t_csv, encoding='utf-8')
    t_dir = _cfg['data']['test_dir']
    m_cfg = _cfg.get('masks', {}).get('test', {'mode': 'self'})
    m_mode = str(m_cfg.get('mode', 'self')).lower()

    if m_mode == 'self':
        m_dir = m_cfg.get('self_mask_dir', _cfg['data'].get('test_mask_dir'))
        ds = TestInpaintDataset(
            t_dir, m_dir, 
            image_size=_cfg['model']['image_size'],
            invert_self_mask=bool(m_cfg.get('invert_self_mask', False)),
            dilate_px=int(m_cfg.get('dilate_px', 0)),
            mask_threshold=int(m_cfg.get('mask_threshold', 0))
        )
        data_loader = enumerate(ds)
        mode_ds = True
    else:
        raw_files = sorted([f for f in os.listdir(t_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        t_func = get_transform(_cfg['model']['image_size'])
        data_loader = enumerate(raw_files)
        mode_ds = False

    records = []
    max_r = _cfg['sampling'].get('utci_threshold_retry', 5)

    for idx, sample_item in data_loader:
        if mode_ds:
            img_t, mask_t, fname = sample_item
        else:
            fname = sample_item
            raw_img = Image.open(os.path.join(t_dir, fname)).convert('RGB')
            img_t = t_func(raw_img)

            def _parse_m(st_cfg):
                st_m = str(st_cfg.get('mode', 'full')).lower()
                def _b_val(blk, k, def_v):
                    sch = blk.get('schedule')
                    if isinstance(sch, dict) and str(sch.get('type')).lower() != 'constant':
                        p = sch.get('params', {}).get(k, {})
                        return float(p.get('start', def_v)) if isinstance(p, dict) else def_v
                    return float(blk.get(k, def_v))

                c_val, r_val, h_val = 0.5, 0.5, 0.25
                if st_m == 'hybrid':
                    c_val = _b_val(st_cfg.get('hybrid', {}), 'hybrid_center', c_val)
                    h_val = _b_val(st_cfg.get('hybrid', {}), 'hybrid_random', h_val)
                elif st_m == 'center':
                    c_val = _b_val(st_cfg.get('center', {}), 'center_scale', c_val)
                elif st_m == 'random':
                    r_val = _b_val(st_cfg.get('random', {}), 'random_ratio', r_val)
                return max(0.0, min(1.0, c_val)), max(0.0, min(1.0, r_val)), max(0.0, min(1.0, h_val))

            c_e, r_e, h_e = _parse_m(m_cfg)
            mb = build_mask_batch(
                mode=m_mode, imgs=img_t.unsqueeze(0),
                center_scale=c_e, random_ratio=r_e, hybrid_random_ratio=h_e,
                filenames=[fname] if m_mode == 'self' else None,
                self_mask_dir=m_cfg.get('self_mask_dir'),
                invert_self_mask=bool(m_cfg.get('invert_self_mask', False)),
                mask_threshold=int(m_cfg.get('mask_threshold', 0))
            )
            mask_t = mb.squeeze(0)

        _log.log(f"Processing #{idx}: {fname}")

        # Mask Diagnostics & Output
        try:
            m_v = mask_t[:1] if (mask_t.dim() == 3 and mask_t.shape[0] != 1) else mask_t
            stem = os.path.splitext(fname)[0]
            save_image(m_v.float(), os.path.join(_paths['samples'], f"{stem}_mask_{m_mode}.png"))
            
            ov = (img_t.clone().cpu() + 1.0) / 2.0
            mk = m_v.squeeze(0).cpu() > 0.5
            ov[0][mk], ov[1][mk], ov[2][mk] = 1.0, 0.0, 0.0
            save_image(ov, os.path.join(_paths['samples'], f"{stem}_overlay_mask.png"))
        except Exception:
            pass

        r_match = meta_df[meta_df['filename'] == fname]
        if r_match.empty:
            continue
        r_dict = r_match.iloc[0].to_dict()

        f_n, b_n, u_n = _n_proc(r_dict, stats_map)
        c_raw = [f_n or 0.0, b_n or 0.0, u_n or 0.0]
        c_flag = [f_n is not None, b_n is not None, u_n is not None]

        c_t = torch.tensor(c_raw, dtype=torch.float32).unsqueeze(0).to(dev)
        i_t = img_t.unsqueeze(0).to(dev)
        m_in = mask_t.unsqueeze(0).to(dev)

        s_cfg = _cfg['sampling']
        g_scale = s_cfg.get('guidance_scale_utci', 3.0) if c_flag[2] else s_cfg.get('guidance_scale_other', 1.0)

        done = False
        step = 0
        while not done and step < max_r:
            step += 1
            gen_tensor = diff_engine.p_sample_loop_inpaint(i_t, m_in, c_t, guidance_scale=g_scale)
            gen_img = (gen_tensor.squeeze(0).cpu() + 1.0) / 2.0

            out_n = f"{os.path.splitext(fname)[0]}_gen_attempt{step}.png"
            out_p = os.path.join(_paths['samples'], out_n)
            save_image(gen_img, out_p)

            p_far, p_nb, p_utci = None, None, None
            if _HAS_EVAL:
                if hasattr(_ev, 'predict_far'):
                    try: p_far = float(_ev.predict_far(out_p))
                    except Exception: pass
                if hasattr(_ev, 'predict_nbuildings'):
                    try: p_nb = float(_ev.predict_nbuildings(out_p))
                    except Exception: pass
                if hasattr(_ev, 'predict_utci'):
                    try: p_utci = float(_ev.predict_utci(out_p))
                    except Exception: pass

            records.append({
                'gen_name': out_n, 'src_fname': fname,
                'FAR': r_dict.get('FAR', 'n'), 'n_buildings': r_dict.get('n_buildings', 'n'), 'UTCI': r_dict.get('UTCI', 'n'),
                'pred_FAR': p_far if p_far is not None else 'n',
                'pred_n_buildings': p_nb if p_nb is not None else 'n',
                'pred_UTCI': p_utci if p_utci is not None else 'n',
                'attempt': step, 'ckpt': os.path.basename(ckpt_p)
            })
            done = True

    out_csv_path = os.path.join(_paths['run_root'], 'test_generation_records.csv')
    fields = ['gen_name', 'src_fname', 'FAR', 'n_buildings', 'UTCI', 'pred_FAR', 'pred_n_buildings', 'pred_UTCI', 'attempt', 'ckpt']
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    _log.log(f"Execution complete. Results logged to {out_csv_path}")
    _log.close()


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    else:
        sys.exit(1)