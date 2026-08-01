import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.image")
import os
import time
import torch
import numpy as np
import math
import csv
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

from folder_manager import load_config, create_run_folders
from utils.logger import ExperimentLogger
from data.dataset import split_train_val, TrainDataset
from model.unet import UNetCond
from model.conditioning import ConditionEmbed
from model.diffusion import Diffusion
from torchvision.utils import save_image
from utils.mask_utils import build_mask_batch

def main(config_path=None):
    if config_path is None:
        config_path = (Path(__file__).resolve().parent.parent / "configs" / "config.yaml")
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = config_path.resolve()

    if not config_path.exists():
        tried = [str(config_path), str((Path(__file__).resolve().parent.parent / "configs" / "config.yaml"))]
        raise FileNotFoundError(f"Config file not found. Tried: {tried}")
        
    cfg = load_config(config_path)
    paths = create_run_folders(cfg)
    logger = ExperimentLogger(paths, use_tb=cfg['logging'].get('tensorboard', True))
    logger.log(f"Loaded config and created folders at {paths['run_root']}")

    origin_dir = cfg['data']['origin_train_dir']
    origin_csv = cfg['meta']['origin_train_csv']
    train_dir = cfg['data']['train_dir']
    val_dir = cfg['data']['val_dir']
    val_ratio = cfg['train'].get('val_split', 0.1)

    origin_csv_path = Path(origin_csv)
    if not origin_csv_path.exists():
        alt = Path(origin_dir) / 'data.csv'
        alt2 = Path(origin_dir) / 'meta.csv'
        if alt.exists():
            origin_csv_path = alt
        elif alt2.exists():
            origin_csv_path = alt2
        else:
            tried = [str(origin_csv_path), str(alt), str(alt2)]
            raise FileNotFoundError(f"Origin CSV not found. Tried: {tried}.")

    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
            logger.log(f"Cleared directory {d}")
    
    split_train_val(origin_dir, str(origin_csv_path), train_dir, val_dir, val_ratio=val_ratio, seed=cfg['experiment'].get('seed', 42))
    logger.log(f"Allocated data: train={train_dir}, val={val_dir}")

    image_size = cfg['model']['image_size']
    train_dataset = TrainDataset(train_dir, meta_csv=os.path.join(train_dir, "meta.csv"), image_size=image_size)
    val_dataset = TrainDataset(val_dir, meta_csv=os.path.join(val_dir, "meta.csv"), image_size=image_size, stats=train_dataset.stats)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['util'].get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['util'].get('num_workers', 4), pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetCond(in_ch=cfg['model'].get('in_channels', 3), base_ch=cfg['model']['base_channels'], cond_dim=cfg['model']['cond_dim']).to(device)
    cond_embed = ConditionEmbed(in_dim=cfg['model'].get('cond_in_dim', 3), cond_dim=cfg['model']['cond_dim']).to(device)
    diffusion = Diffusion(model, cond_embed, timesteps=cfg['train']['timesteps'], device=device)

    optim = torch.optim.Adam(list(model.parameters()) + list(cond_embed.parameters()), lr=cfg['train']['lr'])

    resume = cfg['train'].get('resume_ckpt', None)
    step = 0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        try:
            model.load_state_dict(ck['model_state'])
            cond_embed.load_state_dict(ck['cond_state'])
            optim.load_state_dict(ck['opt_state'])
            step = ck.get('global_step', 0)
            logger.log(f"Resumed from {resume} at step {step}")
        except Exception as e:
            logger.log(f"Warning: resume failed: {e}")

    total_steps = cfg['train']['epochs'] * len(train_loader)
    logger.log(f"Training for {cfg['train']['epochs']} epochs, approx steps {total_steps}")
    losses = []
    start_time = time.time()
    
    train_step_records = []
    train_epoch_means = []
    val_epoch_means = []
    
    val_records_csv = os.path.join(paths['run_root'], 'val_samples_records.csv')
    with open(val_records_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'sample_idx', 'filename', 'cond_0', 'cond_1', 'cond_2', 'sample_path'])

    for epoch in range(cfg['train']['epochs']):
        model.train()
        epoch_start = time.time()
        epoch_losses = []
        
        masks_cfg = cfg.get('masks', {})
        train_mask_cfg = masks_cfg.get('train', {'mode': 'full'})
        val_mask_cfg = masks_cfg.get('val', {'mode': 'full'})

        total_epochs = max(int(cfg['train']['epochs']), 1)
        progress = 1.0 if total_epochs == 1 else float(epoch) / float(total_epochs - 1)

        def _progressive_value(cfg_section, key, default):
            start = cfg_section.get(f"{key}_start", None)
            end = cfg_section.get(f"{key}_end", None)
            if start is not None and end is not None:
                try:
                    start_f = float(start)
                    end_f = float(end)
                    val = start_f + progress * (end_f - start_f)
                    return float(val)
                except Exception:
                    return float(cfg_section.get(key, default))
            return float(cfg_section.get(key, default))

        def _schedule_value(spec: dict, default: float) -> float:
            try:
                stype = str(spec.get('type', 'constant')).lower()
                start = float(spec.get('start', spec.get('value', default)))
                end = float(spec.get('end', spec.get('value', default)))
                if stype == 'constant':
                    return float(spec.get('value', default))
                elif stype == 'linear':
                    return float(start + progress * (end - start))
                elif stype == 'cosine':
                    w = 0.5 * (1.0 - math.cos(math.pi * progress))
                    return float(start + w * (end - start))
                elif stype == 'exponential':
                    if start <= 0 or end <= 0:
                        return float(start + progress * (end - start))
                    ratio = end / start
                    return float(start * (ratio ** progress))
                elif stype == 'step':
                    n_steps = int(spec.get('n_steps', 10))
                    n_steps = max(n_steps, 1)
                    k = int(progress * n_steps)
                    w = float(k) / float(n_steps)
                    return float(start + w * (end - start))
                else:
                    return float(default)
            except Exception:
                return float(default)

        def _schedule_value_by_type(stype: str, param_spec: dict, default: float) -> float:
            try:
                stype = str(stype).lower()
                if stype == 'constant':
                    return float(param_spec.get('value', default))
                start = float(param_spec.get('start', default))
                end = float(param_spec.get('end', default))
                if stype == 'linear':
                    return float(start + progress * (end - start))
                elif stype == 'cosine':
                    w = 0.5 * (1.0 - math.cos(math.pi * progress))
                    return float(start + w * (end - start))
                elif stype == 'exponential':
                    if start <= 0 or end <= 0:
                        return float(start + progress * (end - start))
                    ratio = end / start
                    return float(start * (ratio ** progress))
                elif stype == 'step':
                    n_steps = int(param_spec.get('n_steps', 10))
                    n_steps = max(n_steps, 1)
                    k = int(progress * n_steps)
                    w = float(k) / float(n_steps)
                    return float(start + w * (end - start))
                else:
                    return float(default)
            except Exception:
                return float(default)

        def _get_mask_param(stage_cfg: dict, key: str, default: float) -> float:
            sched_block = stage_cfg.get('schedule', None)
            if isinstance(sched_block, dict):
                stype = sched_block.get('type', 'constant')
                params = sched_block.get('params', {})
                param_spec = params.get(key)
                if isinstance(param_spec, dict):
                    val = _schedule_value_by_type(stype, param_spec, default)
                else:
                    val = float(stage_cfg.get(key, default))
            else:
                scheds = stage_cfg.get('param_schedules', {})
                spec = scheds.get(key, None)
                if isinstance(spec, dict):
                    val = _schedule_value(spec, default)
                else:
                    val = _progressive_value(stage_cfg, key, default)
            
            if key in ('random_ratio', 'hybrid_random_ratio', 'center_scale'):
                val = max(0.0, min(1.0, float(val)))
            return float(val)

        def _get_mode_param(stage_cfg: dict, mode_key: str, param: str, default: float) -> float:
            block = stage_cfg.get(mode_key, {})
            sched = block.get('schedule', None)
            if isinstance(sched, dict):
                stype = str(sched.get('type', 'constant')).lower()
                params = sched.get('params', {})
                spec = params.get(param, None)
                if isinstance(spec, dict):
                    return _schedule_value_by_type(stype, spec, default)
            val = block.get(param, stage_cfg.get(param, default))
            return float(val)

        train_mode = str(train_mask_cfg.get('mode', 'full')).lower()
        val_mode = str(val_mask_cfg.get('mode', 'full')).lower()
        eff_center_train = _get_mode_param(train_mask_cfg, 'center', 'center_scale', 0.5) if train_mode == 'center' else _get_mode_param(train_mask_cfg, 'hybrid', 'hybrid_center', 0.5)
        eff_center_val = _get_mode_param(val_mask_cfg, 'center', 'center_scale', 0.5) if val_mode == 'center' else _get_mode_param(val_mask_cfg, 'hybrid', 'hybrid_center', 0.5)
        eff_random_train = _get_mode_param(train_mask_cfg, 'random', 'random_ratio', 0.5)
        eff_random_val = _get_mode_param(val_mask_cfg, 'random', 'random_ratio', 0.5)
        eff_hybrid_random_train = _get_mode_param(train_mask_cfg, 'hybrid', 'hybrid_random', 0.25)
        eff_hybrid_random_val = _get_mode_param(val_mask_cfg, 'hybrid', 'hybrid_random', 0.25)

        if train_mode == 'hybrid' and val_mode == 'hybrid':
            logger.log(
                f"Epoch {epoch} mask params | train(mode=hybrid): hybrid_center={eff_center_train:.3f}, hybrid_random={eff_hybrid_random_train:.3f} | "
                f"val(mode=hybrid): hybrid_center={eff_center_val:.3f}, hybrid_random={eff_hybrid_random_val:.3f}"
            )
        elif train_mode == 'center' and val_mode == 'center':
            logger.log(
                f"Epoch {epoch} mask params | train(mode=center): center={eff_center_train:.3f} | val(mode=center): center={eff_center_val:.3f}"
            )
        elif train_mode == 'random' and val_mode == 'random':
            logger.log(
                f"Epoch {epoch} mask params | train(mode=random): random={eff_random_train:.3f} | val(mode=random): random={eff_random_val:.3f}"
            )
        else:
            logger.log(
                f"Epoch {epoch} mask params | train(mode={train_mode}): center={eff_center_train:.3f}, random={eff_random_train:.3f}, hybrid_random={eff_hybrid_random_train:.3f} | "
                f"val(mode={val_mode}): center={eff_center_val:.3f}, random={eff_random_val:.3f}, hybrid_random={eff_hybrid_random_val:.3f}"
            )

        for imgs, conds, fnames in train_loader:
            imgs = imgs.to(device)
            conds = conds.to(device)
            B = imgs.shape[0]
            t = torch.randint(0, cfg['train']['timesteps'], (B,), device=device).long()

            try:
                mask_train = build_mask_batch(
                    mode=train_mask_cfg.get('mode', 'full'),
                    imgs=imgs,
                    center_scale=eff_center_train,
                    random_ratio=eff_random_train,
                    hybrid_random_ratio=eff_hybrid_random_train,
                    filenames=fnames if train_mask_cfg.get('mode', 'full') == 'self' else None,
                    self_mask_dir=train_mask_cfg.get('self_mask_dir')
                )
            except Exception:
                mask_train = build_mask_batch(mode='full', imgs=imgs)

            loss = diffusion.p_losses(imgs, t, conds, cond_drop_prob=cfg['train']['cond_drop_prob'], mask=mask_train)
            optim.zero_grad()
            loss.backward()
            
            if cfg['util'].get('clip_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['util'].get('clip_grad_norm', 1.0))
            optim.step()

            step += 1
            losses.append(loss.item())
            epoch_losses.append(loss.item())
            train_step_records.append((step, float(loss.item())))

            if step % cfg['train']['log_interval'] == 0:
                avg = np.mean(losses[-cfg['train']['log_interval']:])
                elapsed = time.time() - start_time
                logger.log(f"Epoch {epoch} Step {step} Loss {avg:.6f} Elapsed {elapsed:.1f}s")
                logger.add_scalar('train/loss', float(avg), step)
                logger.write_metric_row(step, 'train', float(avg), elapsed)

        epoch_time = time.time() - epoch_start
        if len(epoch_losses) > 0:
            mean_epoch_loss = float(np.mean(epoch_losses))
        else:
            mean_epoch_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        logger.log(f"Epoch {epoch} finished in {epoch_time:.1f}s. Train loss: {mean_epoch_loss:.6f}")
        logger.add_scalar('train/epoch_loss', mean_epoch_loss, epoch)
        logger.write_metric_row(step, 'train_epoch', mean_epoch_loss, epoch_time)
        train_epoch_means.append(mean_epoch_loss)

        save_every_epochs = int(cfg['train'].get('save_every_epochs', 1))
        if (epoch + 1) % save_every_epochs == 0:
            ck_path = os.path.join(paths['checkpoints'], f"ckpt_epoch_{epoch+1}.pth")
            torch.save({
                'model_state': model.state_dict(),
                'cond_state': cond_embed.state_dict(),
                'opt_state': optim.state_dict(),
                'global_step': step
            }, ck_path)
            logger.log(f"Saved checkpoint (epoch {epoch+1}) to {ck_path}")

        if (epoch + 1) % cfg['train'].get('val_every_epochs', 1) == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for imgs, conds, fnames in val_loader:
                    imgs = imgs.to(device)
                    conds = conds.to(device)
                    B = imgs.shape[0]
                    t = torch.randint(0, cfg['train']['timesteps'], (B,), device=device).long()

                    try:
                        mask_val = build_mask_batch(
                            mode=val_mask_cfg.get('mode', 'full'),
                            imgs=imgs,
                            center_scale=eff_center_val,
                            random_ratio=eff_random_val,
                            hybrid_random_ratio=eff_hybrid_random_val,
                            filenames=fnames if val_mask_cfg.get('mode', 'full') == 'self' else None,
                            self_mask_dir=val_mask_cfg.get('self_mask_dir')
                        )
                    except Exception:
                        mask_val = build_mask_batch(mode='full', imgs=imgs)

                    l = diffusion.p_losses(imgs, t, conds, cond_drop_prob=cfg['train'].get('cond_drop_prob', 0.1), mask=mask_val)
                    val_losses.append(l.item())

            mean_val = float(np.mean(val_losses))
            logger.log(f"Validation loss at epoch {epoch}: {mean_val:.6f}")
            logger.add_scalar('val/loss', mean_val, step)
            logger.write_metric_row(step, 'val', mean_val, time.time() - start_time)
            val_epoch_means.append(mean_val)

            try:
                sample_batch = next(iter(val_loader))
                imgs_v, conds_v, fnames_v = sample_batch
                n_save = min(4, imgs_v.shape[0])
                conds_gen = conds_v[:n_save].to(device)
                gen = diffusion.p_sample_loop((n_save, cfg['model'].get('in_channels', 3), image_size, image_size), conds_gen, guidance_scale=cfg['sampling'].get('guidance_scale_other', 1.0))
                
                with open(val_records_csv, 'a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    for i in range(n_save):
                        out_img = (gen[i].cpu() + 1.0) / 2.0
                        sample_path = os.path.join(paths['val_generation'], f"epoch{epoch}_sample_{i}.png")
                        save_image(out_img, sample_path)

                        c0_norm, c1_norm, c2_norm = conds_v[i].cpu().numpy()[:3]
                        c0 = float(c0_norm * train_dataset.stats['cond_0']['std'] + train_dataset.stats['cond_0']['mean']) if 'cond_0' in train_dataset.stats else float(c0_norm)
                        c1 = float(c1_norm * train_dataset.stats['cond_1']['std'] + train_dataset.stats['cond_1']['mean']) if 'cond_1' in train_dataset.stats else float(c1_norm)
                        c2 = float(c2_norm * train_dataset.stats['cond_2']['std'] + train_dataset.stats['cond_2']['mean']) if 'cond_2' in train_dataset.stats else float(c2_norm)
                        
                        fname = fnames_v[i] if i < len(fnames_v) else f'sample_{i}'
                        w.writerow([epoch, i, fname, f"{c0:.3f}", f"{c1:.3f}", f"{c2:.3f}", sample_path])

                try:
                    os.makedirs(os.path.join(paths['run_root'], 'val_inpaint'), exist_ok=True)
                    imgs_v = imgs_v[:n_save].to(device)
                    conds_v = conds_v[:n_save].to(device)

                    try:
                        mask_v = build_mask_batch(
                            mode=val_mask_cfg.get('mode', 'full'),
                            imgs=imgs_v,
                            center_scale=eff_center_val,
                            random_ratio=eff_random_val,
                            hybrid_random_ratio=eff_hybrid_random_val,
                            filenames=fnames_v[:n_save] if val_mask_cfg.get('mode', 'full') == 'self' else None,
                            self_mask_dir=val_mask_cfg.get('self_mask_dir')
                        )
                    except Exception:
                        mask_v = build_mask_batch(mode='full', imgs=imgs_v)

                    for i in range(n_save):
                        mask_vis = mask_v[i]
                        if mask_vis.dim() == 2:
                            mask_vis = mask_vis.unsqueeze(0)
                        elif mask_vis.dim() == 3 and mask_vis.shape[0] != 1:
                            mask_vis = mask_vis[:1]
                        mask_vis = mask_vis.float().cpu()
                        mask_path = os.path.join(paths['run_root'], 'val_inpaint', f"epoch{epoch}_mask_{i}_{val_mask_cfg.get('mode', 'full')}.png")
                        save_image(mask_vis, mask_path)

                    gen_inpaint = diffusion.p_sample_loop_inpaint(imgs_v, mask_v, conds_v, guidance_scale=cfg['sampling'].get('guidance_scale_other', 1.0))
                    for i in range(n_save):
                        out_img = (gen_inpaint[i].cpu() + 1.0) / 2.0
                        out_path = os.path.join(paths['run_root'], 'val_inpaint', f"epoch{epoch}_inpaint_{i}.png")
                        save_image(out_img, out_path)
                    logger.log(f"Saved val inpaint samples with mask mode '{val_mask_cfg.get('mode', 'full')}' to {os.path.join(paths['run_root'], 'val_inpaint')}")
                except Exception as e:
                    logger.log(f"Warning: saving val inpaint samples failed: {e}")
            except Exception as e:
                logger.log(f"Warning: saving validation samples failed: {e}")

    final_ckpt = os.path.join(paths['checkpoints'], "final_ckpt.pth")
    torch.save({'model_state': model.state_dict(), 'cond_state': cond_embed.state_dict(), 'opt_state': optim.state_dict(), 'global_step': step}, final_ckpt)
    logger.log(f"Training finished. Final checkpoint saved to {final_ckpt}")

    try:
        train_step_csv = os.path.join(paths['run_root'], 'train_step_losses.csv')
        with open(train_step_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['step', 'loss'])
            for s, l in train_step_records:
                w.writerow([s, f"{l:.6f}"])
        logger.log(f"Saved train step losses CSV to {train_step_csv}")

        train_epoch_csv = os.path.join(paths['run_root'], 'train_epoch_losses.csv')
        with open(train_epoch_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'mean_train_loss'])
            for e, l in enumerate(train_epoch_means):
                w.writerow([e, f"{l:.6f}"])
        logger.log(f"Saved train epoch losses CSV to {train_epoch_csv}")

        val_epoch_csv = os.path.join(paths['run_root'], 'val_epoch_losses.csv')
        with open(val_epoch_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'mean_val_loss'])
            for e, l in enumerate(val_epoch_means):
                w.writerow([e, f"{l:.6f}"])
        logger.log(f"Saved val epoch losses CSV to {val_epoch_csv}")
    except Exception as e:
        logger.log(f"Warning: failed to write loss CSVs: {e}")

    try:
        if len(train_epoch_means) > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(range(len(train_epoch_means)), train_epoch_means, label='Train', color='tab:blue')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train Loss Curve')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            train_curve_path = os.path.join(paths['run_root'], 'train_loss_curve.png')
            plt.savefig(train_curve_path)
            plt.close()
            logger.log(f"Saved train loss curve to {train_curve_path}")

        if len(val_epoch_means) > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(range(len(val_epoch_means)), val_epoch_means, label='Val', color='tab:orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Val Loss Curve')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            val_curve_path = os.path.join(paths['run_root'], 'val_loss_curve.png')
            plt.savefig(val_curve_path)
            plt.close()
            logger.log(f"Saved val loss curve to {val_curve_path}")
    except Exception as e:
        logger.log(f"Warning: failed to save loss curves: {e}")
        
    logger.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        main(None)
    else:
        main(sys.argv[1])