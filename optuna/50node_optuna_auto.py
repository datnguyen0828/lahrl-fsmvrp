##########################################################################################
# Machine Environment Config
##########################################################################################
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
##########################################################################################
import copy
import json
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

import optuna
from optuna.exceptions import TrialPruned
from optuna.importance import get_param_importances
from optuna.pruners import MedianPruner

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import create_logger
from FSMVRP_Trainer import FSMVRPTrainer_PPO as Trainer

##########################################################################################
# Parameters for MAP 50 NODES
##########################################################################################

BASE_ENV_PARAMS = {
    'min_problem_size': 50,
    'max_problem_size': 50,
    'pomo_size': 50,
    'min_agent_num': 3,
    'max_agent_num': 6,
    # Utilization penalty TẮT cho nhất quán với 20-node setup.
    'utilization_penalty': {
        'enable': False,
        'ratio_threshold': 0.8,
        'weight': 3.0,
        'power': 2.0,
        'min_demand': 0.0,
    },
}

BASE_MODEL_PARAMS = {
    'embedding_dim': 128,
    'encoder_layer_num': 6,
    'head_num': 8,
    'qkv_dim': 16,
    'ff_hidden_dim': 256,
    'logit_clipping': 10.0,
    # Initial future_beta = 0 — annealing sẽ nâng dần lên final_future_beta.
    'future_beta': 0.0,
    'eval_type': 'softmax',
}

BASE_OPTIMIZER_PARAMS = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6,
    },
    'scheduler': {
        # Milestones scale theo 60-epoch HPO run.
        'milestones': [25, 40, 55],
        'gamma': 0.5,
    },
}

BASE_TRAINER_PARAMS = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    # 50 node cần nhiều epoch hơn 20 node để policy hội tụ đủ cho HPO so sánh.
    'epochs': 60,
    'train_episodes': 2000,
    # Với gradient_checkpointing=True, VRAM giảm ~60% nên batch_size có thể
    # tăng lên đáng kể. 128 là con số an toàn cho 5090 32GB. Nếu vẫn còn dư
    # VRAM, có thể tăng lên 192 hoặc 256.
    'train_batch_size': 400,
    'critic_hidden_dim': 128,
    'ppo': {
        'epsilon': 0.2,
        'ppo_epochs': 3,
        'gamma': 0.99,
        # lambda_future > 0 xuyên suốt để MLP future_cost_mlp học song song với policy.
        'lambda_future': 0.5,
        'alpha_entropy': 0.05,
        'c_critic': 0.5,
    },

    # [SPEEDUP] Gradient checkpointing: tiết kiệm ~60% VRAM trong _ppo_update
    # bằng cách recompute forward của decoder khi backward. Đánh đổi ~15-25%
    # tốc độ mỗi batch, nhưng cho phép tăng batch_size gấp 2-3× → net tốc độ
    # cao hơn và GPU utilization cao hơn. Khuyến nghị BẬT cho 50+ node.
    'gradient_checkpointing': {
        'enable': True,
    },
    'validation': {
        'enable': True,
        'episodes': 300,
        'batch_size': 100,
        'seed': 9999,
        'aug_factor': 8,
        'primary_eval_type': 'softmax',
        'secondary_eval_type': 'argmax',
        'objective_metric': 'val_softmax_aug_raw_score',
    },
    'annealing': {
        'enable': True,
        # ~30% đầu để MLP warmup (epoch 0–20 giữ β=0), sau đó ~40 epoch tăng β.
        'start_epoch': 20,
        'final_alpha_entropy': 1e-5,
        'final_ppo_epsilon': 0.12,
        'final_future_beta': 1.0,
    },
    'model_load': {
        'enable': False,
        'path': str(PROJECT_ROOT / 'results' / 'train_50nodes'),
        'epoch': 0,
    },
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 5,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_train_score.json',
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_train_loss.json',
        },
    },
}

RESULTS_BASE_DIR = PROJECT_ROOT / 'optuna_results' / 'train_50nodes_auto'
DB_PATH = PROJECT_ROOT / 'optuna_results' / 'fsmvrp_50node_auto_hpo.db'

# Seed có thể override qua env var khi chạy song song 2 process.
import os as _os
_DEFAULT_SEED = 1234
_SEED = int(_os.getenv('OPTUNA_SEED', str(_DEFAULT_SEED)))

OPTUNA_PARAMS = {
    'study_name': 'fsmvrp_50node_auto_hpo',
    'storage': f"sqlite:///{DB_PATH.as_posix()}",
    # Tăng nhẹ từ 120 → 140 để bù thông tin mất khi chạy song song.
    'n_trials': 140,
    'timeout': None,
    'seed': _SEED,
}

logger_params = {
    'log_file': {
        'filepath': '',
        'filename': 'study.log',
    }
}

# if want to resume last run
RESUME_LAST_RUN = False


def _next_run_index(base_dir):
    max_index = 0
    for path in base_dir.glob('run_*'):
        if not path.is_dir():
            continue
        try:
            max_index = max(max_index, int(path.name.split('_')[-1]))
        except ValueError:
            continue
    if RESUME_LAST_RUN and max_index > 0:
        return max_index
    return max_index + 1


RUN_INDEX = _next_run_index(RESULTS_BASE_DIR)
RUN_TAG = f"run_{RUN_INDEX:04d}"
RUN_BASE_DIR = RESULTS_BASE_DIR / RUN_TAG
TRIALS_DIR = RUN_BASE_DIR / 'trials'
logger_params['log_file']['filepath'] = str(RUN_BASE_DIR)


def _load_auto_sampler(seed):
    try:
        import optunahub
    except ImportError as exc:
        raise ImportError(
            "AutoSampler requires optunahub. Install it first, for example with "
            "`pip install optunahub cmaes scipy`."
        ) from exc

    try:
        module = optunahub.load_module(package="samplers/auto_sampler")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load AutoSampler from OptunaHub. Make sure optunahub is installed "
            "and the AutoSampler package is available in the local cache or via network access."
        ) from exc

    return module.AutoSampler(seed=seed)


def _build_trial_configs(trial):
    env_params = copy.deepcopy(BASE_ENV_PARAMS)
    model_params = copy.deepcopy(BASE_MODEL_PARAMS)
    optimizer_params = copy.deepcopy(BASE_OPTIMIZER_PARAMS)
    trainer_params = copy.deepcopy(BASE_TRAINER_PARAMS)

    # --- Optimizer ---
    # Đổi từ categorical sang log-uniform để không gian tìm kiếm rộng hơn.
    optimizer_params['optimizer']['lr'] = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    # Thu hẹp range weight_decay: [1e-8, 1e-4] quá rộng, xoay quanh vùng 20-node best.
    optimizer_params['optimizer']['weight_decay'] = trial.suggest_float('weight_decay', 1e-9, 1e-6, log=True)
    optimizer_params['scheduler']['gamma'] = trial.suggest_float('scheduler_gamma', 0.3, 0.7)

    # --- Trainer core ---
    trainer_params['critic_hidden_dim'] = trial.suggest_categorical('critic_hidden_dim', [64, 128, 256])

    # --- PPO ---
    trainer_params['ppo']['epsilon'] = trial.suggest_float('ppo_epsilon', 0.1, 0.3)
    trainer_params['ppo']['ppo_epochs'] = trial.suggest_int('ppo_epochs', 2, 6)
    trainer_params['ppo']['alpha_entropy'] = trial.suggest_float('alpha_entropy', 1e-3, 1e-1, log=True)
    trainer_params['ppo']['c_critic'] = trial.suggest_float('c_critic', 0.1, 1.0)

    # --- Future Cost ---
    trainer_params['ppo']['lambda_future'] = trial.suggest_float('lambda_future', 0.1, 1.0)
    trainer_params['annealing']['final_future_beta'] = trial.suggest_float(
        'final_future_beta', 0.3, 1.5
    )

    return env_params, model_params, optimizer_params, trainer_params


def _make_trial_logger(trial_number):
    trial_dir = TRIALS_DIR / f"trial_{trial_number:04d}"
    create_logger(log_file={
        'filepath': str(trial_dir),
        'filename': 'log.txt',
    })
    return trial_dir


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _extract_val_metrics(trainer):
    result_log = trainer.result_log
    # Ưu tiên raw-score objective để khớp với objective_metric đã đặt.
    candidate_keys = ['val_softmax_aug_raw_score', 'val_softmax_aug_score', 'val_score', 'val_argmax_aug_score']
    for key in candidate_keys:
        if not result_log.has_key(key):
            continue
        val_history = result_log.get(key)
        if isinstance(val_history, list):
            if val_history and isinstance(val_history[0], (list, tuple)) and len(val_history[0]) == 2:
                values = [value for _epoch, value in val_history]
            else:
                values = [float(value) for value in val_history]
            return min(values), values[-1]
        return float(val_history), float(val_history)
    return None, None


def _collect_partial_metrics(trainer, completed_metrics=None):
    metrics = {} if completed_metrics is None else dict(completed_metrics)
    if trainer is None:
        return metrics

    best_val_score, final_val_score = _extract_val_metrics(trainer)
    if best_val_score is not None:
        metrics.setdefault('best_val_score', best_val_score)
        metrics.setdefault('final_val_score', final_val_score)

    return metrics


def _delete_trial_entries(trial):
    trial_id = getattr(trial, '_trial_id', None)
    if trial_id is None:
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('PRAGMA foreign_keys = OFF')
        for table_name in (
            'trial_heartbeats',
            'trial_intermediate_values',
            'trial_params',
            'trial_system_attributes',
            'trial_user_attributes',
            'trial_values',
        ):
            conn.execute(f'DELETE FROM {table_name} WHERE trial_id = ?', (trial_id,))
        conn.execute('DELETE FROM trials WHERE trial_id = ?', (trial_id,))
        conn.commit()
    finally:
        conn.close()


def _cleanup_interrupted_trial(trial, trial_dir, logger):
    shutil.rmtree(trial_dir, ignore_errors=True)
    logger.warning(
        'KeyboardInterrupt detected. Deleted interrupted Optuna trial %d from storage and trial folder.',
        trial.number,
    )
    import threading
    def _delayed_delete():
        import time
        time.sleep(2.0)
        _delete_trial_entries(trial)

    threading.Thread(target=_delayed_delete, daemon=True).start()


def objective(trial):
    env_params, model_params, optimizer_params, trainer_params = _build_trial_configs(trial)
    trial_dir = _make_trial_logger(trial.number)
    logger = logging.getLogger('root')
    trainer = None

    logger.info("Starting run %s | Optuna trial %d", RUN_TAG, trial.number)
    logger.info("Sampled params: %s", trial.params)

    try:
        trainer = Trainer(
            env_params=env_params,
            model_params=model_params,
            optimizer_params=optimizer_params,
            trainer_params=trainer_params,
        )
        metrics = trainer.run(trial=trial)
        metrics = _collect_partial_metrics(trainer, metrics)

        metric_value = metrics['objective_value']
        objective_name = metrics['objective_name']
        trial.set_user_attr('objective_name', objective_name)
        trial.set_user_attr('objective_value', metric_value)
        trial.set_user_attr('run_index', RUN_INDEX)
        trial.set_user_attr('run_tag', RUN_TAG)
        trial.set_user_attr('best_epoch', metrics['best_epoch'])
        trial.set_user_attr('best_train_score', metrics['best_train_score'])
        trial.set_user_attr('best_train_loss', metrics['best_train_loss'])
        trial.set_user_attr('final_train_score', metrics['final_train_score'])
        trial.set_user_attr('final_train_loss', metrics['final_train_loss'])
        trial.set_user_attr('result_folder', metrics.get('result_folder', str(trial_dir)))
        if metrics.get('best_val_score') is not None:
            trial.set_user_attr('best_val_score', metrics['best_val_score'])
            trial.set_user_attr('final_val_score', metrics['final_val_score'])

        _write_json(
            trial_dir / 'summary.json',
            {
                'trial_number': trial.number,
                'run_index': RUN_INDEX,
                'run_tag': RUN_TAG,
                'state': 'COMPLETE',
                'objective': metric_value,
                'objective_name': objective_name,
                'params': trial.params,
                'metrics': metrics,
            }
        )

        return metric_value

    except TrialPruned:
        _write_json(
            trial_dir / 'summary.json',
            {
                'trial_number': trial.number,
                'run_index': RUN_INDEX,
                'run_tag': RUN_TAG,
                'state': 'PRUNED',
                'params': trial.params,
                'metrics': _collect_partial_metrics(trainer),
            }
        )
        raise
    except KeyboardInterrupt:
        _cleanup_interrupted_trial(trial, trial_dir, logger)
        raise
    except Exception as exc:
        _write_json(
            trial_dir / 'summary.json',
            {
                'trial_number': trial.number,
                'run_index': RUN_INDEX,
                'run_tag': RUN_TAG,
                'state': 'FAIL',
                'params': trial.params,
                'metrics': _collect_partial_metrics(trainer),
                'error': repr(exc),
            }
        )
        raise


def _serialize_trial(trial):
    return {
        'number': trial.number,
        'run_index': RUN_INDEX,
        'run_tag': RUN_TAG,
        'value': trial.value,
        'state': str(trial.state),
        'params': trial.params,
        'user_attrs': trial.user_attrs,
    }


def _analyze_late_stage_improvement(completed_trials):
    valid_trials = [
        trial for trial in completed_trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    valid_trials.sort(key=lambda trial: trial.number)

    if len(valid_trials) < 10:
        return {
            'completed_trials': len(valid_trials),
            'enough_data': False,
            'message': 'Need at least 10 completed trials before checking convergence.',
        }

    split_index = max(1, int(len(valid_trials) * 0.8))
    first_chunk = valid_trials[:split_index]
    last_chunk = valid_trials[split_index:]

    if not last_chunk:
        return {
            'completed_trials': len(valid_trials),
            'enough_data': False,
            'message': 'Need at least one completed trial in the last 20% segment.',
        }

    first_80pct_best = min(trial.value for trial in first_chunk)
    last_20pct_best = min(trial.value for trial in last_chunk)

    denominator = max(abs(first_80pct_best), 1e-12)
    improvement = (first_80pct_best - last_20pct_best) / denominator

    if improvement > 0.02:
        recommendation = 'Best value still improved materially in the last 20%; run more trials.'
        converged = False
    elif improvement < 0.005:
        recommendation = 'Late-stage improvement is very small; the study looks close to converged.'
        converged = True
    else:
        recommendation = 'Late-stage improvement is moderate; reassess after a few more trials.'
        converged = False

    return {
        'completed_trials': len(valid_trials),
        'enough_data': True,
        'split_index': split_index,
        'first_80pct_best': first_80pct_best,
        'last_20pct_best': last_20pct_best,
        'late_stage_improvement': improvement,
        'converged': converged,
        'recommendation': recommendation,
    }


def _write_study_reports(study):
    RUN_BASE_DIR.mkdir(parents=True, exist_ok=True)

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    completed_trials.sort(key=lambda trial: trial.value)

    convergence_summary = _analyze_late_stage_improvement(study.trials)
    importances = get_param_importances(study) if completed_trials else {}

    best_trial = study.best_trial if completed_trials else None
    best_payload = {
        'study_name': study.study_name,
        'best_value': best_trial.value if best_trial is not None else None,
        'best_params': best_trial.params if best_trial is not None else {},
        'best_user_attrs': best_trial.user_attrs if best_trial is not None else {},
        'n_trials': len(study.trials),
        'convergence': convergence_summary,
    }

    _write_json(RUN_BASE_DIR / 'best_trial.json', best_payload)
    _write_json(RUN_BASE_DIR / 'param_importance.json', importances)
    _write_json(RUN_BASE_DIR / 'convergence_summary.json', convergence_summary)
    _write_json(
        RUN_BASE_DIR / 'top_trials.json',
        [_serialize_trial(trial) for trial in completed_trials[:10]]
    )

    report_lines = [
        f"study_name: {study.study_name}",
        f"run_tag: {RUN_TAG}",
        f"trial_count: {len(study.trials)}",
        "",
        "[best_trial]",
    ]
    if best_trial is None:
        report_lines.append("none")
    else:
        report_lines.append(f"number: {best_trial.number}")
        report_lines.append(f"value: {best_trial.value:.6f}")
        report_lines.append(f"params: {json.dumps(best_trial.params, sort_keys=True)}")
        report_lines.append("")
        report_lines.append("[hyperparameter_importance]")
        for key, value in importances.items():
            report_lines.append(f"{key}: {value:.6f}")

    report_lines.append("")
    report_lines.append("[convergence_check]")
    if not convergence_summary.get('enough_data', False):
        report_lines.append(convergence_summary['message'])
    else:
        report_lines.append(
            "late_stage_improvement: {:.2%}".format(convergence_summary['late_stage_improvement'])
        )
        report_lines.append(
            "first_80pct_best: {:.6f}".format(convergence_summary['first_80pct_best'])
        )
        report_lines.append(
            "last_20pct_best: {:.6f}".format(convergence_summary['last_20pct_best'])
        )
        report_lines.append(f"converged: {convergence_summary['converged']}")
        report_lines.append(f"recommendation: {convergence_summary['recommendation']}")

    (RUN_BASE_DIR / 'study_report.txt').write_text("\n".join(report_lines), encoding='utf-8')


def _create_study():
    # SQLite timeout cho phép 2 process cùng ghi DB mà không gặp "database is locked".
    storage = optuna.storages.RDBStorage(
        url=OPTUNA_PARAMS['storage'],
        engine_kwargs={
            'connect_args': {'timeout': 30},
        },
    )
    return optuna.create_study(
        study_name=OPTUNA_PARAMS['study_name'],
        direction='minimize',
        storage=storage,
        load_if_exists=True,
        sampler=_load_auto_sampler(seed=OPTUNA_PARAMS['seed']),
        pruner=MedianPruner(n_startup_trials=12, n_warmup_steps=20, interval_steps=5, n_min_trials=5),
    )


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    study = _create_study()
    try:
        study.optimize(
            objective,
            n_trials=OPTUNA_PARAMS['n_trials'],
            timeout=OPTUNA_PARAMS['timeout'],
            gc_after_trial=True,
            catch=(RuntimeError, ValueError, ImportError),
        )
    except KeyboardInterrupt:
        logger = logging.getLogger('root')
        logger.warning('Optuna run interrupted by keyboard. Current interrupted trial cleanup completed.')
        _write_study_reports(study)
        raise
    _write_study_reports(study)
    create_logger(**logger_params)

    logger = logging.getLogger('root')
    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if completed_trials:
        logger.info('Best trial value: %s', study.best_value)
        logger.info('Best trial params: %s', study.best_trial.params)
        logger.info('Hyperparameter importance: %s', get_param_importances(study))
        convergence_summary = _analyze_late_stage_improvement(study.trials)
        if convergence_summary.get('enough_data', False):
            logger.info(
                'Late-stage improvement: %.2f%% | Recommendation: %s',
                convergence_summary['late_stage_improvement'] * 100.0,
                convergence_summary['recommendation'],
            )
        else:
            logger.info('Convergence check skipped: %s', convergence_summary['message'])
    else:
        logger.info('No completed Optuna trials were produced.')


def _set_debug_mode():
    global BASE_TRAINER_PARAMS
    BASE_TRAINER_PARAMS = copy.deepcopy(BASE_TRAINER_PARAMS)
    BASE_TRAINER_PARAMS['epochs'] = 2
    BASE_TRAINER_PARAMS['train_episodes'] = 64
    BASE_TRAINER_PARAMS['validation']['episodes'] = 8
    BASE_TRAINER_PARAMS['validation']['batch_size'] = 8


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    logger.info('BASE_ENV_PARAMS: {}'.format(BASE_ENV_PARAMS))
    logger.info('BASE_MODEL_PARAMS: {}'.format(BASE_MODEL_PARAMS))
    logger.info('BASE_OPTIMIZER_PARAMS: {}'.format(BASE_OPTIMIZER_PARAMS))
    logger.info('BASE_TRAINER_PARAMS: {}'.format(BASE_TRAINER_PARAMS))
    logger.info('OPTUNA_PARAMS: {}'.format(OPTUNA_PARAMS))
    logger.info('RUN_TAG: {}'.format(RUN_TAG))


if __name__ == "__main__":
    main()