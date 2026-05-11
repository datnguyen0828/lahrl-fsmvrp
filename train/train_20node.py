##########################################################################################
# Machine Environment Config
##########################################################################################
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config (Chuẩn bài nghiên cứu, gọi từ thư mục nào cũng chạy được)
##########################################################################################
import copy
import json
import os
import logging
import sys
from datetime import datetime
from pathlib import Path

# Dùng đường dẫn tuyệt đối của file để ép Python ưu tiên module nội bộ của project.
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent

# Đưa project root lên đầu sys.path để tránh import nhầm package `utils` từ site-packages.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Giữ nguyên hành vi chạy bằng relative path cũ sau khi bootstrap import đã ổn định.
os.chdir(FILE_DIR)

# Kéo công cụ và class Trainer của m vào
from utils import create_logger
from FSMVRP_Trainer import FSMVRPTrainer_PPO as Trainer

##########################################################################################
# Parameters cho MAP 20 NODES
##########################################################################################

env_params = {
    'min_problem_size': 20,
    'max_problem_size': 20,
    'pomo_size': 20,
    'min_agent_num': 3,
    'max_agent_num': 6,
    'utilization_penalty': {
        'enable': False,
        'ratio_threshold': 0.8,
        'weight': 3.0,
        'power': 2.0,
        'min_demand': 0.0,
    },
}

model_params = {
    'embedding_dim': 128,
    'encoder_layer_num': 6,
    'head_num': 8,
    'qkv_dim': 16,
    'ff_hidden_dim': 512,
    'logit_clipping': 10.0,
    'future_beta': 0.0,
    'eval_type': 'softmax',  # Train luôn luôn dùng softmax
}

optimizer_params = {
    'optimizer': {
        'lr': 0.001,             # Optuna Best Trial
        'weight_decay': 2.6922887226696767e-08   # Optuna Best Trial
    },
    'scheduler': {
        'milestones': [150, 300, 400],
        'gamma': 0.4211052570538756              # Optuna Best Trial
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 500,
    'train_episodes': 10000,
    'train_batch_size': 128,
    'critic_hidden_dim': 256,                    # Optuna Best Trial

    # Cấu hình trái tim PPO của m
    'ppo': {
        'epsilon': 0.23372320650686193,          # Optuna Best Trial
        'ppo_epochs': 6,                         # Optuna Best Trial
        'gamma': 0.99,
        'lambda_future': 0.42233036906684057,    # Optuna Best Trial
        'alpha_entropy': 0.05,  # Optuna Best Trial
        'c_critic': 0.994933978112219,           # Optuna Best Trial
    },

    # [FIX #9d] Cấu hình Validation Set
    'validation': {
        'enable': True,
        'episodes': 400,
        'batch_size': 100,
        'seed': 9999,
        'aug_factor': 8,
        'primary_eval_type': 'softmax',
        'secondary_eval_type': 'argmax',
        'objective_metric': 'val_softmax_aug_raw_score',
    },
    'annealing': {
        'enable': True,
        'start_epoch': 75,
        'final_alpha_entropy': 5e-4,
        'final_ppo_epsilon': 0.12,
    },

    'model_load': {
        'enable': False,
        'path': './results/train_20nodes',
        'epoch': 0,
    },

    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 5,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_train_score.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_train_loss.json'
        },
    }
}

# Đổi tên thư mục log để không bị lẫn lộn giữa 20 nodes và 50 nodes
logger_params = {
    'log_file': {
        'filepath': './results/train_20nodes',
        'desc': 'train_20'
    }
}


# Giữ một biến global để mọi helper dùng chung cùng một thư mục run hiện tại.
RUN_OUTPUT_DIR = None


def _get_shared_output_root():
    # Dùng Path tuyệt đối để tránh lệch thư mục khi script được gọi từ nhiều nơi khác nhau.
    return Path(__file__).resolve().parents[1] / 'results' / 'train_20nodes'


def _next_run_dir(output_root):
    # Quét các thư mục run_* có sẵn để sinh run id tăng dần và không đè artifact cũ.
    max_index = 0
    for child in output_root.glob('run_*'):
        if not child.is_dir():
            continue
        try:
            max_index = max(max_index, int(child.name.split('_')[-1]))
        except ValueError:
            continue

    run_dir = output_root / f'run_{max_index + 1:04d}'
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _prepare_run_output():
    global RUN_OUTPUT_DIR

    # Tạo root chung nếu chưa có để toàn bộ run 20 nodes nằm dưới cùng một cây thư mục.
    output_root = _get_shared_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    # Tạo thư mục riêng cho lần train hiện tại rồi trỏ logger vào đó.
    RUN_OUTPUT_DIR = _next_run_dir(output_root)
    logger_params['log_file']['filepath'] = str(RUN_OUTPUT_DIR)

    # Ghi lại run mới nhất ở root chung để người dùng vẫn có một điểm tra cứu nhanh.
    latest_run_file = output_root / 'latest_run.txt'
    latest_run_file.write_text(f"{RUN_OUTPUT_DIR.name}\n", encoding='utf-8')


def _resolve_model_load_path():
    # Khi resume từ root dùng chung, tự động ánh xạ sang run mới nhất để khỏi trỏ nhầm vào thư mục container.
    model_load_cfg = trainer_params.get('model_load', {})
    if not model_load_cfg.get('enable', False):
        return

    configured_path = Path(model_load_cfg.get('path', ''))
    if not configured_path:
        return

    # Chuẩn hóa sang path tuyệt đối dựa trên thư mục của script để resume ổn định hơn.
    if not configured_path.is_absolute():
        configured_path = (Path(__file__).resolve().parent / configured_path).resolve()

    shared_output_root = _get_shared_output_root().resolve()
    if configured_path != shared_output_root:
        trainer_params['model_load']['path'] = str(configured_path)
        return

    latest_run_file = shared_output_root / 'latest_run.txt'
    if not latest_run_file.is_file():
        raise FileNotFoundError(
            f"Cannot resolve latest train run because '{latest_run_file}' does not exist."
        )

    latest_run_name = latest_run_file.read_text(encoding='utf-8').strip()
    if not latest_run_name:
        raise ValueError(f"'{latest_run_file}' is empty, so model_load.path cannot be resolved.")

    resolved_run_dir = shared_output_root / latest_run_name
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(
            f"Resolved latest run directory '{resolved_run_dir}' does not exist."
        )

    trainer_params['model_load']['path'] = str(resolved_run_dir)


def _build_config_snapshot():
    # Sao chép sâu config để file snapshot không bị thay đổi ngoài ý muốn trong lúc train chạy.
    return {
        'run_id': RUN_OUTPUT_DIR.name,
        'script_name': Path(__file__).name,
        'created_at': datetime.now().astimezone().isoformat(),
        'output_dir': str(RUN_OUTPUT_DIR),
        'env_params': copy.deepcopy(env_params),
        'model_params': copy.deepcopy(model_params),
        'optimizer_params': copy.deepcopy(optimizer_params),
        'trainer_params': copy.deepcopy(trainer_params),
        'logger_params': copy.deepcopy(logger_params),
    }


def _write_json_file(path, payload):
    # Chuẩn hóa JSON UTF-8 để config và summary dễ đọc lại bằng tay lẫn script.
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _save_run_config():
    # Lưu đầy đủ config vào thư mục run để mỗi checkpoint luôn đi kèm đúng cấu hình tạo ra nó.
    _write_json_file(RUN_OUTPUT_DIR / 'config.json', _build_config_snapshot())


def _save_run_summary(summary_payload):
    # Ghi summary cuối run để tra cứu nhanh metric tốt nhất mà không cần mở log.txt.
    summary = {
        'run_id': RUN_OUTPUT_DIR.name,
        'script_name': Path(__file__).name,
        'completed_at': datetime.now().astimezone().isoformat(),
        'output_dir': str(RUN_OUTPUT_DIR),
        'metrics': summary_payload,
    }
    _write_json_file(RUN_OUTPUT_DIR / 'summary.json', summary)

##########################################################################################
# Main
##########################################################################################

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    # Resolve checkpoint path trước khi khởi tạo trainer để resume từ run mới nhất hoạt động với layout mới.
    _resolve_model_load_path()

    # Chuẩn bị thư mục run trước khi tạo logger để trainer tự động ghi toàn bộ artifact đúng chỗ.
    _prepare_run_output()
    _save_run_config()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)
    # Lưu summary cuối run để complement cho log.txt và checkpoint trong cùng thư mục.
    run_summary = trainer.run()
    _save_run_summary(run_summary)


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + ": {}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()
