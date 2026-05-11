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
import torch
import torch.distributed as dist

# Dùng đường dẫn tuyệt đối của file để ép Python ưu tiên module nội bộ của project.
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent

# Đưa project root lên đầu sys.path để tránh import nhầm package `utils` từ site-packages.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Giữ nguyên hành vi chạy bằng relative path cũ sau khi bootstrap import đã ổn định.
os.chdir(FILE_DIR)
from utils import create_logger
from FSMVRP_Trainer import FSMVRPTrainer_PPO as Trainer

##########################################################################################
# Parameters cho MAP 50 NODES
##########################################################################################

env_params = {
    'min_problem_size': 50,
    'max_problem_size': 50,
    'pomo_size': 50,
    'min_agent_num': 3,
    'max_agent_num': 6,
}

model_params = {
    'embedding_dim': 128,
    'encoder_layer_num': 6,
    'head_num': 8,
    'qkv_dim': 16,
    'ff_hidden_dim': 512,
    'logit_clipping': 10.0,
    'future_beta': 1.042624381788913,
    'eval_type': 'softmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 0.00012887318729861474,
        'weight_decay': 8.953933957212514e-09
    },
    'scheduler': {
        'milestones': [500, 1001, 1500],
        'gamma': 0.6917007536530753
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 500,
    'train_episodes': 10000,
    'train_batch_size': 128,
    'critic_hidden_dim': 256,
    'gradient_checkpointing': {  # <--- ĐẶT Ở NGOÀI LOGGING CHO ĐÚNG
        'enable': True
    },
    'ppo': {
        'epsilon': 0.16342857022805277,
        'ppo_epochs': 6,
        'gamma': 0.99,
        'lambda_future': 0.8285759385365766,
        'alpha_entropy': 0.0019130902106044754,
        'c_critic': 0.40809721256511683,
    },
    'validation': {
        'enable': True,
        'episodes': 400,
        'batch_size': 100,
        'seed': 9999,
    },
    'model_load': {
        'enable': False,
        'path': './results/train_50nodes',
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
        }
    }  # <--- Đóng của logging
}  # <--- Đóng của trainer_params

logger_params = {
    'log_file': {
        'filepath': './results/train_50nodes',
        'desc': 'train_50'
    }
}


##########################################################################################
# Main
##########################################################################################

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    # Khởi tạo thông số DDP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)

    # CHỐNG TRÙNG DỮ LIỆU: Seed phải khác nhau trên mỗi GPU
    seed = 1234 + local_rank
    torch.manual_seed(seed)

    # Đưa local_rank và world_size vào trainer_params
    trainer_params['local_rank'] = local_rank
    trainer_params['world_size'] = world_size

    # Ở cấu hình phân tán, batch_size được hiểu là khối lượng cho 1 GPU.
    # Nếu bạn muốn tổng = 400 thì mỗi GPU gánh 200. Hãy tự chỉnh trong params.

    # CHỈ GPU 0 mới được quyền tạo log và in config
    if local_rank == 0:
        create_logger(**logger_params)
        _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


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
