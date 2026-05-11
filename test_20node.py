##########################################################################################
# Machine Environment Config
##########################################################################################
import torch
import os
import logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))

USE_CUDA=torch.cuda.is_available()
CUDA_DEVICE_NUM = 0

from utils import create_logger
from FSMVRP_Tester import FSMVRPTester_PPO as Tester
    # Environment parameters
env_params = {
    'min_problem_size': 20,
    'max_problem_size': 20,
    'pomo_size': 1000,
    'min_agent_num': 3,
    'max_agent_num': 6,
    'utilization_penalty': {
        'enable': False,
    },
}

model_params = {
    'embedding_dim': 128,
    'encoder_layer_num': 6,
    'head_num': 8,
    'qkv_dim': 16,
    'ff_hidden_dim': 512,
    'logit_clipping': 10.0,
    'future_beta': 1.0,
    'eval_type': 'softmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'critic_hidden_dim': 256,

    'model_load': {
        'path': './results/train_20nodes',  # Folder chứa checkpoint-{epoch}.pt
        'epoch': "best",
    },

    # data load
    'test_data_load': {
        'enable': True,  # Bật True nếu m đã đẻ sẵn file .pt để test đối chiếu Gurobi
        'filename': ".\data\test_tensor(20)_6_100_1234.pt"  # File .pt đã đẻ sẵn để test đối chiếu Gurobi,
    },
    'test_random_seed': 1234,  # Seed cố định để sinh 100 map giống hệt nhau ở mọi lần chạy
    'test_episodes': 1,  # Cho thi 100 bài
    'test_batch_size': 1,  # Test thì chạy từng batch 1 là chuẩn

    #  data augmentation (Xoay bản đồ 8 hướng)
    'augmentation_enable': True,
    'aug_factor': 8,
    'solution_detail': {
        'enable': True,
        'max_episodes': 100,
        'compare_augmented_best': True,
        'skip_empty_routes': True,
    },
    'csv_export': {
        'enable': True,
        'summary_path': './result/test_20nodes/solution_summary_demo.csv',
        'routes_path': './result/test_20nodes/solution_routes_demo.csv',
    },
}

# Tạo riêng một thư mục log kết quả test để không đụng chạm log train
logger_params = {
    'log_file': {
        'filepath': './result/test_20nodes',
        'desc': 'test_20'
    }
}


##########################################################################################
# Main
##########################################################################################

def main():
    # 1. Bật công cụ ghi chép
    create_logger(**logger_params)
    _print_config()

    # 2. Nạp đạn cho Tester
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    # 3. Kéo còi báo danh!
    no_aug_score, aug_score = tester.run()

    print("\n" + "=" * 50)
    print(f"TỔNG KẾT KẾT QUẢ TEST 20 NODES:")
    print(f"   - Chi phí (Chưa Augment) : {no_aug_score:.4f}")
    print(f"   - Chi phí (Có Augment): {aug_score:.4f} ")
    print("=" * 50 + "\n")


def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + ": {}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()
