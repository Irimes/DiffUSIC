import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Diffusion.TrainSpectrumConditionalV2 import train_v2, eval_v2, compare_v2
from Diffusion.TrainBaselines import train_baselines


def main(model_config=None):
    modelConfig = {
        'state': 'compare',       
        'epoch': 500,           
        'batch_size': 100,       
        'T': 400,
        'beta_1': 1e-4,
        'beta_T': 0.02,
        'dropout': 0.1,
        'lr': 1e-4,
        'multiplier': 2.0,
        'grad_clip': 1.0,
        'device': 'cuda:0',
        'training_load_weight': None,
        'save_weight_dir': './CheckpointsSpectrumConditionalV2/',
        'test_load_weight': 'SNR-N-Best.pt',

        # 数据参数
        'M': 8,
        'npz_path': r"D:\Files\Academic_Resourses\Codes\diffusion again - 副本 - 副本 - 副本\DOA_Dataset\SNRStandardLowAngle.npz",
        'test_ratio': 1,      
        'split_seed': 42,
        'd_lambda': 0.5,

        # 谱网格参数
        'angle_min': -60.0,
        'angle_max': 60.0,
        'angle_step': 1.0,
        'spec_floor_db': -50.0,

        # 谱标签类型
        'spec_label_type': 'gaussian',
        'gaussian_sigma': 1.0,

        # 全分辨率网络参数
        'spec_base_ch': 192,            
        'spec_res_blocks': 2,           

        # SNR条件
        'use_snr_cond': True,
        'snr_range_min': -20.0,         
        'snr_range_max': 20.0,          

        # K条件
        'use_k_cond': True,
        'k_range_min': 1,               
        'k_range_max': 3,               

        # CFG
        'cfg_drop_prob': 0.1,           
        'cfg_scale': 2.0,               

        # SNR-weighted 损失
        'snr_loss_weight_max': 3.0,     

        # 条件编码器参数
        'tau': 8,                       
        'use_anti_rectifier': True,    

        # Peak 损失 
        'peak_loss_lambda': 1.0,       
        'peak_weight': 6.0,            
        'peak_neighborhood': 4,      

        'use_curriculum': True,        
        'curriculum_epochs': 250,       
        'curriculum_start_snr': 5.0,     

        # 测试参数
        'test_samples_per_snr': 1000,
        'save_every': 10,               
        'plot_worst_n': 16,

        # 对比
        'compare_random_seed': 42,
        'compare_plot_dir': './ComparePlotsV2',
        'compare_show_plot': False,

        # Baseline 参数 
        'baseline_angle_step': 1.0,
        'baseline_save_dir': './BaselineCheckpoints',
        'baseline_tau': 8,
        'baseline_epochs': 100,
        'baseline_batch_size': 100,
        'baseline_lr': 1e-4,
        'baseline_test_ratio': 0.1,
        'baseline_split_seed': 42,
        'baseline_models': [
            #'DeepSFNS', 'DeepSSE',
            #'IQResNet', 'DOALowSNRNet', 'daMUSIC',
        ],
    }

    if model_config is not None:
        modelConfig = model_config

    if modelConfig['state'] == 'train':
        train_v2(modelConfig)
    elif modelConfig['state'] == 'eval':
        eval_v2(modelConfig)
    elif modelConfig['state'] == 'compare':
        compare_v2(modelConfig)
    elif modelConfig['state'] == 'train_baselines':
        train_baselines(modelConfig)
    else:
        raise ValueError(f"未知 state: {modelConfig['state']}")


if __name__ == '__main__':
    main()