--config mpn_o_config.yaml --save 'mpn_r'_191r_test  # our mpn  pms+ce

# test
python main.py --test_only --config ./mpn_o_config.yaml --pre_train ./experiment/\'mpn_o\'_191r_mss/model-best.pth

# train
python main.py --config ./mpn_o_config.yaml


# ablation study
# 1 magrin (1-10)
# 2 mulitiple branch