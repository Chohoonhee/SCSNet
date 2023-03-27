# Selection and Cross Similarity for Event-Image Deep Stereo (SCSNet) ECCV 2022

This code is an official code of our ECCV paper "Selection and Cross Similarity for Event-Image Deep Stereo"

# Dataset
DSEC
https://dsec.ifi.uzh.ch/uzh/disparity-benchmark/

# Train
python main.py --n_GPUs 4 --batch_size 8 --dataset indoor_flying_1 --split 1 --data_root ../../DSEC_data --save_dir max_disp_120_homo_batch_8 --model pertu_select_recon --loss 1*L1+1*LPIPS --lr 1e-4 --test_every 200 --save_every 1 --disp_model gwc_pertu_noise_with_affinity --end_epoch 160 --validate_every 10

# Inference for benchmark
CUDA_VISIBLE_DEVICES=3 python main.py --n_GPUs 1 --batch_size 1 --split 1 --data_root ../../DSEC_data --save_dir max_disp_120_homo_batch_8 --model pertu_select_recon --loss 1*L1+1*LPIPS --lr 1e-4 --test_every 100 --save_every 1 --disp_model gwc_pertu_noise_with_affinity --end_epoch 99 --validate_every 1 --load_epoch 77


# Paper Reference
@inproceedings{cho2022selection,
  title={Selection and Cross Similarity for Event-Image Deep Stereo},
  author={Cho, Hoonhee and Yoon, Kuk-Jin},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXII},
  pages={470--486},
  year={2022},
  organization={Springer}
}

we borrow the works from three repositories. Thanks for the excellent codes!
- Nah, Seungjun, Tae Hyun Kim, and Kyoung Mu Lee. "Deep multi-scale convolutional neural network for dynamic scene deblurring." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. 
  https://github.com/SeungjunNah/DeepDeblur_release
- Berthet, Quentin, et al. "Learning with differentiable pertubed optimizers." Advances in neural information processing systems 33 (2020): 9508-9519.
  https://github.com/tuero/perturbations-differential-pytorch

# TBD
- Pretrained model
- MVSEC dataloader
- There may be minor code errors due to accidental deletion of parts. But, performance was confirmed to be reproducible.
