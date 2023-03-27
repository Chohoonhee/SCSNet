"""main file that does everything"""
from utils import interact

from option import args, setup, cleanup
from data import Data
from model import Model
from loss import Loss
from optim import Optimizer
from train import Trainer
import torch
from disp_loss import Loss as Disp_Loss


## train
# python main.py --n_GPUs 4 --batch_size 8 --dataset indoor_flying_1 --split 1 --data_root ../../DSEC_data --save_dir max_disp_120_homo_batch_8 --model pertu_select_recon --gaussian_pyramid False --loss 1*L1+1*LPIPS --lr 1e-4 --test_every 200 --save_every 1 --disp_model gwc_pertu_noise_with_affinity --end_epoch 160 --validate_every 10

## test
# CUDA_VISIBLE_DEVICES=3 python main.py --n_GPUs 1 --batch_size 1 --dataset indoor_flying_1 --split 1 --data_root ../../DSEC_data --save_dir max_disp_120_homo_batch_8 --model pertu_select_recon --gaussian_pyramid False --loss 1*L1+1*LPIPS --lr 1e-4 --test_every 100 --save_every 1 --disp_model gwc_pertu_noise_with_affinity --end_epoch 99 --validate_every 1 --load_epoch 76


def main_worker(rank, args):
    args.rank = rank
    args = setup(args)

    loaders = Data(args).get_loader()


    model = Model(args)
    model.parallelize()
    if args.load_dir is not None:
        checkpoint = torch.load(args.load_dir)
        # import pdb
        # pdb.set_trace()
        if model.load_state_dict(args.load_dir):
            print('load the checkpoint {}'.format(args.load_dir))
    
    optimizer = Optimizer(args, model)

    criterion = Loss(args, model=model, optimizer=optimizer)

    disp_criterion = Disp_Loss(args, model=model, optimizer=optimizer)

    trainer = Trainer(args, model, criterion, disp_criterion, optimizer, loaders)

    if args.stay:
        interact(local=locals())
        exit()

    if args.demo:
        trainer.evaluate(epoch=args.start_epoch, mode='demo')
        exit()

    # for epoch in range(1, args.start_epoch):
    #     if args.do_validate:
    #         if epoch % args.validate_every == 0:
    #             trainer.fill_evaluation(epoch, 'val')
    #     if args.do_test:
    #         if epoch % args.test_every == 0:
    #             trainer.fill_evaluation(epoch, 'test')

    for epoch in range(args.start_epoch, args.end_epoch+1):

        # epoch = 24

        # if args.do_test:
        #     # if epoch % args.test_every == 0:
        #         # if trainer.epoch != epoch:
        #     # trainer.load(epoch)
        #         trainer.test(epoch)
        # import pdb; pdb.set_trace()


        # if args.do_validate:
        #     if epoch % args.validate_every == 0:
        #         if trainer.epoch != epoch:
        #             trainer.load(epoch)
        # trainer.validate(epoch)
        # import pdb; pdb.set_trace()

        if args.do_train:
            trainer.train(epoch)
        # import pdb; pdb.set_trace()
        if args.do_validate:
            if epoch % args.validate_every == 0:
                if trainer.epoch != epoch:
                    trainer.load(epoch)
                trainer.validate(epoch)

        if args.do_test:
            if epoch % args.test_every == 0:
                if trainer.epoch != epoch:
                    trainer.load(epoch)
                trainer.test(epoch)

        if args.rank == 0 or not args.launched:
            print('')

    trainer.imsaver.join_background()

    cleanup(args)

def main():
    main_worker(args.rank, args)

if __name__ == "__main__":
    main()