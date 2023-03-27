import os
from tqdm import tqdm

import torch

import data.common
from utils import interact, MultiSaver

import torch.cuda.amp as amp

class Trainer():

    def __init__(self, args, model, criterion, disp_criterion, optimizer, loaders):
        print('===> Initializing trainer')
        self.args = args
        self.mode = 'train' # 'val', 'test'
        self.epoch = args.start_epoch
        self.save_dir = args.save_dir

        self.model = model
        self.criterion = criterion
        self.disp_criterion = disp_criterion
        self.optimizer = optimizer
        self.loaders = loaders

        self.do_train = args.do_train
        self.do_validate = args.do_validate
        self.do_test = args.do_test

        self.device = args.device
        self.dtype = args.dtype
        self.dtype_eval = args.dtype

        if self.args.demo and self.args.demo_output_dir:
            self.result_dir = self.args.demo_output_dir
        else:
            self.result_dir = os.path.join(self.save_dir, 'result')
        os.makedirs(self.result_dir, exist_ok=True)
        print('results are saved in {}'.format(self.result_dir))

        self.imsaver = MultiSaver(self.result_dir)

        self.is_slave = self.args.launched and self.args.rank != 0

        self.scaler = amp.GradScaler(
            init_scale=self.args.init_scale,
            enabled=self.args.amp
        )

    def save(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        if epoch % self.args.save_every == 0:
            if self.mode == 'train':
                self.model.save(epoch)
                self.optimizer.save(epoch)
            self.criterion.save()
            self.disp_criterion.save()
        return

    def load(self, epoch=None, pretrained=None):
        if epoch is None:
            epoch = self.args.load_epoch
        self.epoch = epoch
        self.model.load(epoch, pretrained)
        self.optimizer.load(epoch)
        self.criterion.load(epoch)
        self.disp_criterion.load(epoch)

        return

    def train(self, epoch):
        self.mode = 'train'
        self.epoch = epoch


        self.model.train()
        self.model.to(dtype=self.dtype)

        self.criterion.train()
        self.criterion.epoch = epoch
        self.disp_criterion.train()
        self.disp_criterion.epoch = epoch

        if not self.is_slave:
            print('[Epoch {} / lr {:.2e}]'.format(
                epoch, self.optimizer.get_lr()
            ))

        if self.args.distributed:
            self.loaders[self.mode].sampler.set_epoch(epoch)
        if self.is_slave:
            tq = self.loaders[self.mode]
        else:
            tq = tqdm(self.loaders[self.mode], ncols=60, smoothing=0, bar_format='{desc}|{bar}{r_bar}')

        torch.set_grad_enabled(True)
        for idx, batch in enumerate(tq):
            self.optimizer.zero_grad()

            left_image, right_image, left_event, right_event, disparity_image = \
                data.common.to(batch[0]['left_image'], batch[0]['right_image'], batch[0]['representation']['left'], batch[0]['representation']['right'],\
                batch[0]['disparity_gt'], device=self.device, dtype=self.dtype)
            
            # left_image, right_image, left_event, right_event, left_noise, right_noise, disparity_image = \
            #     data.common.to(batch[0], batch[1], batch[5], batch[6], batch[7], batch[8], batch[9], device=self.device, dtype=self.dtype)
            

            left_input = [left_image, left_event]
            right_input = [right_image, right_event]

            input = [left_image, right_image, left_event, right_event]


            with amp.autocast(self.args.amp):
                
                ## recon ##
                # pred_l = self.model(left_input)
                # pred_r = self.model(right_input)
                
                # if self.args.model in ['pertu_select_recon']:
                #     loss1 = self.criterion(pred_l, left_image[0])
                #     loss2 = self.criterion(pred_r, right_image[0])
                #     loss = loss1 + loss2    
                # else:
                #     loss1 = self.criterion(pred_l, left_image)
                #     loss2 = self.criterion(pred_r, right_image)
                #     loss = loss1 + loss2

                if self.args.disp_model in ['gwc_pertu_noise', 'gwc_pertu_noise_deform']:
                    pred = self.model.disp_forward(input)

                    disp_pred = pred[:4]
                    left_recon = pred[4]
                    right_recon = pred[5]
                    
                    loss1 = self.disp_criterion(disp_pred, disparity_image)
                    loss2 = self.criterion(left_recon, left_image)
                    loss3 = self.criterion(right_recon, right_image)
                    loss = loss1 + loss2 + loss3
                
                elif self.args.disp_model in ['gwc_pertu_noise_KD']:
                    pred = self.model.disp_forward(input)

                    disp_pred = pred[:4]
                    left_recon = pred[4]
                    right_recon = pred[5]
                    KD_loss = pred[6]
                    
                    loss1 = self.disp_criterion(disp_pred, disparity_image)
                    loss2 = self.criterion(left_recon, left_image)
                    loss3 = self.criterion(right_recon, right_image)
                    loss = loss1 + loss2 + loss3 + KD_loss

                elif self.args.disp_model in ['aanet', 'gwc_event', 'gwc_image']:
                    disp_pred = self.model.disp_forward(input)
                    loss = self.disp_criterion(disp_pred, disparity_image)
                elif self.args.disp_model in ['gwc_pertu_noise_affinity']:
                    
                    pred = self.model.disp_forward(input)

                    disp_pred = pred[:3]
                    left_recon = pred[3]
                    right_recon = pred[4]
                    
                    # pair_loss = pred[5]
                    
                    loss1 = self.disp_criterion(disp_pred, disparity_image)
                    loss2 = self.criterion(left_recon, left_image)
                    loss3 = self.criterion(right_recon, right_image)
                    # loss = loss1 + loss2 + loss3 + 5 * pair_loss
                    loss = loss1 + loss2 + loss3
                elif self.args.disp_model in ['gwc_pertu_noise_with_affinity']:
                    
                    pred = self.model.disp_forward(input)
                    
                    disp_pred = pred[:4]
                    left_recon = pred[4]
                    right_recon = pred[5]
                    
                    # pair_loss = pred[5]
                    
                    loss1 = self.disp_criterion(disp_pred, disparity_image)
                    loss2 = self.criterion(left_recon, left_image)
                    loss3 = self.criterion(right_recon, right_image)
                    # loss = loss1 + loss2 + loss3 + 5 * pair_loss
                    loss = loss1 + loss2 + loss3
                elif self.args.disp_model in ['pasm_pertu_noise']:
                    
                    pred = self.model.disp_forward(input)

                    disp_pred = pred[:-1]
                    left_recon, right_recon = pred[-1]
                    
                    loss1 = self.disp_criterion(disp_pred, disparity_image)
                    loss2 = self.criterion(left_recon, left_image)
                    loss3 = self.criterion(right_recon, right_image)
                    loss = loss1 + 0.3 * (loss2 + loss3)
                

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer.G)
            self.scaler.update()

            if isinstance(tq, tqdm):
                # tq.set_description(self.criterion.get_loss_desc())
                tq.set_description(self.disp_criterion.get_loss_desc())

        self.criterion.normalize()
        self.disp_criterion.normalize()
        if isinstance(tq, tqdm):
            # tq.set_description(self.criterion.get_loss_desc())
            tq.set_description(self.disp_criterion.get_loss_desc())
            tq.display(pos=-1)  # overwrite with synchronized loss

        self.disp_criterion.step()
        self.criterion.step()
        self.optimizer.schedule(self.disp_criterion.get_last_loss())




        if self.args.rank == 0:
            self.save(epoch)

        return

    def evaluate(self, epoch, mode='val'):
        
        self.mode = mode
        self.epoch = epoch


        self.model.eval()
        self.model.to(dtype=self.dtype_eval)

        if mode == 'val':
            self.disp_criterion.validate()
        elif mode == 'test':
            self.disp_criterion.test()
        
        self.disp_criterion.epoch = epoch

        self.imsaver.join_background()


        if self.is_slave:
            tq = self.loaders[self.mode]
        else:
            tq = tqdm(self.loaders[self.mode], ncols=100, nrows=2, smoothing=0, bar_format='{desc}|{bar}{r_bar}')
        # import pdb
        # pdb.set_trace()
        compute_loss = True
        torch.set_grad_enabled(False)
        for idx, batch in enumerate(tq):

            left_image, right_image, left_event, right_event, disparity_image = \
                data.common.to(batch[0]['left_image'], batch[0]['right_image'], batch[0]['representation']['left'], batch[0]['representation']['right'],\
                batch[0]['disparity_gt'], device=self.device, dtype=self.dtype)
            
            left_input = [left_image, left_event]
            right_input = [right_image, right_event]

            input = [left_image, right_image, left_event, right_event]


            with amp.autocast(self.args.amp):
                
                # pred_l = self.model(left_input)
                # pred_r = self.model(right_input)
                disp_pred = self.model.disp_forward(input)


                # disp_pred = data.common.disp_pad(disp_pred, pad_width=batch[2], negative=True)




            # if mode == ('demo' or 'test'):  # remove padded part
            #     pad_width = batch[2]
            #     if self.args.model in ['pertu_select_recon']:
            #         pred_l, _ = data.common.pad(pred_l, pad_width=pad_width, negative=True)
            #         pred_r, _ = data.common.pad(pred_r, pad_width=pad_width, negative=True)
            #     else:
            #         pred_l[0], _ = data.common.pad(pred_l[0], pad_width=pad_width, negative=True)
                    # pred_r[0], _ = data.common.pad(pred_r[0], pad_width=pad_width, negative=True)

            # output = pred_l


            if compute_loss:
                # if self.args.model in ['pertu_select_recon']:
                #     self.criterion(pred_l, left_image[0])
                # else:
                #     self.criterion(pred_l, left_image)
                self.disp_criterion(disp_pred, disparity_image)

                if isinstance(tq, tqdm):
                    tq.set_description(self.disp_criterion.get_loss_desc())

            if self.args.save_results != 'none':
                # if isinstance(output, (list, tuple)):
                #     result = output[0]  # select last output in a pyramid
                # elif isinstance(output, torch.Tensor):
                #     result = output

                if isinstance(disp_pred, (list, tuple)):
                    result = disp_pred[0]  # select last output in a pyramid
                elif isinstance(disp_pred, torch.Tensor):
                    result = disp_pred

                names = batch[2]
                pred_disp_names = batch[3]
                gt_disp_names = batch[4]


                if self.args.save_results == 'part' and compute_loss: # save all when GT not available
                    indices = batch[5]
                    save_ids = [save_id for save_id, idx in enumerate(indices) if idx % 1 == 0]
                    
                    result = result[save_ids]
                    left_image = left_image[save_ids]
                    disparity_image = disparity_image[save_ids]

                    names = [names[save_id] for save_id in save_ids]
                    pred_disp_names = [pred_disp_names[save_id] for save_id in save_ids]
                    gt_disp_names = [gt_disp_names[save_id] for save_id in save_ids]
              
                self.imsaver.save_image(left_image, names)
                self.imsaver.save_disp(result, pred_disp_names)
                self.imsaver.save_disp(disparity_image, gt_disp_names)

        if compute_loss:
            self.disp_criterion.normalize()
            if isinstance(tq, tqdm):
                tq.set_description(self.disp_criterion.get_loss_desc())
                tq.display(pos=-1)  # overwrite with synchronized loss

            self.disp_criterion.step()
            if self.args.rank == 0:
                self.save()

        self.imsaver.end_background()

    def ptest(self, epoch, mode='test'):
        
        self.mode = mode
        self.epoch = epoch


        self.model.eval()
        self.model.to(dtype=self.dtype_eval)

        if mode == 'val':
            self.disp_criterion.validate()
        elif mode == 'test':
            self.disp_criterion.test()
        
        self.disp_criterion.epoch = epoch

        self.imsaver.join_background()


        if self.is_slave:
            tq = self.loaders[self.mode]
        else:
            tq = tqdm(self.loaders[self.mode], ncols=80, nrows=2, smoothing=0, bar_format='{desc}|{bar}{r_bar}')
        # import pdb
        # pdb.set_trace()
        compute_loss = True
        torch.set_grad_enabled(False)
        for idx, batch in enumerate(tq):

            left_image, right_image, left_event, right_event= \
                data.common.to(batch[0]['left_image'], batch[0]['right_image'], batch[0]['representation']['left'], batch[0]['representation']['right'],\
                 device=self.device, dtype=self.dtype)
            
            left_input = [left_image, left_event]
            right_input = [right_image, right_event]

            input = [left_image, right_image, left_event, right_event]

            

            with amp.autocast(self.args.amp):
                
                # pred_l = self.model(left_input)
                # pred_r = self.model(right_input)
                disp_pred = self.model.disp_forward(input)


                # disp_pred = data.common.disp_pad(disp_pred, pad_width=batch[2], negative=True)




            # if mode == ('demo' or 'test'):  # remove padded part
            #     pad_width = batch[2]
            #     if self.args.model in ['pertu_select_recon']:
            #         pred_l, _ = data.common.pad(pred_l, pad_width=pad_width, negative=True)
            #         pred_r, _ = data.common.pad(pred_r, pad_width=pad_width, negative=True)
            #     else:
            #         pred_l[0], _ = data.common.pad(pred_l[0], pad_width=pad_width, negative=True)
                    # pred_r[0], _ = data.common.pad(pred_r[0], pad_width=pad_width, negative=True)

            # output = pred_l


            # if compute_loss:
            #     # if self.args.model in ['pertu_select_recon']:
            #     #     self.criterion(pred_l, left_image[0])
            #     # else:
            #     #     self.criterion(pred_l, left_image)
            #     self.disp_criterion(disp_pred, disparity_image)

            #     if isinstance(tq, tqdm):
            #         tq.set_description(self.disp_criterion.get_loss_desc())

            if self.args.save_results != 'none':
                # if isinstance(output, (list, tuple)):
                #     result = output[0]  # select last output in a pyramid
                # elif isinstance(output, torch.Tensor):
                #     result = output

                if isinstance(disp_pred, (list, tuple)):
                    result = disp_pred[0]  # select last output in a pyramid
                elif isinstance(disp_pred, torch.Tensor):
                    result = disp_pred

                names = batch[2]
                pred_disp_names = batch[3]
                gt_disp_names = batch[4]

                if self.args.save_results == 'part' and compute_loss: # save all when GT not available
                    indices = batch[5]
                    # save_ids = [save_id for save_id, idx in enumerate(indices) if idx % 10 == 0]
                    
                    # result = result[save_ids]
                    # left_image = left_image[save_ids]
                    # # disparity_image = disparity_image[save_ids]

                    # names = [names[save_id] for save_id in save_ids]
                    # pred_disp_names = [pred_disp_names[save_id] for save_id in save_ids]
                    # gt_disp_names = [gt_disp_names[save_id] for save_id in save_ids]
              
                # import pdb; pdb.set_trace()
                # self.imsaver.save_image(left_image, names)
                # import pdb; pdb.set_trace()
                self.imsaver.save_disp_test(result, pred_disp_names)
                # self.imsaver.save_disp(result, pred_disp_names)
                # self.imsaver.save_disp(disparity_image, gt_disp_names)

        if compute_loss:
            self.disp_criterion.normalize()
            if isinstance(tq, tqdm):
                tq.set_description(self.disp_criterion.get_loss_desc())
                tq.display(pos=-1)  # overwrite with synchronized loss

            self.disp_criterion.step()
            if self.args.rank == 0:
                self.save()

        self.imsaver.end_background()

    def validate(self, epoch):
        self.evaluate(epoch, 'val')
        return

    def test(self, epoch):
        self.ptest(epoch, 'test')
        return

    def fill_evaluation(self, epoch, mode=None, force=False):
        if epoch <= 0:
            return

        if mode is not None:
            self.mode = mode

        do_eval = force
        if not force:
            loss_missing = epoch not in self.criterion.loss_stat[self.mode]['Total']    # should it switch to all loss types?

            metric_missing = False
            for metric_type in self.criterion.metric:
                if epoch not in self.criterion.metric_stat[mode][metric_type]:
                    metric_missing = True

            do_eval = loss_missing or metric_missing

        if do_eval:
            try:
                self.load(epoch)
                self.evaluate(epoch, self.mode)
            except:
                # print('saved model/optimizer at epoch {} not found!'.format(epoch))
                pass

        return
