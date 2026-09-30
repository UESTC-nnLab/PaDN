import os
import torch
from tqdm import tqdm
from utils.utils import get_lr
import numpy as np
import torch.nn as nn

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    epoch_step = epoch_step // 5 
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        
        #for name, param in model.named_parameters():
                #for frozen_prefix in frozen_parameters:
                    #if frozen_prefix in name:
                        #print(name)
        #    param.requires_grad = False
        #for param in model.experts[1].parameters():
        #    param.requires_grad = True    

        #for module in model.children():
        #    if not hasattr(module,'experts.1'):
        #        module.eval()
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        modules_to_train = [
            model.head,
            #model.motion,
            model.motion.domain_heads,
            model.fusion.lora_pre,
            model.fusion.dynamic_dilate,
            #model.fusion.ca
        ]

        # Helper function to check if a module is a submodule of any module in modules_to_train
        def is_submodule_of(module, modules_list):
            for m in modules_list:
                # Check if the module is the same as or a submodule of m
                for sub_m in m.modules():
                    if module is sub_m:
                        return True
            return False

        # Freeze BN layers except those in modules_to_train
        for module in model.modules():
            if isinstance(module, bn_types):
                # Only freeze BN layers that are NOT in modules_to_train
                if not is_submodule_of(module, modules_to_train):
                    #print(f"Freezing BN layer: {module}")
                    module.eval()  

        images, targets = batch[0], batch[1]
        #print(captions[0].shape,images.shape)

        if cuda:
            images  = images.cuda(local_rank)
            targets = [ann.cuda(local_rank) for ann in targets]

        optimizer.zero_grad()
        if not fp16:   
            #print(captions.shape)
            model.backbone.eval()
            model.head.eval()            
            outputs, motion_loss = model_train(images)
            loss_value = yolo_loss(outputs, targets) +  motion_loss

            loss_value.backward()
            #print("\nGradients after backward:")
            #for name, param in model.named_parameters():
                #if param.grad is not None:
                  #  grad_norm = param.grad.norm().item()
                 #   print(f"{name}: gradient norm = {grad_norm:.6f}")
                #else:
                    #print(f"{name}: no gradient")
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images) 
                loss_value = yolo_loss(outputs, targets)

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

            optimizer.zero_grad()
            outputs,_ = model_train_eval(images)
            
            loss_value = yolo_loss(outputs, targets)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
