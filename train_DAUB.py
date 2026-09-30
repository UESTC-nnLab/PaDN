import datetime
import os,random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#import pycocotools.coco as coco
from nets.GMM.PaDN2 import PaDN as Model
#from nets.ablation.seventh.PaDN1 import TaDAN as Model
from nets.training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
#from utils.dataloader import YoloDataset
from utils.dataloader import seqDataset, dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit_new import fit_one_epoch
from typing import Dict, Any, Set, Tuple
def show_trainable_groups(model: nn.Module, optimizer: optim.Optimizer, top_k: int = 40):
    # 1. 找出所有在优化器中且可训练的参数的ID
    # 注意：一个参数可能出现在多个param_groups中，但这通常不常见，这里使用set去重是正确的。
    in_optim: Set[int] = {id(p) for g in optimizer.param_groups for p in g.get('params', []) if p is not None}
    
    group_sum: Dict[str, int] = {}
    rows: list[Tuple[str, int]] = []
    
    # 2. 遍历模型的命名参数
    for name, p in model.named_parameters():
        # 确保参数在优化器中 (即可训练)
        if id(p) in in_optim:
            # 提取根模块名称 (如 backbone, adapter, heads)
            root = name.split('.')[0]
            
            num_params = p.numel()
            group_sum[root] = group_sum.get(root, 0) + num_params
            rows.append((name, num_params))

    # 3. 计算所有可训练参数的总和
    total_trainable_params: int = sum(group_sum.values())
    
    # 4. 打印按组划分的可训练参数量
    print("[Trainables by group]")
    for k, v in sorted(group_sum.items(), key=lambda kv: kv[0]):
        print(f"  {k:12s}: {v/1e6:.3f}M")
        
    # 5. 打印所有可训练参数的总和
    print("-" * 27)
    print(f"  {'Total':12s}: {total_trainable_params/1e6:.3f}M")
    print("-" * 27)

    # 6. 打印参数列表 (如果参数数量不多于 top_k) - 保持原有功能
    if len(rows) > top_k:
        print(f"  ... and {len(rows)-top_k} more groups/parameters not listed above.")

if __name__ == "__main__":
    
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    classes_path    = 'model_data/classes.txt'
    model_path      = '/home/zsc/TaDAN/logs/online/unknown/ITSDT5/best_epoch_weights.pth'
    input_shape     = [512, 512]
    phi             = 's'
    mosaic              = False
    mosaic_prob         = 0.5
    mixup               = False
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 2
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    Freeze_Train        = False
    frozen_parameters =['backbone','head','motion']
    Init_lr             = 1e-2 
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd" 
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 1
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 100
    num_workers         = 4
    #train_annotation_path = '/home/zsc/MIFL/fewshot/ITSDT/16_4.txt'
    #val_annotation_path = '/home/zsc/MIFL/fewshot/ITSDT/val_40.txt'
    train_annotation_path = '/home/zsc/TaDAN/embeddings/DAUB/txt/16_4.txt'
    val_annotation_path = '/home/zsc/TaDAN/embeddings/DAUB/txt/val_1.txt'   
    ngpus_per_node  = torch.cuda.device_count()
    
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0
        
    seed = 2023
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
        
    class_names, num_classes = get_classes(classes_path)
    
    model = Model(num_classes=1,  num_frame=2) 
    weights_init(model)
    if model_path != '':
        
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
       
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        model.initialize_new_task_modules(current_task_id = 2)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，部分参数没有载入是正常现象，这里只使用了预训练模型的部分参数权重。\033[0m")

    yolo_loss    = YOLOLoss(num_classes, fp16, strides=[8])
   
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],find_unused_parameters=True)
        else:
            model_train = model.cuda()
            cudnn.benchmark = True

    ema = ModelEMA(model_train)
    
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

     
    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = log_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small for training. Please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size above %d. \033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total training data amount of this run is %d, the Unfreeze_batch_size is %d, a total of %d epochs are trained, and the total training step size is %d. \033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training step size is %d, which is less than the recommended total step size %d, it is recommended to set the total epoch to %d. \033[0m"%(total_step, wanted_step, wanted_epoch))

    
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for name, param in model.named_parameters():
                # 1. 默认设置为可训练 (True)
                param.requires_grad = True 
                
                # 2. 检查是否需要冻结
                for frozen_prefix in frozen_parameters:
                    if frozen_prefix in name:
                        #print(name)
                        param.requires_grad = False # 找到匹配，冻结
                        break                       # 立即跳出内部循环，进入下一个参数  

        #for name, param in model.named_parameters():
            #if param.requires_grad == False:
                #print(name)
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []  
        
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   

            
        params_to_train = [
            {'params': model.head[2].parameters()},
            #{'params': model.motion.text_fc1.parameters()},
            #{'params': model.task_disc.task_specific_layers[str(1)].parameters()}, 
            #{'params': model.motion.text_fc_domain.parameters()},
            #{'params': model.motion.visual_fc_invariant.parameters()},
            #{'params': model.motion.domain_heads[2].parameters()},
            {'params': model.fusion.lora_pre[2].parameters()},
            {'params': model.fusion.dynamic_dilate[2].parameters()},
            #{'params': model.fusion.ca.parameters()},  # 假设CoordinateAttention层暂时不参与训练
        ]


        optimizer = optim.AdamW(
            params_to_train,
            lr=Init_lr_fit,
            betas=(0.9, 0.999),  # AdamW的标准beta值，通常无需修改
            eps=1e-8,            # 防止除以零，通常无需修改
            weight_decay=weight_decay
        )
        #optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        #optimizer.add_param_group({"params": pg2})
        
        #for name, param in model.named_parameters():
         #   print(f"{name}: requires_grad = {param.requires_grad}")
        show_trainable_groups(model, optimizer, top_k=40)

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset. ")
        
        if ema:
            ema.updates     = epoch_step * Init_Epoch
        
        train_dataset = seqDataset(train_annotation_path, input_shape[0], 2, 'train')
        val_dataset = seqDataset(val_annotation_path, input_shape[0], 2, 'val')

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True 

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)
        
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                    
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
               
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node
                    
                if ema:
                    ema.updates     = epoch_step * epoch
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, log_dir, local_rank)
                        
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()


