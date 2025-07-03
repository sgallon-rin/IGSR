import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import average_precision_score

import torch.optim as optim
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist
import torch.nn.functional as F
import sys, os
from params import is_DDIM, is_att, is_matrix
from cn_clip.training.params import parse_args

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parse_args()
from cn_clip.clip.new_model import convert_state_dict
from new_data import get_all_features
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt


def is_master(args):
    return args.rank == 0

intent2id = {}
id2intent = {}
is_intent_loss = False

def get_loss(model,classifier_model, images, session_ids, image_ids,
             summarys, raw_texts, fined_intents,loss_img, loss_txt,
             args, accum_image_features=None, accum_text_features=None,
             accum_idx=-1, teacher_model=None, teacher_accum_image_features=None):
    intent_features,flag = model(1, images, text=raw_texts, history=summarys,
                            mask_ratio=args.mask_ratio)
    intent_prob, predict_label = classifier_model(intent_features,flag)
    predict_label = predict_label.tolist()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    intent_ids = fined_intents
    intent_ids =torch.tensor(np.array(intent_ids, dtype=int))
    intent_loss = criterion(intent_prob.cpu(), intent_ids.cpu())

    image_features, text_features, logit_scale = model(0, images, text=raw_texts, history=summarys,
                                                       intent_features=intent_features,
                                                       mask_ratio=args.mask_ratio)


    logit_scale = logit_scale.mean()

    if args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if args.gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)

            if args.distllation:
                all_teacher_image_features = torch.cat(torch.distributed.nn.all_gather(teacher_image_features), dim=0)
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]

            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)

            all_image_features = torch.cat(
                [image_features]
                + gathered_image_features[:rank]
                + gathered_image_features[rank + 1:]
            )
            all_text_features = torch.cat(
                [text_features]
                + gathered_text_features[:rank]
                + gathered_text_features[rank + 1:]
            )

        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        if args.distllation:
            gathered_teacher_image_features = [
                torch.zeros_like(teacher_image_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_teacher_image_features, teacher_image_features)
            all_teacher_image_features = torch.cat(
                [teacher_image_features]
                + gathered_teacher_image_features[:rank]
                + gathered_teacher_image_features[rank + 1:]
            )
            kd_loss = cosineSimilarityLoss(all_teacher_image_features, all_image_features)

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        if args.distllation:
            kd_loss = cosineSimilarityLoss(teacher_image_features, image_features)

    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)
    if is_intent_loss:
        total_loss = (intent_loss +
                      loss_img(logits_per_image, ground_truth)
                      + loss_txt(logits_per_text, ground_truth)
                      ) / 3
    else:
        total_loss = (
                      loss_img(logits_per_image, ground_truth)
                      + loss_txt(logits_per_text, ground_truth)
                      ) / 2


    acc = None
    intent_acc = 0

    for j in range(len(predict_label)):
        if predict_label[j] == intent_ids[j]:
            intent_acc += 1

    if args.report_training_batch_acc:
        i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
        intent_acc = intent_acc / len(predict_label)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc, 'intent_acc': intent_acc}


    return total_loss, acc


def freeze_vision_bn(args, model):
    if 'RN' in args.vision_model:
        RN_visual_modules = model.module.visual.modules() if isinstance(model,
                                                                        nn.parallel.DistributedDataParallel) else model.visual.modules()
        for m in RN_visual_modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def train(model, classifier_model, attn_model, data, memory_dict, speaker_list, epoch, optimizer, scaler, scheduler,
          args, global_trained_steps, teacher_model=None):

    print('train')
    model.train()
    classifier_model.train()
    if args.freeze_vision:
        freeze_vision_bn(args, model)

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    print('dataloader')
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)
    print('loss')
    if sampler is not None:
        sampler.set_epoch(epoch)

    num_steps_per_epoch = dataloader.num_batches // args.accum_freq
    data_iter = iter(dataloader)
    all_data_iter = iter(dataloader)

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
        if args.distllation:
            teacher_accum_image_features = []

    end = time.time()
    epoch_trained_steps = 0
    print('num:',dataloader.num_batches)
    for i in range(0, dataloader.num_batches):
        batch = next(data_iter)
        i_accum = i // args.accum_freq
        step = num_steps_per_epoch * epoch + i_accum
        if step >= args.max_steps:
            logging.info("Stopping training due to step {} has reached max_steps {}".format(step,
                                                                                            args.max_steps // args.accum_freq))
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        session_ids, image_ids, images, summarys, raw_texts, fined_intents, \
        all_image_features,imgid2intent, origin_text = batch

        images = images.cuda(args.local_device_rank, non_blocking=True)
        data_time = time.time() - end

        m = model.module

        if args.accum_freq == 1:
            if args.precision == "amp":
                with autocast():
                    total_loss, acc = get_loss( model, classifier_model, images,session_ids,
                                                image_ids,summarys, raw_texts,fined_intents,
                                                loss_img, loss_txt,args)

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                scaler.update()

            else:
                if args.distllation:
                    total_loss, acc = get_loss(model, images, texts, sent_texts, loss_img, loss_txt, args,
                                               teacher_model=teacher_model)
                else:
                    total_loss, acc = get_loss(model, images, texts, sent_texts, loss_img, loss_txt, args)
                total_loss.backward()
                optimizer.step()

        torch.cuda.empty_cache()

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            batch_size = len(images) * args.accum_freq
            num_samples = (i_accum + 1) * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i_accum + 1) / num_steps_per_epoch

            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {total_loss.item():.6f} | " +
                (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"Intent Acc: {acc['intent_acc'] * 100:.2f} | " if args.report_training_batch_acc else "") +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"logit_scale: {m.logit_scale.data:.3f} | " +
                f"Global Batch Size: {batch_size * args.world_size}"
            )

        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(
                        model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
                                                                                             step + 1,
                                                                                             time.time() - t1))

            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(
                        model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
                                                                                             step + 1,
                                                                                             time.time() - t1))

            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"classifier_epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": classifier_model.state_dict() if not args.use_flash_attention else convert_state_dict(
                        classifier_model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
                                                                                             step + 1,
                                                                                             time.time() - t1))


    assert "val" in data, "Error: Valid dataset has not been built."
    print('evaluation')
    if not args.use_flash_attention:
        evaluate(model, data, epoch, args,classifier_model)
    else:
        # fp16 is needed in flash attention
        with autocast():
            evaluate(model, data, epoch, args, classifier_model)
    # set model back to train mode
    model.train()
    if args.freeze_vision:
        freeze_vision_bn(args, model)

    return epoch_trained_steps


def get_top_N_images(query, data_list, top_K=4):
    sim_list = []
    for img in data_list:
        similarity = cosine_similarity(query.cpu().detach().numpy(), img.cpu().detach().numpy())
        sim_list.append(similarity)

    sorted_indices = sorted(enumerate(sim_list), key=lambda x: x[1], reverse=True)

    top_five_indices = [index for index, value in sorted_indices[:top_K]]

    return top_five_indices


def get_images_similarity(query, data_list, top_K=4):
    sim_list = []
    for img in data_list:
        similarity = cosine_similarity(query.cpu().detach().numpy(), img.cpu().detach().numpy())
        sim_list.append(similarity)

    return sim_list


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    similarity = dot_product / (norm_vector1 * norm_vector2)

    return similarity


def metric_intent_label(json_file):
    with open('/home/sgallon/research/sticker-conv/IGSR/Dataset/' +
              args.data_mode + '/' + args.data_mode + '_response.json', 'r',
              encoding='utf-8') as f:
        datas = json.load(f)
    img_list = []
    img_dict = {}
    label_list = []
    for data in datas:
        if 'split' not in data and data['msgtype'] == 'sticker':
            label = data['fined_intent'].lower()
            if label not in label_list:
                label_list.append(label)
            img_name = data["type_info"]
            if '\\' in img_name:
                img_name = img_name.split('\\')[-1].split('.')[0].split('_')[-1]
            elif '/' in img_name:
                img_name = img_name.split('/')[-1].split('.')[0].split('_')[-1]
            else:
                exit()
            if img_name not in img_list:
                img_list.append(img_name)
                img_dict[img_name] = []

            img_dict[img_name].append(label)

    new_img = {}
    for item, value in img_dict.items():
        new_value = []
        for v in value:
            new_value.append(label_list.index(v))
        new_img[item] = new_value

    with open(json_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    pred_list = []
    real_list = []
    for data in datas:
        image_id = data['imageid']
        text_id = data['textid']
        raw_text = data['raw_text']
        target_image = data['target_image']

        target_list = img_dict[target_image.split('_')[1]]
        flag = 1
        for i in range(len(image_id)):
            if img_dict[image_id[i].split('_')[-1]] in target_list:  # 这应该是个list
                pred_list.append(target_list[0])
                real_list.append(target_list[0])
                flag = 0
                break
        if flag:
            pred_list.append(img_dict[image_id[0].split('_')[-1]][0])
            real_list.append(target_list[0])


    accuracy = accuracy_score(real_list, pred_list)
    precision = precision_score(real_list, pred_list, average='weighted')
    f1 = f1_score(real_list, pred_list, average='weighted')
    return accuracy, precision, f1


def metric_one_by_one(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    acc = 0
    pred_list = []
    real_list = []
    for data in datas:
        image_id = data['imageid']
        text_id = data['textid']
        raw_text = data['raw_text']
        target_image = data['target_image']
        flag = 1
        for i in range(len(image_id)):
            if image_id[i] in target_image[i]:
                pred_list.append(str(id.split('_')[1]))
                flag = 0
                break

        if flag:
            pred_list.append(str(image_id[0].split('_')[1]))

        real_list.append(str(target_image.split('_')[1]))

    accuracy = accuracy_score(real_list, pred_list)
    precision = precision_score(real_list, pred_list, average='weighted')
    f1 = f1_score(real_list, pred_list, average='weighted')

    return accuracy, precision, f1


def evaluate(model, data, epoch, args,classifier_model,is_test=0):
    logging.info("Begin to eval on validation set (epoch {} )...".format(epoch + 1))
    model.eval()
    if is_test:
        dataloader = data['test'].dataloader
        print('test dataloader:', len(dataloader), dataloader.num_batches)

    else:
        dataloader = data['val'].dataloader
        print('val dataloader:', len(dataloader), dataloader.num_batches)
    data_iter = iter(dataloader)

    top_1_acc, top_3_acc, top_5_acc = 0, 0, 0
    intent_1_acc, intent_3_acc, intent_5_acc = 0, 0, 0
    total_map = []

    with torch.no_grad():
        for i in range(dataloader.num_batches):
            print('=======================================')
            print('i:',i,dataloader.num_batches)
            batch = next(data_iter)

            session_ids, image_ids, images, summarys, raw_texts, fined_intents, \
            all_image_features,imgid2intent, origin_text = batch
            print('raw_texts:',origin_text)

            intent_features,flag = model(1, images, text=raw_texts, history=summarys,
                                    mask_ratio=args.mask_ratio)

            intent_prob, predict_label = classifier_model(intent_features,flag)
            predict_label = predict_label.tolist()

            intent_ids = fined_intents
            intent_ids = torch.tensor(np.array(intent_ids, dtype=int))
            
            img_dict, text_features, logit_scale = model(2, all_image_features, text=raw_texts, history=summarys,
                                                               intent_features=intent_features,
                                                               mask_ratio=args.mask_ratio)

            bz = len(text_features)
            cos = nn.CosineSimilarity(dim=0)
            for j in range(len(text_features)):
                score_dict = {}
                map_tmp = []
                for id, img in img_dict.items():
                    sim = cos(text_features[j], img[0]).item()
                    score_dict[id] = sim
                    if int(imgid2intent[id][0][0])==intent_ids.item():
                        if sim<0:
                            map_tmp.append(0)
                        else:
                            map_tmp.append(sim)
                if len(map_tmp)!=0:
                    map_tmp = sum(map_tmp)/len(map_tmp)
                    total_map.append(map_tmp)
                sorted_dict = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))

                top_1 = list(sorted_dict.keys())[:1]
                top_3 = list(sorted_dict.keys())[:3]
                top_5 = list(sorted_dict.keys())[:5]

                for each in top_1:
                    if each == image_ids[j]:
                        top_1_acc+=1
                        break
                for each in top_3:
                    if each == image_ids[j]:
                        top_3_acc+=1
                        break
                for each in top_5:
                    if each == image_ids[j]:
                        top_5_acc+=1
                        break

                for each in top_1:
                    tmp = []
                    for inte in imgid2intent[each][0]:
                        tmp.append(inte)
                    if fined_intents[j] in tmp:
                        intent_1_acc+=1

                for each in top_3:
                    tmp = []
                    for inte in imgid2intent[each][0]:
                        tmp.append(inte)
                    if fined_intents[j] in tmp:
                        intent_3_acc += 1
                for each in top_5:
                    tmp = []
                    for inte in imgid2intent[each][0]:
                        tmp.append(inte)
                    if fined_intents[j] in tmp:
                        intent_5_acc+=1


    total = dataloader.num_batches*bz
    if is_test:
        print('Test mAP',sum(total_map)/len(total_map))
        print('Test obviouse:', top_1_acc/total, top_3_acc/total, top_5_acc/total)
        print('Test intent:', intent_1_acc/total, intent_3_acc/total, intent_5_acc/total)
    else:
        print('Val mAP',sum(total_map)/len(total_map))
        print('Val intent:', intent_1_acc/total, intent_3_acc/total, intent_5_acc/total)


def cosineSimilarityLoss(feature1, feature2):
    scale_factor_h = feature1.shape[0] / feature2.size(0)
    scale_factor_w = feature1.shape[1] / feature2.size(1)

    feature2_interpolated = F.interpolate(feature2.unsqueeze(0).unsqueeze(0),
                                          size=(feature1.shape[0], feature1.shape[1]),
                                          mode='bilinear',
                                          align_corners=False)
    feature2_interpolated = feature2_interpolated.squeeze(0).squeeze(0)

    cosine_sim = F.cosine_similarity(feature1, feature2_interpolated, dim=1)
    similarity_loss = 1 - cosine_sim.mean()
    return similarity_loss