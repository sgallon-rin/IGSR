from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass
from params import data_mode
import lmdb
import pickle
from nltk.corpus import wordnet
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from cn_clip.clip import _tokenizer
from cn_clip.clip import tokenize
from params import type


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text

def get_all_features(lmdb_path,split_mode):
    print('lmdb_path:',lmdb_path,split_mode)
    assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split_mode)
    lmdb_pairs = os.path.join(lmdb_path, "pairs")

    assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(
        lmdb_pairs, split_mode)

    lmdb_imgs_1 = os.path.join('/'+'/'.join(lmdb_path.split('/')[:-2])+'/lmdb_'+data_mode+'_intent_style_attribute', split_mode+"_imgs")

    env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
    txn_pairs = env_pairs.begin(buffers=True)
    # txn_pairs.commit()
    # env_pairs.close()
    env_imgs_1 = lmdb.open(lmdb_imgs_1, readonly=True, create=False, lock=False, readahead=False,
                                meminit=False)
    txn_imgs_1 = env_imgs_1.begin(buffers=True)


    number_samples = int(txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
    # print(number_samples)
    # exit()
    img_id_list_1, img_id_list_2, img_id_list_3, img_id_list_4 = [], [], [], []
    cursor = env_imgs_1.begin().cursor()
    for key, value in cursor:
        if key.decode('utf-8') != 'num_images':
            img_id_list_1.append(key.decode('utf-8'))

    id2intent_txt = '/home/sgallon/research/sticker-conv/IGSR/MultiChat/all/intent.txt'
    id2intent = {}
    intent2id = {}
    intent2token = {}

    with open(id2intent_txt, 'r', encoding='utf-8') as f:
        datas = f.readlines()
        f.close()

    for data in datas:
        data = data.strip('\n').split('\t')
        id2intent[data[0]] = data[1].lower()
        intent2id[data[1].lower()] = data[0]

    imgid2intent = {}
    mode_list = ['boba', 'kuaile', 'quanguo', 'yongyuan', 'siban']
    for mo in mode_list:
        with open(
                'your_path/IGSR/MultiChat/' + mo + '_sample_' + type + '.json') as f:
            datas = f.readlines()
            f.close()
        print('your_path/IGSR/MultiChat/' + mo + '_sample_' + type + '.json')
        for data in datas:
            data = eval(data)
            image_id = data['image_ids']  # + '.jpg'
            if image_id not in imgid2intent:
                imgid2intent[image_id] = []
            # print(data)
            imgid2intent[image_id].append(intent2id[data['fined_intent']])

    def has_different_elements(lst):
        return len(set(lst))==1

    session_ids = []
    sample_ids = []
    img_ids, historys, contexts,speakers,texts,fined_intents,summaries = {},{},{},{},{},{},{}

    resolution=224
    e_transform = create_transform(
                input_size=resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment='original',
                interpolation='bicubic',
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            # print('transform')
    e_transform = Compose(e_transform.transforms[:-3] + [_convert_to_rgb] + e_transform.transforms[-3:])

    image_features = []

    for i in range(number_samples):
        pair = pickle.loads(txn_pairs.get("{}".format(i).encode('utf-8')).tobytes())
        session_id, sample_id,summary,history, context, speaker,text,fined_intent, image_id= pair

        session_ids.append(session_id)
        sample_ids.append(sample_id)
        img_ids[sample_id] = image_id
        historys[sample_id] = history
        contexts[sample_id] = context
        speakers[sample_id] = speaker
        texts[sample_id] = text
        summaries[sample_id] = summary

        intent_id = intent2id[fined_intent.lower()]
        fined_intents[sample_id] = intent_id
        image_b64 = txn_imgs_1.get("{}".format(image_id).encode('utf-8')).tobytes()

        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = e_transform(image)
        image_features.append(image)

    histories = historys

    all_image_features = {}

    for i in range(len(img_id_list_1)):
        image_b64 = txn_imgs_1.get("{}".format(img_id_list_1[i]).encode('utf-8')).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))
        image = e_transform(image)
        all_image_features[img_id_list_1[i]]=image

    return session_ids,sample_ids,img_ids, histories, contexts,speakers,\
           texts,fined_intents,summaries,image_features,number_samples, all_image_features,imgid2intent

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        lmdb_path = lmdb_path
        print(lmdb_path,split)
        self.max_txt_length = max_txt_length
        self.session_ids,self.sample_ids, self.img_ids, self.histories, self.contexts, self.speakers, \
        self.texts, self.fined_intents, self.summaries, self.image_features,\
        self.number_samples, self.all_image_features,self.intent2id  = get_all_features(lmdb_path,split)
        self.dataset_len = self.number_samples
    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = create_transform(
                input_size=resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment='original',
                interpolation='bicubic',
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples
        session_id = self.session_ids[sample_index]
        sample_id = self.sample_ids[sample_index]
        image_feature = self.image_features[sample_index]

        image_id = self.img_ids[sample_id]
        history = self.histories[sample_id]
        context = self.contexts[sample_id]
        speaker = self.speakers[sample_id]
        text = self.texts[sample_id]
        fined_intent = self.fined_intents[sample_id]
        summary = self.summaries[sample_id]
        image = image_feature

        raw_text = context.split('\t')[1:]
        raw_text = '[SEP]'.join(raw_text)
        origin_text = raw_text
        raw_text= tokenize([_preprocess_text(raw_text)],[_preprocess_text(raw_text)], context_length=self.max_txt_length)
        new_summary = summary

        summary = tokenize([_preprocess_text(new_summary)],[_preprocess_text(new_summary)], context_length=self.max_txt_length)

        return session_id,image_id,image,summary,raw_text,fined_intent,self.all_image_features,self.intent2id, origin_text

def pad_dataset(dataset, global_batch_size):
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    vision_model_config_file = Path(
        __file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: LMDBDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train==1:
        db_path = args.train_data
        split='train'
    elif is_train==2:
        db_path = args.val_data
        split = 'val'
    elif is_train==0:
        db_path = args.test_data
        split = 'test'
    assert db_path is not None

    dataset = LMDBDataset(
        db_path,
        split=split,
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model),
    )

    batch_size = args.batch_size if is_train else args.valid_batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args,
            is_train=1,
            max_txt_length=max_txt_length,
            epoch_id=epoch_id)

    if args.val_data:
        data["val"] = get_dataset(
            args,
            is_train=2,
            max_txt_length=max_txt_length,
            epoch_id=epoch_id)

    if args.test_data:
        data["test"] = get_dataset(
            args,
            is_train=0,
            max_txt_length=max_txt_length,
            epoch_id=epoch_id)

    return data
