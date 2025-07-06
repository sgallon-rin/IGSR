# -*- coding: utf-8 -*-

import argparse
import os
from tqdm import tqdm
import lmdb
import json
import pickle

num = 8

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=False, help="the directory which stores the image tsvfiles and the text jsonl annotations"
    )
    parser.add_argument(
        "--splits", type=str, required=True, help="specify the dataset splits which this script processes, concatenated by comma \
            (e.g. train,valid,test)"
    )
    parser.add_argument(
        "--type", type=str, required=True, help="specify the dataset splits which this script processes, concatenated by comma \
                (e.g. intent)"
    )
    parser.add_argument(
        "--mode", type=str, required=True, help="specify the dataset splits which this script processes, concatenated by comma \
                (e.g. boba, kuaile,quanguo,siban)"
    )
    parser.add_argument(
        "--threshold", type=int, required=True, help="specify the dataset splits which this script processes, concatenated by comma \
                    (e.g. boba, kuaile,quanguo,siban)"
    )
    parser.add_argument(
        "--lmdb_dir", type=str, default=None, help="specify the directory which stores the output lmdb files. \
            If set to None, the lmdb_dir will be set to {args.data_dir}/lmdb"
    )
    return parser.parse_args()

def get_b64_list(base64_path):
    image_id_list, b64_list = [], []
    print('base64_path:', base64_path)
    with open(base64_path, "r", encoding="utf-8") as fin_imgs:
        for line in tqdm(fin_imgs):
            line = line.strip()
            image_id, b64 = line.split("\t")
            image_id_list.append(image_id)
            b64_list.append(b64)

    return b64_list,image_id_list

def commit_pair(image_id_list,b64_list,txn_img_1,write_idx,env_img_1):
    for i in range(len(image_id_list)):
        image_id = image_id_list[i]
        b64 = b64_list[i]
        print('image:',image_id)
        txn_img_1.put(key="{}".format(image_id).encode('utf-8'), value=b64.encode("utf-8"))
        write_idx += 1
        if write_idx % 1000 == 0:
            txn_img_1.commit()
            txn_img_1 = env_img_1.begin(write=True)

    txn_img_1.put(key=b'num_images',
                value="{}".format(write_idx).encode('utf-8'))
    txn_img_1.commit()
    env_img_1.close()
    return write_idx

if __name__ == "__main__":
    args = parse_args()
    # args.data_dir = '/home/sgallon/research/sticker-conv/IGSR/MultiChat/'
    args.data_dir = '/mnt/tsubasa/sticker_chat_datasets/MultiChat_Dataset/MultiChat-Dataset'
    specified_splits = list(set(args.splits.strip().split(",")))
    print("Dataset splits to be processed: {}".format(", ".join(specified_splits)))
    threshold = args.threshold
    # build LMDB data files
    if args.lmdb_dir is None:
        if args.mode!='all':
            args.lmdb_dir = os.path.join(args.data_dir, "lmdb_" + args.mode + '_' + args.type)
        else:
            args.lmdb_dir = os.path.join(args.data_dir, "lmdb_"+args.mode+'_'+args.type+'_'+str(threshold))
    for split in specified_splits:
        # open new LMDB files
        lmdb_split_dir = os.path.join(args.lmdb_dir, split)
        if os.path.isdir(lmdb_split_dir):
            print("We will overwrite an existing LMDB file {}".format(lmdb_split_dir))
        os.makedirs(lmdb_split_dir, exist_ok=True)

        lmdb_img_1 = os.path.join(args.lmdb_dir, ""+args.splits+"_imgs")
        env_img_1 = lmdb.open(lmdb_img_1, map_size=1024**4)
        txn_img_1 = env_img_1.begin(write=True)

        lmdb_pairs = os.path.join(lmdb_split_dir, "pairs")
        env_pairs = lmdb.open(lmdb_pairs, map_size=1024**4)
        txn_pairs = env_pairs.begin(write=True)

        if args.type == 'intent_style_attribute':
            name = 'all'
        else:
            name = args.type

        if args.mode!='all':
            pairs_annotation_path = os.path.join(args.data_dir,
                                                 args.mode + "/" + args.type + '/' + args.mode + "_session_" + name + '_' + args.splits + ".jsonl")
        else:
            pairs_annotation_path = os.path.join(args.data_dir, args.mode+"/"+args.type+'/'+args.mode+"_session_"+name+'_'+args.splits+"_"+str(threshold)+".jsonl")

        with open(pairs_annotation_path, "r", encoding="utf-8") as fin_pairs:
            write_idx = 0
            for line in tqdm(fin_pairs):
                line = line.strip()
                obj = eval(line)
                for field in ('session_id',"sample_id", "summary","related_history_session_ids", "dialogue_context",'speaker','text',
                              'fined_intent','image_ids'):
                    assert field in obj, "Field {} does not exist in line {}. \
                        Please check the integrity of the text annotation Jsonl file."

                dump = pickle.dumps(( obj['session_id'],obj['sample_id'], obj['summary'], obj['related_history_session_ids'],
                                     obj['dialogue_context'], obj['speaker'],
                                     obj['text'], obj['fined_intent'],
                                     obj['image_ids']))  # encoded (image_id, text_id, text)
                if 'User' not in obj['speaker']:
                    print(obj['speaker'], obj['response'])
                    exit()
                txn_pairs.put(key="{}".format(write_idx).encode('utf-8'), value=dump)
                write_idx += 1

            txn_pairs.put(key=b'num_samples',
                    value="{}".format(write_idx).encode('utf-8'))
            txn_pairs.commit()
            env_pairs.close()
        print("Finished serializing {} {} split pairs into {}.".format(write_idx, split, lmdb_pairs))

        base64_path = os.path.join(args.data_dir,
                                         args.mode+"_"+args.splits+"_img.tsv")

        write_idx = 0
        write_idx2 = 0
        write_idx3 = 0
        b64_list, image_id_list = get_b64_list(base64_path)
        write_idx = commit_pair(image_id_list, b64_list, txn_img_1, write_idx, env_img_1)


        print("Finished serializing {} {} split images into {}.".format(write_idx, split, lmdb_img_1))

    print("done!")