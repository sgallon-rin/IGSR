## Setup Instructions

1. Download the [pre-trained weights](https://pan.baidu.com/s/1tNYBhYKbIzYPw7BP7Be3rQ?pwd=3dyx) `clip_cn_vit-b-16.pt` and place them in the directory `IGSR/cn_clip/pretrained_weights`.
2. In the code, replace `sys.path.append('')` with the appropriate path based on your environment.

## How to get the dataset
1. Sign the copyright announcement with your name and organization and mail to bingbing.wang@stu.hit.edu.cn with the copyright announcement attachment.
2. Then complete the form online(https://forms.gle/g5eLLZVmaE43e3DF9), and we will provide access rights when approved.

## Running the Code

### 1. Build LMDB Features

Run the following script to build LMDB features:

```bash
python cn_clip/preprocess/build_lmdb_dataset_img.py
```
Example usage:
```bash
python build_lmdb_dataset_img.py --split val --type intent_style_attribute --mode yongyuan --threshold 600
```

### 2. Start Training
```bash
python3 -m torch.distributed.launch --use_env --nproc_per_node=1 --master_port=5555 cn_clip/training/main.py
python -m cn_clip.training.new_main
```

## Copyright
The original copyright of all the conversations belongs to the source owner. The copyright of the annotations belongs to our group, and they are free to the public. The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes or distributed to others.

## cite the paper
```bibtex
@inproceedings{wang2025new,
  title={A New Formula for Sticker Retrieval: Reply with Stickers in Multi-Modal and Multi-Session Conversation},
  author={Wang, Bingbing and Du, Yiming and Liang, Bin and Bai, Zhixin and Yang, Min and Wang, Baojun and Wong, Kam-Fai and Xu, Ruifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={24},
  pages={25327--25335},
  year={2025}
}
```
