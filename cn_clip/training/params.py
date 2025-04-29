import argparse
is_DDIM = 0
is_att = 0
is_matrix = 0
is_sent_embedding = False

data_mode = 'kuaile'
type = 'intent_style_attribute'

def get_default_params(model_name):
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B-32", "ViT-B-16", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    elif model_name in ["ViT-L-14", "ViT-L-14-336"]:
        return {"lr": 4.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
    # data_mode = data
    global num_class, type


    if type == 'intent_style_attribute':
        name = 'all'
    else:
        name = type
    if data_mode == 'quanguo' or data_mode == 'total':
        num_class = 23
    else:
        num_class = 25
    mode = 'wi'
    parser.add_argument(
        "--num_class", type=int, default=25, help="The number of workers for training dataloader."
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default=data_mode
    )
    parser.add_argument(
        "--type",
        type=str,
        default=type
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=mode
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default='your_path/IGSR/MultiChat/lmdb_'+data_mode+'_intent_style_attribute/train',
        required=False,
        help="Path to the LMDB directory with training data split",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default='your_path/IGSR/MultiChat/lmdb_'+data_mode+'_intent_style_attribute/val',
        help="Path to the LMDB directory with validation data split, default to None which disables validation",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default='your_path/IGSR/MultiChat/lmdb_'+data_mode+'_intent_style_attribute/test',
        help="Path to the LMDB directory with validation data split, default to None which disables validation",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="The number of workers for training dataloader."
    )

    parser.add_argument(
        "--valid-num-workers", type=int, default=1, help="The number of workers for validation dataloader (if making validation)."
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="your_path/IGSR/cn_clip/save/",
        help="Where to store logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train_clip",
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="How often to log loss info."
    )
    parser.add_argument(
        "--report-training-batch-acc", default=True, action="store_true", help="Whether to report training batch accuracy."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training per GPU."
    )
    parser.add_argument(
        "--valid-batch-size", type=int, default=1, help="Batch size for validation per GPU."
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Number of steps to train for (in higher priority to --max_epochs)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=30, help="Number of full epochs to train for (only works if --max_steps is None)."
    )
    parser.add_argument(
        "--valid-step-interval", type=int, default=None, help="The step interval for validation (default to None which disables validation between steps)."
    )
    parser.add_argument(
        "--valid-epoch-interval", type=int, default=1, help="The epoch interval for validation (default to 1, set None to disable validation between epochs)."
    )
    parser.add_argument(
        "--context-length", type=int, default=512, help="The maximum length of input text (include [CLS] & [SEP] tokens). Default to 52."
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.001, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=100, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--context_length", type=int, default=512, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--accum_freq", type=int, default=1, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync."
    )
    parser.add_argument("--use-augment",
        default=False,
        action="store_true",
        help="Whether to use image augment."
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-epoch-frequency", type=int, default=1, help="How often to save checkpoints by epochs."
    )
    parser.add_argument(
        "--save-step-frequency", type=int, default=200, help="How often to save checkpoints by steps."
    )
    parser.add_argument(
        "--resume",
        default='your_path/IGSR/cn_clip/pretrained_weights/clip_cn_vit-b-16.pt',
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--classifier-resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        default=True,
        help="If resumed from a checkpoint, whether to reset the optimizer states.",
    )
    parser.add_argument(
        "--reset-data-offset",
        action="store_true",
        default=True,
        help="If resumed from a checkpoint, whether to reset the dataset offset to the beginning.",
    )    
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--mask-ratio",
        default=0,
        type=float,
        help="Random mask ratio of patches during finetuning. Default to zero which does not mask any patches.",
    )
    parser.add_argument(
        "--clip-weight-path",
        default=None,
        type=str,
        help="The path of openai pretrained weight, used to initialize the image encoder, should be set to None if you do not use pretrained CLIP",
    )
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=False,
        help="Freeze the weight of vision encoder.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--bert-weight-path",
        default=None,
        type=str,
        help="The path of bert pretrained weight, used to initialize the text encoder, should be set to None if you do not use pretrained BERT",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--use-flash-attention",
        default=False,
        action="store_true",
        help="Enable flash attention."
    )
    parser.add_argument(
        "--accum-freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    # arguments for distributed training
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed."
    )
    # arguments for distllation
    parser.add_argument(
        "--distllation",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--teacher-model-name",
        type=str,
        default=None,
        help="The name of teacher model."
    )
    parser.add_argument(
        "--kd_loss_weight",
        type=float,
        default=0.5,
        help="Weight of KD loss."
    )
    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    default_params = get_default_params(args.vision_model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
