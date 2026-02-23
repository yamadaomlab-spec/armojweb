import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--cosinelrscheduler', action='store_true',
                        help="If true , we use cosine LR Scheduler")
    # Model parameters
    parser.add_argument('--pretrain_weights', type=str, default=None,
                        help="Path to the pretrained model")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str, choices =['resnet101', 'swin'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--label_ext_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parametersbase
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # DHP 
    parser.add_argument('--name_dataset', default='dhp20231110', choices =['dhp', 'kuzushiji', 'dhp20231110', 'minhon', 'aozora', 'ndlocr', 'kuzushijiyorei', 'marge'],
                        help='dataset name')
    parser.add_argument('--charset', default='dhp20231110_charset.pkl',
                        help='charset name')   
    parser.add_argument('--data_dir', default='../data', 
                        help='path to data dir')    
    parser.add_argument('--fold_id', default='0', type=int,  choices = [0,1,2,3,4],
                        help='from 0 to 4 fo 5-fold cross validation')
    parser.add_argument('--method', default='crbbcu', choices =['base', 'cr', 'crbb', 'crbbcu'],
                        help='base: baseline, cr: use line crop augumentation, bb: predict bbox, cu: user cursor position')
    parser.add_argument('--widthout_eval', action='store_true')    
    parser.add_argument('--no_test', action='store_true') 
    parser.add_argument('--small', default='', choices= ['', '02', 'small'],
                        help='small vertion of dataset')   
    parser.add_argument('--subtrain', default='', choices= ['', '01', '02', '03', '04', '05', '06', '07', '08', '09'],
                        help='traindata ratio')  
    parser.add_argument('--max_crop_length', default=None, type=int)
    parser.add_argument('--kfold', default='5', type=int, choices = [1, 5,10])   

    parser.add_argument('--transformer_type', default='original', choices =['original', 'gtp_parallel_tsa', 'gtp_serial_tsa'])
    parser.add_argument('--addstr', default='', help='addstr is added to name_method')
    parser.add_argument('--outfilemark', type=str, default='',
                        help="filemark for output")
    parser.add_argument('--test_data', default='test', choices =['test', 'train'],
                        help='data for evaluation')
    parser.add_argument('--decoder_input_is_representative_chars', action='store_true')
    parser.add_argument('--decoder_output_representative_chars', action='store_true')
    
    parser.add_argument('--checkpoint_path', default='./checkpoint.pth',
                    help='path where to checkpoint, empty for no saving')
    
    return parser