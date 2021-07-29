"""

Copyright (c) 2019 Gurkirt Singh 
 All Rights Reserved.

"""

import torch.nn as nn
import torch.nn.functional as F
import torch, pdb, time
from modules import box_utils
from modules import utils
logger = utils.get_logger(__name__)


# Credits:: from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/smooth_l1_loss.py
# smooth l1 with beta
def smooth_l1_loss(input, target, beta=1. / 9, reduction='sum'):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    return loss.sum()


def sigmoid_focal_loss(preds, labels, num_pos, alpha, gamma):
    '''Args::
        preds: sigmoid activated predictions
        labels: one hot encoded labels
        num_pos: number of positve samples
        alpha: weighting factor to baclence +ve and -ve
        gamma: Exponent factor to baclence easy and hard examples
       Return::
        loss: computed loss and reduced by sum and normlised by num_pos
     '''
    loss = F.binary_cross_entropy(preds, labels, reduction='none')
    alpha_factor = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    pt = preds * labels + (1.0 - preds) * (1.0 - labels)
    focal_weight = alpha_factor * ((1-pt) ** gamma)
    loss = (loss * focal_weight).sum() / num_pos
    return loss


def sigmoid_cb_focal_loss(preds, labels, num_pos, cls_num_list, alpha, gamma):
    if preds.shape[0] > 0:
        min_preds = torch.min(preds)
        if min_preds < 0:
            logger.warn("min_preds={} is NEGATIVE. Be careful with reverse the sign for sigmoid.".format(min_preds))

    gamma = 1
    beta = 0.9999
    weight = reweight(cls_num_list, beta=beta).cuda()
    ## weight_per_sample = torch.tensor([weight[y] for y in labels]).cuda()
    # print("cls_num_list:\n", cls_num_list)
    # print("len(cls_num_list) = ", len(cls_num_list))
    # print("weight:\n", weight)
    # print("weight.shape = ", weight.shape)
    # print("labels=", labels)
    # print("labels.shape=", labels.shape)

    loss = F.binary_cross_entropy(preds, labels, reduction='none')
    # logger.info("preds = {}".format(preds))
    # logger.info("labels = {}".format(labels))
    # logger.info("loss = {}".format(loss))
    # print("loss.shape=", loss.shape)

    pt = preds * labels + (1.0 - preds) * (1.0 - labels)

    # # original
    # alpha_factor = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    # focal_weight_alpha = alpha_factor * ((1 - pt) ** gamma)
    # print("focal_weight_alpha=\n", focal_weight_alpha)

    # cb
    cb_factor = weight * labels + (1.0 - beta) * (1.0 - labels)
    focal_weight = cb_factor * ((1 - pt) ** gamma)
    # print("focal_weight=\n", focal_weight)

    loss_adj = (loss * focal_weight).sum() / num_pos

    if loss_adj.item() > 200:
        logger.error("loss > 200.")
        logger.error("preds = {}".format(preds))
        logger.error("labels = {}".format(labels))
        logger.error("weight = {}".format(weight))
        logger.error("cb_factor = {}".format(cb_factor))
        logger.error("focal_weight = {}".format(focal_weight))
        logger.error("loss = {}".format(loss))
        logger.error("loss_adj = {}".format(loss_adj))
        logger.error("num_pos = {}".format(num_pos))

    return loss_adj


def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, default value of 0.9999 is set based on tuning performance in a2 report
    :return:
    '''
    per_cls_weights = torch.tensor([(1 - beta) / (1 - beta **n) for n in cls_num_list])
    # print("per_cls_weights=\n", per_cls_weights)
    # per_cls_weights = len(cls_num_list) * per_cls_weights / torch.sum(per_cls_weights)
    per_cls_weights = per_cls_weights / torch.sum(per_cls_weights)
    # print("(after normalization) per_cls_weights=\n", per_cls_weights)

    return per_cls_weights

#
# def sigmoid_cb_focal_loss_v1(preds, labels, num_pos, cls_num_list):
#     """
#     :param preds: sigmoid activated predictions
#     :param labels: integer labels (NOT one-hot encoded)
#     :param num_pos: number of positve samples
#     :return:
#     """
#     weight = reweight(cls_num_list)
#     print("cls_num_list:\n", cls_num_list)
#     print("len(cls_num_list) = ", len(cls_num_list))
#     print("weight:\n", weight)
#     print("weight.shape = ", weight.shape)
#     print("labels=", labels)
#     # weight = [1] + weight # [1] is for agent_ness having no CB effect. 7/26 - no need as agentness count is added. now len(cls_num_list) should be 149.
#
#     m = labels.shape[0]
#     # gamma = 1
#     # weight_per_sample = torch.tensor([weight[y] for y in labels]).cuda()
#     # log_likelihood = -weight_per_sample * torch.log(preds[range(m), labels]) * torch.pow(
#     #     1 - preds[range(m), labels], gamma)
#     alpha = 0.25
#     gamma = 2.0
#     log_likelihood = -alpha * torch.log(preds[range(m), labels]) * torch.pow(
#         1 - preds[range(m), labels], gamma)
#     loss = torch.sum(log_likelihood) / num_pos
#     print("log_likelihood=\n", log_likelihood)
#     print("torch.sum(log_likelihood)={}; num_pos={}; m={}".format(torch.sum(log_likelihood), num_pos, m))
#     return loss


def get_one_hot_labels(tgt_labels, numc):
    new_labels = torch.zeros([tgt_labels.shape[0], numc], device=tgt_labels.device)
    new_labels[:, tgt_labels] = 1.0
    return new_labels


def onehot2int(one_hot):
    """
    Convert one-hot tensor
    :param one_hot:
    :return:
    """
    print("one_hot.shape=", one_hot.shape)
    print("torch.sum(one_hot) = ", torch.sum(one_hot))
    print("torch.max(one_hot, 1)[:10] = ", torch.max(one_hot, 1)[:10])
    # print("one_hot.tolist()[:10] = ", one_hot.tolist()[:10])
    out = torch.argmax(one_hot, dim=-1)
    debug_filter = [i for i in out.tolist() if i > 0]
    print("debug_filter=", debug_filter)
    # print("out.tolist()[:10] = ", out.tolist()[:10])
    return out




class FocalLoss(nn.Module):
    def __init__(self, args, alpha=0.25, gamma=2.0):
        """Implement YOLO Loss.
        Basically, combines focal classification loss
         and Smooth L1 regression loss.
        """
        super(FocalLoss, self).__init__()
        self.positive_threshold = args.POSTIVE_THRESHOLD
        self.negative_threshold = args.NEGTIVE_THRESHOLD
        self.num_classes = args.num_classes
        self.num_label_types = args.num_label_types
        self.num_classes_list = args.num_classes_list
        self.alpha = 0.25
        self.gamma = 2.0
        self.cls_num_list = args.cls_num_list


    def forward(self, confidence, predicted_locations, gt_boxes, gt_labels, counts, anchors, ego_preds, ego_labels):
        ## gt_boxes, gt_labels, counts, ancohor_boxes
        
        """
        
        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4): predicted locations.
            boxes list of len = batch_size and nx4 arrarys
            anchors: (num_anchors, 4)

        """
        ego_preds = torch.sigmoid(ego_preds)
        ps = confidence.shape
        preds = torch.sigmoid(confidence)
        # ps = predicted_locations.shape
        # predicted_locations = predicted_locations.view(ps[0],ps[1], -1, [-1])
        ball_labels = []
        bgt_locations = []
        blabels_bin = []
        # mask = torch.zeros([preds.shape[0],preds.shape[1]], dtype=torch.int)

        with torch.no_grad():
            # gt_boxes = gt_boxes.cpu()
            # gt_labels = gt_labels.cpu()
            # anchors = anchors.cpu()
            # device = torch.device("cpu")
            device = preds.device
            zeros_tensor = torch.zeros(1, gt_labels.shape[-1], device=device)
            for b in range(gt_boxes.shape[0]):
                all_labels = []
                gt_locations = []
                labels_bin = []
                for s in range(gt_boxes.shape[1]):
                    gt_boxes_batch = gt_boxes[b, s, :counts[b,s], :]
                    gt_labels_batch = gt_labels[b, s, :counts[b,s], :]
                    if counts[b,s]>0:
                        gt_dumy_labels_batch = torch.LongTensor([i for i in range(counts[b,s])]).to(device)
                        conf, loc = box_utils.match_anchors_wIgnore(gt_boxes_batch, gt_dumy_labels_batch, 
                            anchors, pos_th=self.positive_threshold, nge_th=self.negative_threshold )
                    else:
                        loc = torch.zeros_like(anchors, device=device)
                        conf = ego_labels.new_zeros(anchors.shape[0], device=device) - 1
                    
                    # print(conf.device)
                    # print(loc.device)
                    gt_locations.append(loc)
                    labels_bin.append(conf)

                    dumy_conf = conf.clone()
                    dumy_conf[dumy_conf<0] = 0
                    labels_bs = torch.cat((zeros_tensor, gt_labels_batch),0)
                    batch_labels = labels_bs[dumy_conf,:]
                    all_labels.append(batch_labels)

                all_labels = torch.stack(all_labels, 0).float()
                gt_locations = torch.stack(gt_locations, 0)
                labels_bin = torch.stack(labels_bin, 0).float()
                ball_labels.append(all_labels)
                bgt_locations.append(gt_locations)
                blabels_bin.append(labels_bin)
            
            all_labels = torch.stack(ball_labels, 0)
            gt_locations = torch.stack(bgt_locations, 0)
            labels_bin = torch.stack(blabels_bin, 0)
            # mask = labels_bin > -1
            # device = ego_preds.device
            # all_labels = all_labels.to(device)
            # gt_locations = gt_locations.to(device)
            # labels_bin = labels_bin.to(device)

        # bgt_locations = []
        # blabels_bin = []
        pos_mask = labels_bin > 0
        num_pos = max(1.0, float(pos_mask.sum()))
        
        gt_locations = gt_locations[pos_mask].reshape(-1, 4)
        predicted_locations = predicted_locations[pos_mask].reshape(-1, 4)
        regression_loss = smooth_l1_loss(predicted_locations, gt_locations)/(num_pos * 4.0)
        
        # if regression_loss.item()>40:
        #     pdb.set_trace()
        
        mask = labels_bin > -1 # Get mask to remove ignore examples
        
        masked_labels = all_labels[mask].reshape(-1, self.num_classes) # Remove Ignore labels
        masked_preds = preds[mask].reshape(-1, self.num_classes) # Remove Ignore preds
        # print("Jing test: cls_loss all_labels[mask]: {} and preds[mask]: {}".format(all_labels[mask].shape, preds[mask].shape))
        # print("self.num_classes = {}".format(self.num_classes))
        # print("all_labels[mask]:\n", all_labels[mask])
        # print("all_labels[mask].shape = ", all_labels[mask].shape)
        # print("all_labels[mask].max() = ", all_labels[mask].max())
        # print("preds[mask]:\n", preds[mask])
        # cls_loss = sigmoid_focal_loss(masked_preds, masked_labels, num_pos, self.alpha, self.gamma)
        # cls_loss = sigmoid_cb_focal_loss(masked_preds, onehot2int(masked_labels), num_pos, self.cls_num_list)
        cls_loss = sigmoid_cb_focal_loss(masked_preds, masked_labels, num_pos, self.cls_num_list, self.alpha, self.gamma) # label is one-hot encoding

        mask = ego_labels>-1
        numc = ego_preds.shape[-1]
        masked_preds = ego_preds[mask].reshape(-1, numc) # Remove Ignore preds
        masked_labels = ego_labels[mask].reshape(-1) # Remove Ignore labels
        # print("ego_preds[mask]:\n", ego_preds[mask])
        # print("ego_labels[mask]:\n", ego_labels[mask])
        one_hot_labels = get_one_hot_labels(masked_labels, numc)
        ego_loss = 0
        if one_hot_labels.shape[0]>0:
            ego_loss = sigmoid_focal_loss(masked_preds, one_hot_labels, one_hot_labels.shape[0], self.alpha, self.gamma)
            # ego_loss = sigmoid_cb_focal_loss(masked_preds, masked_labels, masked_labels.shape[0])
        
        # print(regression_loss, cls_loss, ego_loss)
        # print("regression_loss={}, cls_loss={}, ego_loss={}".format(regression_loss, cls_loss, ego_loss))
        return regression_loss, cls_loss/8.0 + ego_loss/4.0
