# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import os
import torch
import util.misc as utils
from datasets.open_world_eval import OWEvaluator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from models.matcher import build_matcher
import models.OW_FOMO as OW_FOMO

from util import box_ops

from tqdm import tqdm
import time

# Training step per epoch
def train_one_epoch(model, dataloader, epoch, args):
    # Initialize loss_epoch based on the value of nc_epoch in args
    if args.nc_epoch > 0:
        loss_epoch = 9
    else:
        loss_epoch = 0

    # Enable gradients for model attributes
    model.att_embeds.requires_grad_(True)
    # print(f"att_embeds requires_grad: {model.att_embeds.requires_grad}")
    model.att_W.requires_grad_(True)
    # print(f"att_W requires_grad: {model.att_W.requires_grad}")
    
    # Every even epoch uses "attribute_refinement" mode, odd epochs use "attribute_selection
    if epoch % 2 == 0:
        optimizer_attr = torch.optim.AdamW([{'params': model.att_embeds}]) # Optimizer for attribute selection
        model.mode = "attribute_refinement"
    else:
        optimizer_attr = torch.optim.AdamW([{'params': model.att_W}]) # Optimizer for FOMO attribute selection
        model.mode = "attribute_selection"
    
    optimizer_box = torch.optim.AdamW(model.model.box_head.parameters()) # Optimizer for box head
    # Define criteria for attribute and box losses
    criterion_attr = OW_FOMO.SetCriterionAttr()
    criterion_box = OW_FOMO.SetCriterionBox(args)

    # Iterate over the dataloader
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        # Zero the gradients for both optimizers
        print("\n==================================")
        print(f"In batch {batch_idx}")
        print("==================================")
        optimizer_attr.zero_grad()
        optimizer_box.zero_grad()

        # print(f"len(batch): {len(batch)}")
        # print(f"batch[0].tensors.shape: {batch[0].tensors.shape}")
        # print(f"batch[1]: {batch[1]}")

        #Forward
        targets, outputs, matched_query_idx, unmatched_gt_idx = model(batch)
        
        # TODO 1: match outputs and targets according to the indexes

        # TODO 2: Augment targets with high confidence unmatched outputs
        '''
        if epoch > loss_epoch:

            logits, obj, boxes = outputs['logits'], outputs['objectness'], outputs['boxes']
            
            logits_known, obj_known, boxes_known = logits[matched_query_idx], obj[matched_query_idx], boxes_known[matched_query_idx]
            logits_known = logits_known.to(logits.device())
            obj_known = obj_known.to(obj.device())
            boxes_known = boxes_known.to(boxes.device())

            all_indices = torch.arange(logits.size(0))
            # unmatched_query_idx = torch.tensor([i for i in all_indices if i not in matched_query_idx])

            # logits_unmatched = logits[unmatched_query_idx]
            # obj_unmatched = obj[unmatched_query_idx]
            # boxes_unmatched = boxes[unmatched_query_idx]
            # logits_unmatched = logits_unmatched.to(logits.device())
            # obj_unmatched = obj_unmatched.to(obj.device())
            # boxes_unmatched = boxes_unmatched.to(boxes.device())


            prob = torch.sigmoid(logits)
            prob[..., -1] *= obj.squeeze(-1)
            scores, topk_indexes = torch.topk(obj.unsqueeze(-1), 30, dim=1)
            topk_indexes_unmatched = torch.tensor([i for i in topk_indexes if i not in matched_query_idx])
            # scores = scores.squeeze(-1)
            # labels = torch.ones(scores.shape, device=scores.device) * logits_unmatched.shape[-1]
            # import ipdb; ipdb.set_trace()
            # boxes = torch.gather(boxes_unmatched, 1, topk_indexes_unmatched.repeat(1, 1, 4))
            logits_pseudo_gt = torch.zeros((topk_indexes_unmatched.size(0), logits.size[-1])).to(logits.device)
            # [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
            
            tot_logits_target = []
            tot_logits_target.append(targets['logits'])
            tot_logits_target.append(logits_pseudo_gt)
            tot_logits_target = torch.cat(tot_logits_target, dim=0)

            tot_logits_output = []
            tot_logits_output.append(outputs['logits'][matched_query_idx])
            tot_logits_output.append(outputs['logits'][topk_indexes_unmatched])
            tot_logits_output = torch.cat(tot_logits_output, dim=0)
        '''

        # Estimating loss and perform backward propagation on attr head
        logits_with_grad = outputs["logits"][targets["masks"] > 0]
        targets_logits_with_grad = targets["logits"][targets["masks"] > 0]
        loss_attr = criterion_attr(outputs["logits"][targets["masks"]>0], targets["logits"][targets["masks"]>0])
        print(f"loss_attr: {loss_attr}")
        loss_attr.retain_grad()
        loss_attr.backward()
        optimizer_attr.step()

        # print(f"logits_with_grad requires_grad: {logits_with_grad.requires_grad}")
        # print(f"targets_logits_with_grad requires_grad: {targets_logits_with_grad.requires_grad}")
        # print(f"loss_attr.requires_grad: {loss_attr.requires_grad}")
        # print(f"loss_attr.grad: {loss_attr.grad}")
        # Check gradients of model parameters
        #for name, param in model.named_parameters():
        #    if param.requires_grad and param.grad is not None:
        #        print(f"Gradient for {name} after attribute loss backward pass: {param.grad.norm().item()}")
        #print(f"att_W.grad: {model.att_W.grad.norm().item()}")
        # print(f"model.att_embeds before optim: {model.att_embeds}")
        # print(f"model.att_embeds after optim: {model.att_embeds}")
        
        # GT + pseudo GT boxes
        # print(f"outputs['boxes'].shape: {outputs['boxes'].shape}")
        # print(f"outputs['boxes'][matched_query_idx].shape: {outputs['boxes'][matched_query_idx].shape}")
        # print(f"targets['boxes'].shape: {targets['boxes'].shape}")
        # print(f"targets['masks'].shape: {targets['masks'].shape}")

        # Initialize total box loss
        tot_loss_box = 0
        tot_loss_box_list = []
        for img in range(targets['boxes'].shape[0]):
            # Replace -1 with a dummy value (max index) for safe indexing
            my_mask = matched_query_idx[img].squeeze(-1) != -1
            valid_gt_boxes = targets["boxes"][img][my_mask].to(outputs["boxes"].device)
            valid_indices = matched_query_idx[img].squeeze(-1)[my_mask]
            valid_indices_long = valid_indices.long()
            
            # print(f"****************** In image {batch} of batch ******************")
            # print(f"matched_query_idx[batch].squeeze(-1): {matched_query_idx[batch].squeeze(-1)}")
            # print(f"unmatched_gt_idx[batch]: {unmatched_gt_idx[batch]}")
            # print(f"my_mask: {my_mask}")
            # print(f"valid_indices: {valid_indices}")
            # print(f"valid_indices_long: {valid_indices_long}")
            
            matched_pred_boxes = outputs["boxes"][img][valid_indices_long]
            print(f"----------image separation------------")
            print(f"valid_gt_boxes.shape: {valid_gt_boxes.shape}")
            print(f"matched_pred_boxes.shape: {matched_pred_boxes.shape}")
            for i in range(valid_gt_boxes.shape[0]):
                print("****************************************")
                print(f"valid_gt_boxes {i}: {valid_gt_boxes[i]}")
                print(f"matched_pred_boxes {i}: {matched_pred_boxes[i]}")
                
                iou = box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(valid_gt_boxes[i]).unsqueeze(0),
                    box_ops.box_cxcywh_to_xyxy(matched_pred_boxes[i]).unsqueeze(0))
                
                giou = box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(valid_gt_boxes[i]).unsqueeze(0),
                    box_ops.box_cxcywh_to_xyxy(matched_pred_boxes[i]).unsqueeze(0))
                print(f"iou of the two boxes: {iou}")
                print(f"giou of the two boxes: {giou}")

            fn_mask = unmatched_gt_idx[img] != -1
            fns = unmatched_gt_idx[img][fn_mask]
            loss_box = criterion_box(matched_pred_boxes, valid_gt_boxes, fns.shape[0])
            tot_loss_box_list.append(loss_box)
            # print(f"matched_pred_boxes.requires_grad: {matched_pred_boxes.requires_grad}")
            # print(f"matched_query_idx[matched_query_idx>0]: {matched_query_idx[matched_query_idx>0]}")

        #Sum up box losses
        tot_loss_box_tensor = torch.tensor(tot_loss_box_list, dtype=torch.float32).to(outputs["boxes"].device)
        tot_loss_box_tensor.requires_grad_(True)
        tot_loss_box = torch.sum(tot_loss_box_tensor).to(outputs["boxes"].device)
        tot_loss_box.requires_grad_(True)
        
        # Check gradients of model parameters
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"paramenter {name}.requires_grad() = True")
        # print(f"type(tot_loss_box): {type(tot_loss_box)}")   #tensor
        # print(f"tot_loss_box.shape: {tot_loss_box.shape}")   #[]
        # print(f"tot_loss_box.dtype: {tot_loss_box.dtype}")   #float32
        
        print(f"tot_loss_box: {tot_loss_box}")

        # Check if tot_loss_box is a tensor
        if type(tot_loss_box) != torch.Tensor:
            print(f"Found non tensor tot_loss_box")
            tot_loss_box = torch.tensor(tot_loss_box, dtype=torch.float32)
            tot_loss_box.requires_grad_(True)
        # Perform backward propogation and optimizer step for box head
        tot_loss_box.retain_grad()
        tot_loss_box.backward()
        optimizer_box.step()
        
        # print(f"tot_loss_box.requires_grad: {tot_loss_box.requires_grad}")
        # print(f"tot_loss_box.grad: {tot_loss_box.grad}")
    
    #Disable gradients for attribute or weight matrix based on whether the epoch is odd or even
    if epoch % 2 != 0:
        # Attribute refinement
        model.att_embeds.requires_grad_(False)
    else:
        # Attribute selection
        model.att_W.requires_grad_(False)


# Evaluating
@torch.no_grad()
def evaluate(model, postprocessors, data_loader, base_ds, device, output_dir, args):
    # Set model to evaluation mode and set inference mode
    model.eval()
    model.mode = 'inference'
    # Initialize metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # Initialize the COCO evaluator
    coco_evaluator = OWEvaluator(base_ds, args=args)

    # Start iterate over the dataloader
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        print(f"samples.tensors.shape: {samples.tensors.shape}")
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Forward
        outputs = model(batch=None, pixel_values=samples.tensors)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # Post-process
        results = postprocessors(outputs, orig_target_sizes)

        # Extract image IDs and check for duplicates
        image_ids = [str(target['image_id'][0]) for target in targets]
        if len(set(image_ids)) != len(image_ids):
            import ipdb
            ipdb.set_trace()

        # Process image IDs for evaluation
        processed_img_ids = [str(target['image_id'].item())[4:] if str(target['image_id'].item()).startswith("2021") else str(target['image_id'].item()) for target in targets]
        res = {id: output for id, output in zip(processed_img_ids, results)}
        
        # Update the evaluator with the results
        if coco_evaluator is not None:
            # print(f"called here")
            # time.sleep(2)
            coco_evaluator.update(res)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()

    # Gather the stats
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics'] = res
    if coco_evaluator is not None:
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator

# Visualizing
@torch.no_grad()
def viz(model, postprocessors, data_loader, device, output_dir, base_ds, args):
    model.eval()    #Set model to evaluation mode
    # Initialize metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Viz:'

    # Start iterate over the data loader
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Forward
        outputs = model(samples.tensors)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # Post-process
        results = postprocessors(outputs, orig_target_sizes, args.viz)
        plot_batch(samples.tensors, results, targets, args.output_dir,
                   [''.join([chr(int(i)) for i in target['image_id']]) + '.jpg' for target in targets],
                   base_ds.KNOWN_CLASS_NAMES + ['unknown'], orig_target_sizes)

    return

# Plotting bounding boxes on the images
@torch.no_grad()
def plot_batch(samples, results, targets, output_dir, image_names, cls_names, orig_target_sizes):
    # Iterate through each sample and its corresponding result
    for i, r in enumerate(results):
        # Axes converting
        img = samples[i].swapaxes(0, 1).swapaxes(1, 2).detach().cpu()
        # Plot bounding boxes on the image
        plot_bboxes_on_image({k: v.detach().cpu() for k, v in r.items()}, img.numpy(), output_dir, image_names[i],
                             cls_names, num_known=sum(targets[i]['labels'] < len(cls_names) - 1),
                             num_unknown=sum(targets[i]['labels'] == len(cls_names) - 1), img_size=orig_target_sizes[i])

    return

# Individual detections box plotting
def plot_bboxes_on_image(detections, img, output_dir, image_name, cls_names, num_known=10, num_unknown=5,
                         img_size=None):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory
    img = img * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.4578275, 0.40821073])
    # Extract detections from dictionary
    # import ipdb; ipdb.set_trace()
    if True:
        unk_ind = detections['labels'] == len(cls_names) - 1  # Indices of unknown labels
        unk_s = detections['scores'][unk_ind]
        unk_l = detections['labels'][unk_ind]
        unk_b = detections['boxes'][unk_ind]
        unk_s, indices = unk_s.topk(min(num_unknown + 1, len(unk_s)))  # Top unknown scores
        unk_l = unk_l[indices]
        unk_b = unk_b[indices]

        k_s = detections['scores'][~unk_ind]
        k_l = detections['labels'][~unk_ind]
        k_b = detections['boxes'][~unk_ind]
        k_s, indices = k_s.topk(min(num_known + 3, len(k_s)))  # Top known scores
        k_l = k_l[indices]
        k_b = k_b[indices]
        # Combine known and unknown detections
        scores = torch.cat([k_s, unk_s])
        labels = torch.cat([k_l, unk_l])
        boxes = torch.cat([k_b, unk_b])
    else:
        scores = detections['scores']
        labels = detections['labels']
        boxes = detections['boxes']

    # Create a plot 
    fig, ax = plt.subplots(1)
    plt.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Size adjusting
    img_size = (img_size * 840 / img_size.max()).detach().cpu().numpy().astype('int32')
    ax.imshow(img[:img_size[0], :img_size[1], :])

    # Plot bounding boxes on image
    for i in range(len(labels)):
        score = scores[i]
        label = cls_names[int(labels[i])]
        # Conditions
        if (label == 'unknown' and score > -0.025) or \
                (label != 'unknown' and score > 0.25) or label == 'fish':

            box = boxes[i]

            xmin, ymin, xmax, ymax = [int(b) for b in box.numpy().astype(np.int32)]
            # Set bounding box color based on label
            if label == 'unknown':
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='b', facecolor='none')
            else:
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')

            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f'{label}: {score:.2f}', fontsize=10, color='g')
    plt.savefig(os.path.join(output_dir, image_name), dpi=300, bbox_inches='tight', pad_inches=0)
    return
