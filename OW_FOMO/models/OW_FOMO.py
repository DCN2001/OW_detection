# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection
# Orr Zohar, Jackson Wang, Serena Yeung
# ------------------------------------------------------------------------
# Modified from Transformers: 
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py
# ------------------------------------------------------------------------

from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTConfig, OwlViTModel
from transformers.models.owlvit.modeling_owlvit import *
from transformers.models.clip.modeling_clip import _expand_mask

from .utils import *
from .few_shot_dataset import FewShotDataset, aug_pipeline, collate_fn

from util import box_ops
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from torch.utils.data import DataLoader
# from torchvision.ops import sigmoid_focal_loss, generalized_box_iou
from util.box_ops import generalized_box_iou
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json

import warnings
import time

from models.matcher import build_matcher
from copy import deepcopy

# Splits a list into chunks of bacth size.
def split_into_chunks(lst, batch_size):
    # Return a list of chunks, each chunk is a sublist of the original list. 
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

# For classification using various methods and handling unknown dimensions, return with logits and objectness
class UnkDetHead(nn.Module):
    # Does classification of 
    def __init__(self, method, known_dims, att_W, **kwargs):
        super(UnkDetHead, self).__init__()
        # print("UnkDetHead", method)
        self.method = method            # Method used for processing logits
        self.known_dims = known_dims    # Number of known dimensions
        self.att_W = att_W              # Attention weights
        self.process_mcm = nn.Softmax(dim=-1)   # Softmax for mcm method

        # Check the method type and set up the corresponding process_logits function
        if "sigmoid" in method:
            self.process_logits = nn.Sigmoid()
            self.proc_obj = True
        elif "softmax" in method:
            self.process_logits = nn.Softmax(dim=-1)
            self.proc_obj = True
        else:
            self.proc_obj = False

    def forward(self, att_logits):
        # Compute logits by multiplying attention logits with attention weights
        logits = att_logits @ self.att_W
        # Separate known and unknown logits
        k_logits = logits[..., :self.known_dims]
        # unk_logits = logits[..., self.known_dims:].max(dim=-1, keepdim=True)[0]
        
        # print(f"att_logits.shape: {att_logits.shape}")
        # print(f"self.att_W.shape: {self.att_W.shape}")
        # print(f"logits.shape: {logits.shape}")
        # print(f"k_logits.shape: {k_logits.shape}")

        # Concatenate known logits and max of unknown logits
        #logits = torch.cat([k_logits, unk_logits], dim=-1)
        #objectness = torch.ones_like(unk_logits).squeeze(-1)
        
        # Process logits
        if "mean" in self.method:
            sm_logits = self.process_logits(att_logits)
            objectness = sm_logits.mean(dim=-1, keepdim=True)[0]

        elif "max" in self.method:
            sm_logits = self.process_logits(att_logits)
            objectness = sm_logits.max(dim=-1, keepdim=True)[0]

        # Modify objectness (mcm method)
        if "mcm" in self.method:
            mcm = self.process_mcm(k_logits).max(dim=-1, keepdim=True)[0]
            objectness *= (1 - mcm)

        # Normalize objectness if necessary
        if self.proc_obj:
            objectness -= objectness.mean()
            objectness /= objectness.std()
            objectness = torch.sigmoid(objectness)
        
        # print(f"logits.shape: {logits.shape}")
        # print(f"objectness.squeeze(-1).shape: {objectness.squeeze(-1).shape}")
        # print(f"objectness: {objectness}")
        
        # Return logits and objectness
        return logits, objectness.squeeze(-1)  # logits are raw (not softmaxxed)


class OwlViTTextTransformer(OwlViTTextTransformer):
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        num_samples, seq_len = input_shape  # num_samples = batch_size * num_max_text_queries
        # OWLVIT's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(num_samples, seq_len).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [num_samples, seq_len] -> [num_samples, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # take features from the end of tokens embedding (end of token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len)
        mask.fill_(torch.tensor(float("-inf")))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


@add_start_docstrings(OWLVIT_START_DOCSTRING)
class OurOwlViTModel(OwlViTModel):
    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTOutput, config_class=OwlViTConfig)
    def forward_vision(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
        """
        # print(f"In OurOwlVitModel: pixel_values: {pixel_values}")
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get embeddings for all text queries in all batch samples
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        return image_embeds, vision_outputs

    def forward_text(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            """

        # Get embeddings for all text queries in all batch samples
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        text_embeds_norm = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        return text_embeds_norm, text_outputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_base_image_embeds: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, OwlViTOutput]:
        r"""
        Returns:
            """
        # Use OWL-ViT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # normalized features
        image_embeds, vision_outputs = self.forward_vision(pixel_values=pixel_values,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)

        text_embeds_norm, text_outputs = self.forward_text(input_ids=input_ids, attention_mask=attention_mask,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)

        # cosine similarity as logits and set it on the correct device
        logit_scale = self.logit_scale.exp().to(image_embeds.device)

        logits_per_text = torch.matmul(text_embeds_norm, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = owlvit_loss(logits_per_text)

        if return_base_image_embeds:
            warnings.warn(
                "`return_base_image_embeds` is deprecated and will be removed in v4.27 of Transformers, one can"
                " obtain the base (unprojected) image embeddings from outputs.vision_model_output.",
                FutureWarning,
            )
            last_hidden_state = vision_outputs[0]
            image_embeds = self.vision_model.post_layernorm(last_hidden_state)
        else:
            text_embeds = text_embeds_norm

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return OwlViTOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

# Main model
class OWFOMO(nn.Module):
    """This is the OWL-ViT model that performs open-vocabulary object detection"""
    def __init__(self, args, model_name, known_class_names, unknown_class_names, templates, image_conditioned, device):
        """ Initializes the model.
        Parameters:
            model_name: the name of the huggingface model to use
            known_class_names: list of the known class names
            templates:
            attributes: dict of class names (keys) and the corresponding attributes (values).
        """
        super().__init__()   
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)    # Model for object detection
        original_state_dict = self.model.state_dict()                                   
        self.model.owlvit = OurOwlViTModel.from_pretrained(model_name).to(device)       # Load OwlViT model
        filtered_state_dict = {k: v for k, v in original_state_dict.items() if k in self.model.owlvit.state_dict()}
        # assert "box_head.dense1.bias" in filtered_state_dict.keys(), "[box_head.dense1.bias] not in filtered_state_dict.keys()"
        self.model.owlvit.load_state_dict(filtered_state_dict, strict=False)
        self.processor = OwlViTProcessor.from_pretrained(model_name)                    # Initialize the processor
        self.known_class_name = known_class_names
        self.unknown_class_name = unknown_class_names
        self.device = device
        self.mode = None

        '''
        print("\n\nPrinting box related parameters of self.model...")
        for name, param in self.model.named_parameters():
            if "box" in name:
                print(f"=================================")
                print(f"Parameter name: {name}")
                print(f"Parameter shape: {param.shape}")
                print(f"Parameter value: {param.data}")

        print("\n\nPrinting box related parameters of self.model.owlvit...")
        for name, param in self.model.owlvit.named_parameters():
            if "box" in name:
                print(f"=================================")
                print(f"Parameter name: {name}")
                print(f"Parameter shape: {param.shape}")
                print(f"Parameter value: {param.data}")
        '''

        # Attribute refinement setup
        # Load attribute data from file
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.attributes_file}', 'r') as f:
            attributes = json.loads(f.read())
        self.attributes_texts = [f"object which (is/has/etc) {cat} is {a}" for cat, att in attributes.items() for a in att]     # Format attribute texts
        self.att_W = torch.rand(len(self.attributes_texts), len(known_class_names), device=device)      # Initialize attribute weights
        with torch.no_grad():
            # mean_known_query_embeds, embeds_dataset = self.get_mean_embeddings(fs_dataset)
            text_mean_norm, att_query_mask = self.prompt_template_ensembling(self.attributes_texts, templates)      # Generate mean normalized text embeddings and attention masks for the attributes
            self.att_embeds = text_mean_norm.detach().clone().to(device)        # Store normalized text embeddings
            self.att_query_mask = att_query_mask.to(device)          # Store attention masks

        self.unk_head = UnkDetHead(args.unk_method, known_dims=len(known_class_names),
                                   att_W=self.att_W, device=device)         # Initialize unknown detection head
    
    # Compute mean embeddings for the known classes in the dataset.
    def get_mean_embeddings(self, fs_dataset):
        dataset = {i: [] for i in range(len(self.known_class_names))}       # Initialize a dictionary to collect embeddings
        for img_batch in split_into_chunks(range(len(fs_dataset)), 3):
            image_batch = collate_fn([fs_dataset.get_no_aug(i) for i in img_batch])     # Get batch of images without augmentation
            grouped_data = defaultdict(list)                                # Group data by class

            # Group data by label
            for bbox, label, image in zip(image_batch['bbox'], image_batch['label'], image_batch['image']):
                grouped_data[label].append({'bbox': bbox, 'image': image})

            for l, data in grouped_data.items():
                tmp = self.image_guided_forward(torch.stack([d["image"] for d in data]).to(self.device),
                                                [d["bbox"] for d in data]).cpu()
                dataset[l].append(tmp)      # Store embeddings for each class

        # Compute mean embeddings for each class and return them
        return torch.cat([torch.cat(dataset[i], 0).mean(0) for i in range(len(self.known_class_names))], 0).unsqueeze(
            0).to(self.device), dataset
    
    # Ensemble text prompts for each class using the provided templates.
    def prompt_template_ensembling(self, classnames, templates):
        # print('performing prompt ensembling')
        text_sum = torch.zeros((1, len(classnames), self.model.owlvit.text_embed_dim)).to(self.device)      # Initialize tensor to accumulate text embeddings

        for template in templates:
            print('Adding template:', template)
            class_texts = [template.replace('{c}', classname.replace('_', ' ')) for classname in
                           classnames]                                                                      # Generate text for each class using the template

            text_tokens = self.processor(text=class_texts, return_tensors="pt", padding=True, truncation=True).to(      #Tokenize the text inputs
                self.device)

            text_tensor, query_mask = self.forward_text(**text_tokens)                                      # Forward pass through the text encoder

            text_sum += text_tensor

        # Calculate mean of text embeddings
        # text_mean = text_sum / text_count
        text_norm = text_sum / torch.linalg.norm(text_sum, ord=2, dim=-1, keepdim=True) + 1e-6              # Normalize with an esp to avoid division by zero
        return text_norm, query_mask
    
    # Forward pass through the text encoder of the model.
    def forward_text(
            self,
            input_ids,
            attention_mask,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None, ):

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions     # Set output attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states)           # Set output hidden states
        return_dict = return_dict if return_dict is not None else self.model.config.return_dict           # Set return dictionary flag

        # Forward pass through the model's text encoder
        text_embeds, text_outputs = self.model.owlvit.forward_text(input_ids=input_ids, attention_mask=attention_mask,
                                                                   output_attentions=output_attentions,
                                                                   output_hidden_states=output_hidden_states,
                                                                   return_dict=return_dict)

        text_embeds = text_embeds.unsqueeze(0)  #Add a batch dimension

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.unsqueeze(0)
        query_mask = input_ids[..., 0] > 0      # Create a query mask to identify valid queries

        return text_embeds.to(self.device), query_mask.to(self.device)
    
    # Embed image query features for object detection.
    def embed_image_query(
            self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor,
            each_query_boxes, masks) -> torch.FloatTensor:
        
        _, class_embeds = self.model.class_predictor(query_image_features)      # Get class embeddings from query features
        # Get boxes from query embeddings here
        pred_boxes = self.model.box_predictor(query_image_features, query_feature_map)

        pred_boxes_as_corners = box_ops.box_cxcywh_to_xyxy(pred_boxes)  # bs, number_of_objects, 4
        pred_boxes_device = pred_boxes_as_corners.device        # Device of predicted boxes
        
        # Define three list to store embeddings, matched box indexes and unmatched ground truth indexes for each batch
        batch_query_embeds = []
        batch_matched_box_indexes = []
        batch_unmatched_gt_indexes = []

        max_batch_objects = each_query_boxes.shape[1]
        
        # Matching is done over here, Loop over query images
        for i in range(query_image_features.shape[0]):
            # Define three list to store: Best class embeddings and indices for each query image. And indexes of GT labels without matches.
            best_class_embeds = []
            best_box_indices = []
            bad_indexes = []
            each_query_box = torch.tensor(each_query_boxes[i], device=pred_boxes_device)    # GT boxes for current image
            each_image_mask = torch.tensor(masks[i], device=pred_boxes_device)  # Mask
            each_query_pred_boxes = pred_boxes_as_corners[i]    # Predicted boxes 
            ious, _ = box_iou(box_ops.box_cxcywh_to_xyxy(each_query_box[each_image_mask>0]), each_query_pred_boxes) # (GT boxes, prediceted boxes)
            
            # print(f"====================================================")
            # print(f"In OW_FOMO.py: each_query_box.shape: {each_query_box.shape}")
            # print(f"In OW_FOMO.py: each_query_box[each_image_mask].shape: {each_query_box[each_image_mask>0].shape}")
            # print(f"In OW_FOMO.py: each_query_pred_boxes.shape: {each_query_pred_boxes.shape}")
            # print(f"In OW_FOMO.py: ious.shape: {ious.shape}")
            print(f">>>>>>>In image {i}...<<<<<<<<")
            
            # If there are no overlapping boxes, fall back to generalized IoU
            if torch.all(ious[0] == 0.0):
                print(f"generalized_box_iou!!!!!!!!!!!!!!!!!!!!!!!!")
                #print(f"each_query_box[each_image_mask>0]: {each_query_box[each_image_mask>0]}")
                ious = generalized_box_iou(box_ops.box_cxcywh_to_xyxy(each_query_box[each_image_mask>0]), each_query_pred_boxes)
            #print(f"ious.shape: {ious.shape}")

            # Use an adaptive threshold to include all boxes within 80% of the best IoU
            iou_threshold = torch.max(ious) * 0.5
            #print(f"ious: {ious}")
            #print(f"iou_threshold: {iou_threshold}")

            for gt_idx in range(ious.shape[0]):
                selected_inds = (ious[gt_idx] >= iou_threshold).nonzero()       # Select indices with IoU above threshold
                # print(f"In OW_FOMO.py: selected_inds.shape: {selected_inds.shape}")
                
                 # If there are valid indices
                if selected_inds.numel():
                    selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                    mean_embeds = torch.mean(class_embeds[i], axis=0)
                    mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)    # Compute similarity
                    best_box_ind = selected_inds[torch.argmin(mean_sim)]            # Select best box based on similarity
                    best_class_embeds.append(class_embeds[i][best_box_ind].squeeze(0))
                    # print(f'best_box_ind: {best_box_ind}')
                    best_box_indices.append(best_box_ind)
                else:
                    best_class_embeds.append(torch.zeros_like(class_embeds[i][0]).to(self.device))  # No match found, use zero vector
                    best_box_indices.append(torch.Tensor([-1]).to(self.device))         # Invalid index
                    bad_indexes.append(gt_idx)  # Indexes of GT labels that do not have any matched queries
            
            # print(f"bad_indexes before padding: {bad_indexes}")
            for _ in range(max_batch_objects-ious.shape[0]):
                best_class_embeds.append(torch.zeros_like(class_embeds[i][0]).to(self.device))
                best_box_indices.append(torch.Tensor([-1]).to(self.device))
            for _ in range(max_batch_objects-len(bad_indexes)):
                bad_indexes.append(-1)

            if best_class_embeds:
                query_embeds = torch.stack(best_class_embeds)  # Best match embeddings of all GT labels
                box_indices = torch.stack(best_box_indices)  # Best match indexes of all GT labels (indexes according to query embedding)
                bad_indexes = torch.Tensor(bad_indexes).to(self.device)
                
                # print(f"best_class_embeds.shape: {query_embeds.shape}")
                # print(f"best_box_indices: {box_indices}")
                # print(f"bad_indexes: {bad_indexes}")
            else:
                query_embeds, box_indices = None, None

            batch_query_embeds.append(query_embeds)
            batch_matched_box_indexes.append(box_indices)  # list[tensor(number_of_objects,1)], len(list) = bs 
            batch_unmatched_gt_indexes.append(bad_indexes)
        
        batch_query_embeds = torch.stack(batch_query_embeds)
        batch_matched_box_indexes = torch.stack(batch_matched_box_indexes)      # Set matched box indices(0 ~ B-1)
        batch_unmatched_gt_indexes = torch.stack(batch_unmatched_gt_indexes)    # Set unmatched GT indexes(0 ~ A-1)

        # matched_box_indices = box_indices   # 0 ~ B-1
        # unmatched_gt_indexes = bad_indexes  # 0 ~ A-1
        # return query_embeds, matched_box_indices, pred_boxes, unmatched_gt_indexes
        print(f"batch_query_embeds.shape: {batch_query_embeds.shape}")
        print(f"batch_matched_box_indexes.shape: {batch_matched_box_indexes.shape}")
        print(f"pred_boxes.shape: {pred_boxes.shape}")
        print(f"batch_unmatched_gt_index.shape: {batch_unmatched_gt_indexes.shape}")
        return batch_query_embeds, batch_matched_box_indexes, pred_boxes, batch_unmatched_gt_indexes


    # Forward pass for image-guided object detection.
    def image_guided_forward(self,query_pixel_values: Optional[torch.FloatTensor] = None, bboxes=None, cls=None, masks=None):
        # Compute feature maps for the input and query images
        # save_tensor_as_image_with_bbox(query_pixel_values[0].cpu(), bboxes[0][0], f'tmp/viz/{cls}_img.png')
        with torch.no_grad():
            query_feature_map = self.model.image_embedder(pixel_values=query_pixel_values)[0]
            batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
            query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))   # Reshape feature map
            # Get top class embedding and best box index for each query image in batch
        query_embeds, matched_box_indices, pred_boxes, unmatched_gt_indexes = self.embed_image_query(query_image_feats, query_feature_map, bboxes, masks)
    
        if query_embeds is None:
            return None, None
        query_embeds_dup = query_embeds/torch.linalg.norm(query_embeds, ord=2, dim=-1, keepdim=True) + 1e-6     # Normalize and add with an eps
        if cls is not None:
            # print(f"cls: {cls}")
            cls_list = []
            for image in range(cls.shape[0]):
                image_cls_tensor = torch.zeros(cls.shape[1]).to(self.device)
                for idx, elm in enumerate(image_cls_tensor):
                    if masks[image][idx] > 0 and idx not in unmatched_gt_indexes[image]:
                        image_cls_tensor[idx] = cls[image][idx]
                cls_list.append(image_cls_tensor)
            # return query_embeds, [item for index, item in enumerate(cls) if index not in unmatched_gt_indexes], matched_box_indices, pred_boxes, unmatched_gt_indexes
            return query_embeds, cls_list, matched_box_indices, pred_boxes, unmatched_gt_indexes
        return query_embeds


    # Attribute refinement for classification part of object detection
    def attribute_refinement(self, batch):
        # for i in range(len(batch[1])):
        #     print(f"batch[0].tensors[i].shape[1:]: {batch[0].tensors[i].shape[1:]}")
        #     print(f"batch[1][i]['boxes']: {batch[1][i]['boxes']}")
        #     batch[1][i]["boxes"] = self.box_resize(batch[0].tensors[i].shape[1:],batch[1][i]["boxes"])

        # Define three lsit to store box tensors, label tensors, mask tensors 
        batch_boxes = []
        batch_labels = []
        batch_masks = []
        batch_boxes_tensor = None
        batch_labels_tensor = None
        batch_masks_tensor = None
        with torch.no_grad():
            for image in batch[1]:
                image_boxes_tensor = image["boxes"]     # Extract boxes tensor
                image_labels_tensor = image["labels"]   # Extract labels tensor
                image_mask_tensor = image["mask"]       # Extract masks tensor
                batch_boxes.append(image_boxes_tensor)
                batch_labels.append(image_labels_tensor)
                batch_masks.append(image_mask_tensor)
            
            # Stack the tensors together
            batch_boxes_tensor = torch.stack(batch_boxes)
            batch_labels_tensor = torch.stack(batch_labels)
            batch_masks_tensor = torch.stack(batch_masks)

        # print(f"batch_boxes_tensor.shape: {batch_boxes_tensor.shape}")
        # print(f"batch_labels_tensor.shape: {batch_labels_tensor.shape}")

        image_embeds, target, matched_box_indices, pred_boxes, unmatched_gt_indexes = self.image_guided_forward(batch[0].tensors.to(self.device),bboxes=batch_boxes_tensor,cls=batch_labels_tensor, masks=batch_masks_tensor)
        
        with torch.no_grad():
            target_AttEmbed = torch.stack(target).to(self.device)       # Stack target attributes embeddings
            target_AttEmbed = F.one_hot(target_AttEmbed.to(torch.int64), num_classes=len(self.known_class_name)).to(self.device) # Convert target attributes embeddings to onehot
            target_AttEmbed = target_AttEmbed.float()
            # print(f"In OW_FOMO.py: matched_box_indices.shape: {pred_boxes.shape}")
            # print(f"In OW_FOMO.py: pred_boxes.shape: {pred_boxes.shape}")
            # print(f"torch.max(target_AttEmbed): {torch.max(target_AttEmbed)}")
            # time.sleep(3)
            # print(f"In OW_FOMO.py: target_AttEmbed: {target_AttEmbed}")
            # print(f"In OW_FOMO.py: image_embeds.shape: {image_embeds.shape}")
            # print(f"In OW_FOMO.py: self.att_embeds.shape: {self.att_embeds.shape}")
        cos_sim = []
        for idx in range(image_embeds.shape[0]):
            # cos_sim_image = cosine_similarity(image_embeds[idx], self.att_embeds[0], dim=-1)
            # print(f"self.att_embeds.shape: {self.att_embeds.shape}")
            # print(f"image_embeds[idx].shape: {image_embeds[idx].shape}")
            # print(f"self.att_embeds[0].shape: {self.att_embeds[0].shape}")
            
            assert 0 <= idx < len(image_embeds), f"Index {idx} out of bounds"
            print(f"image_embeds[idx].shape: {image_embeds[idx].shape}")
            print(f"self.att_embeds[0].shape: {self.att_embeds[0].shape}")
            cos_sim_image = F.normalize(image_embeds[idx]) @ F.normalize(self.att_embeds[0]).t()    # Compute cosine similarity
            cos_sim.append(cos_sim_image)
        
        cos_sim = torch.stack(cos_sim)      # Accumulate cosine similarity
        output_AttEmbed, output_objectness = self.unk_head(cos_sim) # Get attribute embeddings and objectness score
        
        # print(f"image_embeds.requires_grad: {image_embeds.requires_grad}")
        # print(f"self.att_embeds.requires_grad: {self.att_embeds.requires_grad}")
        # print(f"cos_sim.requires_grad: {cos_sim.requires_grad}")
        # print(f"In OW_FOMO.py: cos_sim.shape: {cos_sim.shape}")
        # cos_sim = cosine_similarity(image_embeds, self.att_embeds, dim=-1)
        # print(f"self.unk_head.requires_grad: {self.unk_head.requires_grad}")
        # print(f"output_AttEmbed.requires_grad: {output_AttEmbed.requires_grad}")
        #output_AttEmbed = torch.matmul(cos_sim, self.att_W)
        
        target_boxes = batch_boxes_tensor
        target_masks = batch_masks_tensor
        return target_AttEmbed, target_boxes, target_masks, output_AttEmbed, pred_boxes, output_objectness, matched_box_indices, unmatched_gt_indexes
    
    # Attribute selection for object detection (Simillar to attribute refinement, only difference in updated parameters by optimizer)
    def attribute_selection(self, batch):
        # for i in range(len(batch[1])):
        #     print(f"batch[0].tensors[i].shape[1:]: {batch[0].tensors[i].shape[1:]}")
        #     print(f"batch[1][i]['boxes']: {batch[1][i]['boxes']}")
        #     batch[1][i]["boxes"] = self.box_resize(batch[0].tensors[i].shape[1:],batch[1][i]["boxes"])
        batch_boxes = []
        batch_labels = []
        batch_masks = []
        batch_boxes_tensor = None
        batch_labels_tensor = None
        batch_masks_tensor = None
        with torch.no_grad():
            for image in batch[1]:
                image_boxes_tensor = image["boxes"]
                image_labels_tensor = image["labels"]
                image_mask_tensor = image["mask"]
                batch_boxes.append(image_boxes_tensor)
                batch_labels.append(image_labels_tensor)
                batch_masks.append(image_mask_tensor)
            batch_boxes_tensor = torch.stack(batch_boxes)
            batch_labels_tensor = torch.stack(batch_labels)
            batch_masks_tensor = torch.stack(batch_masks)

        # print(f"batch_boxes_tensor.shape: {batch_boxes_tensor.shape}")
        # print(f"batch_labels_tensor.shape: {batch_labels_tensor.shape}")

        image_embeds, target, matched_box_indices, pred_boxes, unmatched_gt_indexes = self.image_guided_forward(batch[0].tensors.to(self.device),bboxes=batch_boxes_tensor,cls=batch_labels_tensor, masks=batch_masks_tensor)
        
        with torch.no_grad():
            # print(f"In OW_FOMO.py: matched_box_indices.shape: {pred_boxes.shape}")
            # print(f"In OW_FOMO.py: pred_boxes.shape: {pred_boxes.shape}")
            target_AttEmbed = torch.stack(target).to(self.device)
            # print(f"torch.max(target_AttEmbed): {torch.max(target_AttEmbed)}")
            # time.sleep(3)
            target_AttEmbed = F.one_hot(target_AttEmbed.to(torch.int64), num_classes=len(self.known_class_name)).to(self.device)
            target_AttEmbed = target_AttEmbed.float()
            # print(f"In OW_FOMO.py: target_AttEmbed: {target_AttEmbed}")
            # print(f"In OW_FOMO.py: image_embeds.shape: {image_embeds.shape}")
            # print(f"In OW_FOMO.py: self.att_embeds.shape: {self.att_embeds.shape}")
        cos_sim = []
        for idx in range(image_embeds.shape[0]):
            # cos_sim_image = cosine_similarity(image_embeds[idx], self.att_embeds[0], dim=-1)
            # print(f"self.att_embeds.shape: {self.att_embeds.shape}")
            # print(f"image_embeds[idx].shape: {image_embeds[idx].shape}")
            # print(f"self.att_embeds[0].shape: {self.att_embeds[0].shape}")
            assert 0 <= idx < len(image_embeds), f"Index {idx} out of bounds"
            cos_sim_image = F.normalize(image_embeds[idx]) @ F.normalize(self.att_embeds[0]).t()
            cos_sim.append(cos_sim_image)
        # print(f"image_embeds.requires_grad: {image_embeds.requires_grad}")
        # print(f"self.att_embeds.requires_grad: {self.att_embeds.requires_grad}")
        cos_sim = torch.stack(cos_sim)
        # print(f"cos_sim.requires_grad: {cos_sim.requires_grad}")
        # print(f"In OW_FOMO.py: cos_sim.shape: {cos_sim.shape}")
        # cos_sim = cosine_similarity(image_embeds, self.att_embeds, dim=-1)
        output_AttEmbed, output_objectness = self.unk_head(cos_sim)
        # print(f"self.unk_head.requires_grad: {self.unk_head.requires_grad}")
        # print(f"output_AttEmbed.requires_grad: {output_AttEmbed.requires_grad}")
        #output_AttEmbed = torch.matmul(cos_sim, self.att_W)
        target_boxes = batch_boxes_tensor
        target_masks = batch_masks_tensor
        return target_AttEmbed, target_boxes, target_masks, output_AttEmbed, pred_boxes, output_objectness, matched_box_indices, unmatched_gt_indexes

    def box_resize(img_shape, bbs):
        annotations = []

        for bounding_box in bbs:
            annotation = [
                bounding_box[0] / img_shape[0],
                bounding_box[1] / img_shape[1],
                bounding_box[2] / img_shape[0],
                bounding_box[3] / img_shape[1]
            ]
            annotation = torch.FloatTensor(annotation)
            annotations.append(annotation)
        return annotations

    def forward(self, batch, 
            pixel_values: torch.FloatTensor = 0, 
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None):
        # takes batch
        targets = {}
        outputs = {}    # keys: 'att_embed' 

        #For estimating loss of attribute_embedding
        if self.mode=="attribute_refinement":
            targets['logits'], targets['boxes'], targets["masks"], outputs['logits'], outputs['boxes'], outputs['objectness'], matched_indices, unmatched_gt_indexes = self.attribute_refinement(batch)
            return targets, outputs, matched_indices, unmatched_gt_indexes
        
        #For estimating loss of attribute_weight matrix
        elif self.mode=="attribute_selection":
            targets['logits'], targets['boxes'], targets["masks"], outputs['logits'], outputs['boxes'], outputs['objectness'], matched_indices, unmatched_gt_indexes = self.attribute_selection(batch)
            return targets, outputs, matched_indices, unmatched_gt_indexes
        
        # Inference mode
        elif self.mode=="inference":
            # Set configurations
            output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.model.config.return_dict

            # print(f"In OWFOMO.forward: pixel_values: {pixel_values}")
            # Embed images and text queries
            _, vision_outputs = self.model.owlvit.forward_vision(pixel_values=pixel_values,
                                                                output_attentions=output_attentions,
                                                                output_hidden_states=output_hidden_states,
                                                                return_dict=return_dict)

            # Get image embeddings
            last_hidden_state = vision_outputs[0]
            image_embeds = self.model.owlvit.vision_model.post_layernorm(last_hidden_state)

            # Resize class token
            new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
            class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

            # Merge image embedding with class tokens
            image_embeds = image_embeds[:, 1:, :] * class_token_out
            image_embeds = self.model.layer_norm(image_embeds)

            # Resize to [batch_size, num_patches, num_patches, hidden_size]
            new_size = (
                image_embeds.shape[0],
                int(np.sqrt(image_embeds.shape[1])),
                int(np.sqrt(image_embeds.shape[1])),
                image_embeds.shape[-1],
            )

            image_embeds = image_embeds.reshape(new_size)

            batch_size, num_patches, num_patches, hidden_dim = image_embeds.shape
            image_feats = torch.reshape(image_embeds, (batch_size, num_patches * num_patches, hidden_dim))

            # Predict object boxes
            pred_boxes = self.model.box_predictor(image_feats, image_embeds)

            # Predict logits and class embeddings
            (pred_logits, class_embeds) = self.model.class_predictor(image_feats, self.att_embeds.repeat(batch_size, 1, 1),
                                                        self.att_query_mask)

            # Create an output object
            out = OwlViTObjectDetectionOutput(
                image_embeds=image_embeds,
                text_embeds=self.att_embeds,
                pred_boxes=pred_boxes,
                logits=pred_logits,
                class_embeds=class_embeds,
                vision_model_output=vision_outputs,
            )

            # Cosine similarity with attribute embeddings
            cos_sim = []
            for idx in range(class_embeds.shape[0]):
                assert 0 <= idx < len(class_embeds), f"Index {idx} out of bounds"
                cos_sim_image = F.normalize(class_embeds[idx]) @ F.normalize(self.att_embeds[0]).t()
                cos_sim.append(cos_sim_image)
            cos_sim = torch.stack(cos_sim)
            
            # Feed through unk_head
            out.att_logits = cos_sim  
            out.logits, out.obj = self.unk_head(cos_sim)
            return out

# Estimate loss for box head
class SetCriterionBox(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.unmatched_boxes = args.unmatched_boxes
        self.l1_weight = 1.0
        self.giou_weight = 1.0
        self.fn_penalty = 1.0

    def forward(self, outputs, targets, false_negatives):
        for t in targets:
            if t.tolist()[0] == 0 and t.tolist()[1] == 0 and t.tolist()[2] == 0 and t.tolist()[3] == 0:
                print(f"No valid boxes for this image")
                return torch.tensor(0, dtype=torch.float32).to(outputs.device)
        # print(f"outputs_boxes: {outputs}")
        # print(f"targets_boxes: {targets}")
        if outputs.shape[0] == 0:
            return 0
        
        # Estimate L1 loss first 
        loss_bbox = F.l1_loss(outputs, targets, reduction='sum')
        
        # Compute the generalized intersection over union (GIoU) loss
        # Convert boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format => Compute the GIoU and take the diagonal elements (correspond to the matched boxes)
        loss_giou = 1 - torch.mean(torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(outputs),
        box_ops.box_cxcywh_to_xyxy(targets))))

        # print(f"loss_bbox: {loss_bbox}")
        # print(f"loss_giou: {loss_giou}")
        # print(f"false_negatives: {false_negatives}")

        # Total loss
        tot_loss = (1.0 * loss_bbox + 1.0 * loss_giou + 2.0 * false_negatives) / (targets.shape[0] + false_negatives)
        return tot_loss


# Estimate loss for att_embed and att_W (Per Batch) 
class SetCriterionAttr(nn.Module):
    def __init__(self):
        super().__init__()
        self.est_loss = nn.BCEWithLogitsLoss()
    def forward(self, outputs, targets):
        # Calculate Binary cross entropy loss 
        loss = self.est_loss(outputs, targets)

        #print(f"In attr_loss calculation: outputs.shape = {outputs.shape}")
        #print(f"In attr_loss calculation: outputs = {outputs}")
        #print(f"In attr_loss calculation: targets.shape = {targets.shape}")
        #print(f"In attr_loss calculation: targets = {targets}")
        #print(f"In attr_loss calculation: loss = {loss}")
        return loss


'''
class SetCriterionObjecness(nn.Module):
    def __init__(self, ):
'''


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, model_name, pred_per_im=100, image_resize=768, device='cpu', method='regular'):
        super().__init__()
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.pred_per_im = pred_per_im
        self.method=method
        self.image_resize = image_resize
        self.device = device
        self.clip_boxes = lambda x, y: torch.cat(
            [x[:, 0].clamp_(min=0, max=y[1]).unsqueeze(1),
             x[:, 1].clamp_(min=0, max=y[0]).unsqueeze(1),
             x[:, 2].clamp_(min=0, max=y[1]).unsqueeze(1),
             x[:, 3].clamp_(min=0, max=y[0]).unsqueeze(1)], dim=1)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, viz=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        if viz:
            reshape_sizes = torch.Tensor([[self.image_resize, self.image_resize]]).repeat(len(target_sizes), 1)
            target_sizes = (target_sizes * self.image_resize / target_sizes.max(1, keepdim=True).values).long()
        else:
            max_values, _ = torch.max(target_sizes, dim=1)
            reshape_sizes = max_values.unsqueeze(1).repeat(1, 2)

        if self.method =="regular":
            results = self.post_process_object_detection(outputs=outputs, target_sizes=reshape_sizes)
        elif self.method == "attributes":
            results = self.post_process_object_detection_att(outputs=outputs, target_sizes=reshape_sizes)
        elif self.method == "seperated":
            results = self.post_process_object_detection_seperated(outputs=outputs, target_sizes=reshape_sizes)

        for i in range(len(results)):
            results[i]['boxes'] = self.clip_boxes(results[i]['boxes'], target_sizes[i])
        return results

    def post_process_object_detection(self, outputs, target_sizes=None):
        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, logits, boxes):
            scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
            topk_boxes = topk_indexes // logits.shape[2]
            labels = topk_indexes % logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob, logits, boxes)
        return results

    def post_process_object_detection_att(self, outputs, target_sizes=None):
        ## this post processing should produce the same predictions as `post_process_object_detection`
        ## but also report what are the most dominant attribute per class (used to produce some of the
        ## figures in the MS
        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob_att = torch.sigmoid(outputs.att_logits)
        
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
        # check prob values. why is scores always 0
        #print(f"logits: {logits}")
        #print(f"max value of logits: {torch.max(logits)}")
        #print(f"obj: {obj}")
        #print(f"max value of obj: {torch.max(obj)}")
        #print(f"prob: {prob}")
        #print(f"max value of prob: {torch.max(prob)}")
        def get_known_objs(prob, logits, prob_att, boxes):
            #print(f"prob.shape: {prob.shape}")
            #print(f"prob.view(logits.shape[0], -1).shape: {prob.view(logits.shape[0], -1).shape}")
            #print(f"logits.shape: {logits.shape}")
            scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
            topk_boxes = topk_indexes // logits.shape[2]
            labels = topk_indexes % logits.shape[2]

            # Get the batch indices and prediction indices to index into prob_att
            batch_indices = torch.arange(logits.shape[0]).view(-1, 1).expand_as(topk_indexes)
            pred_indices = topk_boxes

            # Gather the attributes corresponding to the top-k labels
            # You will gather along the prediction dimension (dim=1)
            gathered_attributes = prob_att[batch_indices, pred_indices, :]

            # Now gather the boxes in a similar way as before
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            # Combine the results into a list of dictionaries
            return [{'scores': s, 'labels': l, 'boxes': b, 'attributes': a} for s, l, b, a in zip(scores, labels, boxes, gathered_attributes)]
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob, logits, prob_att, boxes)
        return results

    def post_process_object_detection_seperated(self, outputs, target_sizes=None):
        ## predicts the known and unknown objects seperately. Used when the known and unknown classes are
        ## derived one from text and the other from images.

        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj.squeeze(-1)

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, out_logits, boxes):
            # import ipdb; ipdb.set_trace()
            scores, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.pred_per_im//2, dim=1)
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        def get_unknown_objs(obj, out_logits, boxes):

            scores, topk_indexes = torch.topk(obj.unsqueeze(-1), self.pred_per_im//2, dim=1)
            scores = scores.squeeze(-1)
            labels = torch.ones(scores.shape, device=scores.device) * out_logits.shape[-1]
            # import ipdb; ipdb.set_trace()
            boxes = torch.gather(boxes, 1, topk_indexes.repeat(1, 1, 4))
            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob[..., :-1].clone(), logits[..., :-1].clone(), boxes)
        unknown_results = get_unknown_objs(prob[..., -1].clone(), logits[..., :-1].clone(), boxes)

        out = []
        for k, u in zip(results, unknown_results):
            out.append({
                "scores": torch.cat([k["scores"], u["scores"]]),
                "labels": torch.cat([k["labels"], u["labels"]]),
                "boxes": torch.cat([k["boxes"], u["boxes"]])
            })
        return out

# Entire pipeline, return with two class Model and Postprocessor
def build(args):
    device = torch.device(args.device)      # Device setting

    # Read the list of all known class names from the specified file
    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.classnames_file}', 'r') as file:
        ALL_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

    # Read the list of previously known class names from the specified file
    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.prev_classnames_file}', 'r') as file:
        PREV_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

    # Determine the current known class names by excluding previous known class names from all known class names
    CUR_KNOWN_ClASSNAMES = [cls for cls in ALL_KNOWN_CLASS_NAMES if cls not in PREV_KNOWN_CLASS_NAMES]
    known_class_names = PREV_KNOWN_CLASS_NAMES + CUR_KNOWN_ClASSNAMES

    # If the unknown proposal flag is set and the unknown class names file is specified, read the unknown class names
    if args.unk_proposal and args.unknown_classnames_file != "None":
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.unknown_classnames_file}', 'r') as file:
            unknown_class_names = sorted(file.read().splitlines())
        unknown_class_names = [k for k in unknown_class_names if k not in known_class_names]    # Exclude known class names from the list of unknown class names
        unknown_class_names = [c.replace('_', ' ') for c in unknown_class_names]        # Replace underscores with spaces in the unknown class names
    # Single "object" class if no unknown class names file is specified
    else:
        unknown_class_names = ["object"]

    #If a templates file is specified, read the templates
    if args.templates_file:
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.templates_file}', 'r') as file:
            templates = file.read().splitlines()
    #Default to a generic template if no templates file is specified
    else:
        templates = ["a photo of a {c}"]

    #Initialize the model with the specified parameters
    model = OWFOMO(args, args.model_name, known_class_names, unknown_class_names,
                 templates, args.image_conditioned, device)

    #Initialize the post-processors with the specified parameters
    postprocessors = PostProcess(args.model_name, args.pred_per_im, args.image_resize, device, method=args.post_process_method)
    return model, postprocessors
