# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
path: python/lib/site-packages/torchvision/models/detection/
"""

from collections import OrderedDict
import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, partition=0, node='client', edge_out=None, input_features=None, input_proposals=None,
                input_proposal_losses=None, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        if partition == 0:  # no partition
            print("No partition")
            print("Running Backbone")
            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([('0', features)])
            print("Running rpn")
            proposals, proposal_losses = self.rpn(images, features, targets)
            print("Running roi_heads")
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
            print("Running postprocess")
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            if torch.jit.is_scripting():
                if not self._has_warned:
                    warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                    self._has_warned = True
                return (losses, detections)
            else:
                return self.eager_outputs(losses, detections)

        elif partition == 1:  # partition at resnet Layer1
            if node == 'client':
                print("Partition at resnet Layer1")
                edge_out = self.backbone(images.tensors, "layer1")
                return edge_out
            else:
                features = self.backbone(edge_out['0'], "layer1", edge_out)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([('0', features)])
                proposals, proposal_losses = self.rpn(images, features, targets)
                detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
                detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)

                if torch.jit.is_scripting():
                    if not self._has_warned:
                        warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                        self._has_warned = True
                    return (losses, detections)
                else:
                    return self.eager_outputs(losses, detections)

        elif partition == 2:  # partition at resnet Layer2
            if node == 'client':
                print("Partition at resnet Layer2")
                edge_out = self.backbone(images.tensors, "layer2")
                return edge_out
            else:
                features = self.backbone(edge_out['1'], "layer2", edge_out)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([('0', features)])
                proposals, proposal_losses = self.rpn(images, features, targets)
                detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
                detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)

                if torch.jit.is_scripting():
                    if not self._has_warned:
                        warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                        self._has_warned = True
                    return (losses, detections)
                else:
                    return self.eager_outputs(losses, detections)

        elif partition == 3:  # partition at resnet Layer3
            if node == 'client':
                print("Partition at resnet Layer3")
                edge_out = self.backbone(images.tensors, "layer3")
                return edge_out
            else:
                features = self.backbone(edge_out['2'], "layer3", edge_out)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([('0', features)])
                proposals, proposal_losses = self.rpn(images, features, targets)
                detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
                detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)

                if torch.jit.is_scripting():
                    if not self._has_warned:
                        warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                        self._has_warned = True
                    return (losses, detections)
                else:
                    return self.eager_outputs(losses, detections)

        elif partition == 4:  # partition at resnet Layer4
            if node == 'client':
                print("Partition at resnet Layer4")
                edge_out = self.backbone(images.tensors, "layer4")
                return edge_out
            else:
                features = self.backbone(edge_out['3'], "layer4", edge_out)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([('0', features)])
                proposals, proposal_losses = self.rpn(images, features, targets)
                detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
                detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)

                if torch.jit.is_scripting():
                    if not self._has_warned:
                        warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                        self._has_warned = True
                    return (losses, detections)
                else:
                    return self.eager_outputs(losses, detections)

        elif partition == 5:  # partition at backbone
            if node == 'client':
                print("Partition at backbone")
                features = self.backbone(images.tensors)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([('0', features)])
                return features
            else:
                proposals, proposal_losses = self.rpn(images, input_features, targets)
                detections, detector_losses = self.roi_heads(input_features, proposals, images.image_sizes, targets)
                detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)

                if torch.jit.is_scripting():
                    if not self._has_warned:
                        warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                        self._has_warned = True
                    return (losses, detections)
                else:
                    return self.eager_outputs(losses, detections)

        elif partition == 6:  # partition at rpn
            if node == 'client':
                print("Partition at rpn")
                features = self.backbone(images.tensors)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([('0', features)])
                proposals, proposal_losses = self.rpn(images, features, targets)
                return features, proposals, proposal_losses
            else:
                detections, detector_losses = self.roi_heads(input_features, input_proposals, images.image_sizes,
                                                             targets)
                detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                losses = {}
                losses.update(detector_losses)
                losses.update(input_proposal_losses)

                if torch.jit.is_scripting():
                    if not self._has_warned:
                        warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                        self._has_warned = True
                    return (losses, detections)
                else:
                    return self.eager_outputs(losses, detections)
