import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.modules.loss._Loss):
    def __init__(self, reduction:str, needs_logits:bool):
        """
        Initializes the base loss class.

        Args:
        - reduction (str): Reduction method for the loss term. Default: 'mean'.
        """
        super(BaseLoss, self).__init__()
        self.passed_test = None
        self.reduction = reduction
        self.needs_logits = needs_logits

    def _test_predictions(self, predictions:torch.Tensor):
        """
        If self.needs_logits: Test if the predictions are logits and not probabilities. Sum of across classes should not be 1. 
        If not self.needs_logits: Test if the predictions are probabilities and not logits. Sum of across classes should be 1

        Args:
        - predictions (torch.Tensor): Model (logits) predictions of shape (N, C, H, W), where:
            N: Batch size
            C: Number of classes (num_classes)
            H, W: Height and width of the prediction map
        """
        if self.passed_test is None:
           if torch.sum(predictions, dim=1).allclose(torch.ones([predictions.shape[0]]+list(predictions.shape[2:]), dtype=predictions.dtype), atol=1e-3):
                if self.needs_logits:
                    self.passed_test = False
                    raise ValueError("The predictions are probabilities and not logits. Please ensure that the model outputs logits.")
                else:
                    self.passed_test = True
           else:
                if self.needs_logits:
                    self.passed_test = True
                else:
                    self.passed_test = False
                    raise ValueError("The predictions are logits and not probabilities. Please ensure that the model outputs probabilities")

    def one_hot_encoding(self, targets:torch.Tensor, num_classes) -> torch.Tensor:
        """
        Converts the target segmentation map into one-hot encoded format for each class.

        Args:
        - targets (torch.Tensor): Ground truth segmentation map of shape (N, H, W).
        - num_classes (int): Number of classes.

        Returns:
        - one_hot (torch.Tensor): One-hot encoded masks of shape (N, C, H, W).
        """
        # N, H, W = targets.shape # Batch size, height, width
        # one_hot = torch.zeros(N, num_classes, H, W, device=targets.device)
        # one_hot.scatter_(1, targets.unsqueeze(1), 1)  # Set the corresponding class position to 1
        return F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()            
    

class Mask2FormerStyleLoss(BaseLoss):
    def __init__(self, num_classes:int, reduction:str='mean', loss_weights:dict={'loss_mask': 1.0, 'loss_dice': 1.0, 'loss_bce': 1.0}):
        """
        Initializes the loss function for CNN-based segmentation models using a Mask2Former-style approach.
        
        Args:
        - num_classes (int): Number of classes (excluding background).
        - reduction (str): Reduction method for the loss term. Default: 'mean'.
        - loss_weights (dict): Weights for each loss term. Default: {'loss_mask': 1.0, 'loss_dice': 1.0, 'loss_bce': 1.0}
        """
        super(Mask2FormerStyleLoss, self).__init__(reduction, needs_logits=True)
        self.num_classes = num_classes
        # check all the loss weights are present. If one is missing, add it with a default value of 1.0. If any extra loss weight is present, remove it.
        for loss_name in ['loss_mask', 'loss_dice', 'loss_bce']:
            if loss_name not in loss_weights:
                loss_weights[loss_name] = 1.0
        self.loss_weights = {k: v for k, v in loss_weights.items() if k in ['loss_mask', 'loss_dice', 'loss_bce']}    
               

    def forward(self, predictions:torch.Tensor, targets:torch.Tensor) -> dict:
        """
        Calculates the total loss given the model predictions and ground truth.

        Args:
        - predictions (torch.Tensor): Model (logits) predictions of shape (N, C, H, W), where:
            N: Batch size
            C: Number of classes (num_classes)
            H, W: Height and width of the prediction map
        - targets (torch.Tensor): Ground truth segmentation map of shape (N, H, W)

        Returns:
        - total_loss (torch.Tensor): The combined loss term.
        """
        #0. Make sure that predictions are logits and not probabilities. Sum of across classes should not be 1
        self._test_predictions(predictions.detach().cpu())
        probs = torch.softmax(predictions, dim=1)
        # 1. Calculate the per-pixel cross-entropy loss
        pixel_ce_loss = self.per_pixel_cross_entropy_loss(predictions, targets)

        # 2. Convert targets to one-hot encoding for each class
        targets_one_hot = self.one_hot_encoding(targets, predictions.shape[1])

        # 3. Calculate binary cross-entropy loss for each class mask
        bce_loss = self.binary_cross_entropy_loss(predictions, targets_one_hot)

        # 4. Calculate Dice loss for each class mask
        dice_loss = self.dice_loss(probs, targets_one_hot)

        # 5. Combine the losses using the given weights
        losses = {
            "loss_mask": pixel_ce_loss,
            "loss_dice": dice_loss,
            "loss_bce":  bce_loss}

        return losses
    
    def _reduce_entropy_loss(self, loss:torch.Tensor, reduction, axis):
        """
        Reduce the loss term based on the specified reduction method.

        Args:
        - loss (torch.Tensor): Loss term to be reduced.
        - reduction (str): Reduction method for the loss term.
        - axis (list): Dimensions to be reduced.

        Returns:
        - loss (torch.Tensor): Reduced loss term.
        """
        loss = loss.mean(axis)  # Average over the height and width dimensions
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction type for Cross-Entropy loss: {reduction}")

    def per_pixel_cross_entropy_loss(self, predictions, targets):
        """
        Per-pixel cross-entropy loss for the segmentation model.

        Args:
        - predictions (torch.Tensor): Model output of shape (N, C, H, W).
        - targets (torch.Tensor): Ground truth segmentation map of shape (N, H, W).

        Returns:
        - loss (torch.Tensor): Per-pixel cross-entropy loss.
        """
        ce = F.cross_entropy(predictions, targets, reduction='none')
        return self._reduce_entropy_loss(ce, self.reduction, axis=[-1, -2])

    def binary_cross_entropy_loss(self, predictions, targets_one_hot):
        """
        Binary Cross-Entropy loss for each class channel.

        Args:
        - predictions (torch.Tensor): Model logits of shape (N, C, H, W).
        - targets_one_hot (torch.Tensor): One-hot encoded ground truth masks of shape (N, C, H, W).

        Returns:
        - loss (torch.Tensor): Binary cross-entropy loss averaged over all classes.
        """

        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets_one_hot, reduction='none')
        return self._reduce_entropy_loss(bce_loss, self.reduction, axis=[-1, -2])


    def dice_loss(self, probs:torch.Tensor, targets_one_hot:torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        """
        Dice loss for each class channel.

        Args:
        - probs (torch.Tensor): Model predictions of shape (N, C, H, W).
        - targets_one_hot (torch.Tensor): One-hot encoded ground truth masks of shape (N, C, H, W).
        - epsilon (float): Small value to prevent division by zero.

        Returns:
        - loss (torch.Tensor): Dice loss averaged over all classes.
        """
        # softmax activation to get predicted probabilities
        # probs = torch.softmax(predictions, dim=1)

        # Compute intersection and union
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))  # Sum over H and W. Results in shape (N, C)
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))  # Sum over H and W. Results in shape (N, C)

        # Compute Dice Loss
        dice_loss = 1.0 - (2.0 * intersection + epsilon) / (union + epsilon)  # Dice loss for each class and sample

        # Average over classes for each sample
        dice_loss_per_sample = dice_loss.mean(dim=1)  # Shape: (N,) - Average over classes

        # Apply the appropriate reduction
        if self.reduction == 'mean':
            return dice_loss_per_sample.mean()  # Average over all samples
        elif self.reduction == 'sum':
            return dice_loss_per_sample.sum()  # Sum over all samples
        else:
            raise ValueError(f"Invalid reduction type for Dice loss: {self.reduction}")
        
    
    def __repr__(self):
        return f"Mask2FormerStyleLoss(num_classes={self.num_classes}, reduction={self.reduction}, loss_weights={self.loss_weights})"


class AsymUnifiedFocalLoss(BaseLoss):
    def __init__(self, 
                 loss_weights:dict={'asymmetric_ftl':0.5, 'asymmetric_fl':0.5}, 
                 delta=0.6, gamma=0.5, 
                 reduction='mean'):
        """
        Unified Focal Loss that combines Asymmetric Focal Tversky Loss and Asymmetric Focal Loss.

        Args:
        - loss_weights (dict): Represents the lambda parameter to control the weighting 
                          between Asymmetric Focal Tversky Loss and Asymmetric Focal Loss. Default is 0.5 for each.
        - delta (float): Controls the weight given to each class. Default is 0.6.
        - gamma (float): Focal parameter to control the degree of background suppression 
                         and foreground enhancement. Default is 0.5.
        - reduction (str): Reduction method for the loss term. Default: 'mean'.
        """
        super(AsymUnifiedFocalLoss, self).__init__(reduction, needs_logits=False)

        self.delta = delta
        self.gamma = gamma
        for loss_name in ['asymmetric_ftl', 'asymmetric_fl']:
            if loss_name not in loss_weights:
                loss_weights[loss_name] = 0.5
        self.loss_weights = {k: v for k, v in loss_weights.items() if k in ['asymmetric_ftl', 'asymmetric_fl']}


    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> dict:
        """
        Forward pass to compute the loss.

        Args:
        - y_pred (torch.Tensor): Predictions of shape (N, C, H, W).
        - y_true (torch.Tensor): Ground truth of shape (N, C, H, W).

        Returns:
        - loss (torch.Tensor): Computed unified focal loss.
        """
        self._test_predictions(y_pred.detach().cpu())   
        self.axis = [-1,-2] # Aggregate over the height and width dimensions

        # check if y_true is one-hot encoded. If not, convert it to one-hot encoding
        if y_true.shape[1] != 1:
            y_true = self.one_hot_encoding(y_true, y_pred.shape[1])

        if y_pred.shape[1] == 1:
            # safeguard for cases where only one class is present (foreground). 
            # Background is assumed to be the complement and added to the tensor
            y_pred = torch.cat((1-y_pred, y_pred), dim=1)
            y_true = torch.cat((1-y_true, y_true), dim=1)

        # Calculate Asymmetric Focal Tversky Loss
        asymmetric_ftl = self.asymmetric_focal_tversky_loss(y_pred, y_true)

        # Calculate Asymmetric Focal Loss
        asymmetric_fl = self.asymmetric_focal_loss(y_pred, y_true)

        # Combine the losses using the specified weight
        losses = {
            "asymmetric_ftl": asymmetric_ftl,
            "asymmetric_fl": asymmetric_fl}

        return losses
    
    def __repr__(self):
        return f"AsymUnifiedFocalLoss(delta={self.delta}, gamma={self.gamma}, reduction={self.reduction}, loss_weights={self.loss_weights})"

    def asymmetric_focal_tversky_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor, epsilon:float=1e-8) -> torch.Tensor:
        """
        Calculate Asymmetric Focal Tversky Loss.

        Args:
        - y_pred (torch.Tensor): Predictions of shape (N, C, H, W).
        - y_true (torch.Tensor): Ground truth of shape (N, C, H, W).
        - epsilon (float): Small constant to avoid division by zero.

        Returns:
        - loss (torch.Tensor): Calculated Tversky loss.
        """
        # Clip predictions to prevent division by zero
        y_pred = torch.clamp(y_pred, min=epsilon, max=1.0 - epsilon)

        # Calculate true positives, false negatives, and false positives
        tp = torch.sum(y_true * y_pred, dim=self.axis)
        fn = torch.sum(y_true * (1 - y_pred), dim=self.axis)
        fp = torch.sum((1 - y_true) * y_pred, dim=self.axis)

        # Tversky index (equivalent to Dice coefficient)
        tversky_index = (tp + epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_tversky = (1-tversky_index[:,:1])
        fore_tversky = (1-tversky_index[:,1:]) * torch.pow(1-tversky_index[:,1:], - self.gamma)
        
        # Average class scores
        loss = torch.concat([back_tversky, fore_tversky], dim=1)
        loss = loss.mean(1) 

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':    
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction type for Tversky loss: {self.reduction}")
    

    def asymmetric_focal_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
        """
        Calculate Asymmetric Focal Loss.

        Args:
        - y_pred (torch.Tensor): Predictions of shape (N, C, H, W).
        - y_true (torch.Tensor): Ground truth of shape (N, C, H, W).

        Returns:
        - loss (torch.Tensor): Calculated asymmetric focal loss.
        """
        # Clip predictions to prevent division by zero
        epsilon = 10e-8
        y_pred = torch.clamp(y_pred, min=epsilon, max=1.0 - epsilon)

        # Calculate cross-entropy
        cross_entropy = -y_true * torch.log(y_pred)

        # Separate loss calculations for background and foreground
        back_ce = torch.pow(1 - y_pred[:, :1,...], self.gamma) * cross_entropy[:, :1,...]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:, 1:,...]  # Only foreground
        fore_ce = self.delta * fore_ce.sum(1, keepdim=True)

        # Combine background and foreground losses
        loss = torch.cat([back_ce, fore_ce], dim=1)
        loss = loss.sum(1).mean(self.axis)  # Mean over the H and W dimensions

        return loss.mean() if self.reduction == 'mean' else loss.mean()
    

